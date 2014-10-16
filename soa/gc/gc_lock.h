/* gc_lock.h                                                       -*- C++ -*-
   Jeremy Barnes, 19 November 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   "Lock" that works by deferring the destruction of objects into a garbage collection
   process which is only run when nothing could be using them.
*/

#ifndef __mmap__gc_lock_h__
#define __mmap__gc_lock_h__

#define GC_LOCK_DEBUG 0

#include "jml/utils/exc_assert.h"
#include "jml/arch/atomic_ops.h"
#include "jml/arch/thread_specific.h"
#include <vector>
#include <iostream>

#if GC_LOCK_DEBUG
#  include <iostream>
#endif

/** Deterministic memory reclamation is a fundamental problem in lock-free
    algorithms and data structures.

    When concurrently updating a lock-free data structure, it is not safe to
    immediately reclaim the old value since a bazillon other threads might still
    hold a reference to the old value. What we need is a safe memory reclamation
    mechanism. That is why GcLock exists. GcLock works by deferring the
    destruction of an object until it is safe to do; when the system decides
    that nobody still holds a reference to it.

    GcLock works by defining "critical sections" that a thread should hold to
    safely read a shared object.

    Note that this class contains many types of critical sections which are all
    specialized for various situations:

    - Shared CS: Regular read side critical sections.
    - Exclusive CS: Acts as the write op in a read-write lock with the shared CS
    - Speculative CS: Optimization of Shared CS whereby the CS may or may not be
      unlocked when requested to save on repeated accesses to the CS.

    Further details is available in the documentation of each respective
    operand.

*/

namespace Datacratic {

extern int32_t SpeculativeThreshold;

/*****************************************************************************/
/* GC LOCK BASE                                                              */
/*****************************************************************************/

struct GcLockBase : public boost::noncopyable {

public:

    /** Enum for type safe specification of whether or not we run deferrals on
        entry or exit to a critical sections.  Thoss places that are latency
        sensitive should use RD_NO.
    */
    enum RunDefer {
        RD_NO = 0,      ///< Don't run deferred work on this call
        RD_YES = 1      ///< Potentially run deferred work on this call
    };

    /// A thread's bookkeeping info about each GC area
    struct ThreadGcInfoEntry {
        ThreadGcInfoEntry()
            : inEpoch(-1), readLocked(0), writeLocked(0),
              specLocked(0), specUnlocked(0),
              owner(0)
        {
        }

        ~ThreadGcInfoEntry() {
            /* We are not in a speculative critical section, check if
             * Gc has been left locked
             */
            if (!specLocked && !specUnlocked && (readLocked || writeLocked))
                ExcCheck(false, "Thread died but GcLock is still locked");

            /* We are in a speculative CS but Gc has not beed unlocked
             */
            else if (!specLocked && specUnlocked) {
                unlockShared(RD_YES);
                specUnlocked = 0;
            }
           
        } 


        int inEpoch;  // 0, 1, -1 = not in 
        int readLocked;
        int writeLocked;

        int specLocked;
        int specUnlocked;

        GcLockBase *owner;

        void init(const GcLockBase * const self) {
            if (!owner) 
                owner = const_cast<GcLockBase *>(self);
        }
                

        void lockShared(RunDefer runDefer) {
            if (!readLocked && !writeLocked)
                owner->enterCS(this, runDefer);

            ++readLocked;
        }

        void unlockShared(RunDefer runDefer) {
            if (readLocked <= 0)
                throw ML::Exception("Bad read lock nesting");

            --readLocked;
            if (!readLocked && !writeLocked) 
                owner->exitCS(this, runDefer);
        }

        bool isLockedShared() {
            return readLocked + writeLocked;
        }

        void lockExclusive() {
            if (!writeLocked)
                owner->enterCSExclusive(this);
            
             ++writeLocked;
        }

        void unlockExclusive() {
            if (writeLocked <= 0)
                throw ML::Exception("Bad write lock nesting");

            --writeLocked;
            if (!writeLocked)
                owner->exitCSExclusive(this);
        }

        void lockSpeculative(RunDefer runDefer) {
            if (!specLocked && !specUnlocked) 
                lockShared(runDefer);

            ++specLocked;
        }

        void unlockSpeculative(RunDefer runDefer) {
            if (!specLocked) 
                throw ML::Exception("Bad speculative lock nesting");

            --specLocked;
            if (!specLocked) {
                if (++specUnlocked == SpeculativeThreshold) {
                    unlockShared(runDefer);
                    specUnlocked = 0;
                }
            }
        }

        void forceUnlock(RunDefer runDefer) {
            ExcCheckEqual(specLocked, 0, "Bad forceUnlock call");

            if (specUnlocked) {
                unlockShared(runDefer);
                specUnlocked = 0;
            }
        }


        std::string print() const;
    };

    typedef ML::ThreadSpecificInstanceInfo<ThreadGcInfoEntry, GcLockBase>
        GcInfo;
    typedef typename GcInfo::PerThreadInfo ThreadGcInfo;

    struct Data {
        Data();
        Data(const Data & other);

        Data & operator = (const Data & other);

        typedef uint64_t q2 __attribute__((__vector_size__(16)));
        
        volatile union {
            struct {
                int32_t epoch;       ///< Current epoch number (could be smaller).
                int16_t in[2];       ///< How many threads in each epoch
                int32_t visibleEpoch;///< Lowest epoch number that's visible
                int32_t exclusive;   ///< Mutex value to lock exclusively
            };
            struct {
                uint64_t bits;
                uint64_t bits2;
            };
            struct {
                q2 q;
            };
        } JML_ALIGNED(16);

        int16_t inCurrent() const { return in[epoch & 1]; }
        int16_t inOld() const { return in[(epoch - 1)&1]; }

        void setIn(int32_t epoch, int val)
        {
            //if (epoch != this->epoch && epoch + 1 != this->epoch)
            //    throw ML::Exception("modifying wrong epoch");
            in[epoch & 1] = val;
        }

        void addIn(int32_t epoch, int val)
        {
            //if (epoch != this->epoch && epoch + 1 != this->epoch)
            //    throw ML::Exception("modifying wrong epoch");
            in[epoch & 1] += val;
        }

        /** Check that the invariants all hold.  Throws an exception if not. */
        void validate() const;

        /** Calculate the appropriate value of visibleEpoch from the rest
            of the fields.  Returns true if waiters should be woken up.
        */
        bool calcVisibleEpoch();
        
        /** Human readable string. */
        std::string print() const;

        bool operator == (const Data & other) const
        {
            return bits == other.bits && bits2 == other.bits2;
        }

        bool operator != (const Data & other) const
        {
            return ! operator == (other);
        }

    } JML_ALIGNED(16);


    void enterCS(ThreadGcInfoEntry * entry = 0, RunDefer runDefer = RD_YES);
    void exitCS(ThreadGcInfoEntry * entry = 0, RunDefer runDefer = RD_YES);
    void enterCSExclusive(ThreadGcInfoEntry * entry = 0);
    void exitCSExclusive(ThreadGcInfoEntry * entry = 0);

    int myEpoch(GcInfo::PerThreadInfo * threadInfo = 0) const
    {
        return getEntry(threadInfo).inEpoch;
    }

    int currentEpoch() const
    {
        return data->epoch;
    }

    JML_ALWAYS_INLINE ThreadGcInfoEntry &
    getEntry(GcInfo::PerThreadInfo * info = 0) const
    {
        ThreadGcInfoEntry *entry = gcInfo.get(info);
        entry->init(this);
        return *entry;

        //return *gcInfo.get(info);
    }

    GcLockBase();

    virtual ~GcLockBase();

    /** Permanently deletes any resources associated with this lock. */
    virtual void unlink() = 0;

    void lockShared(GcInfo::PerThreadInfo * info = 0,
                    RunDefer runDefer = RD_YES)
    {
        ThreadGcInfoEntry & entry = getEntry(info);

        entry.lockShared(runDefer);

#if GC_LOCK_DEBUG
        using namespace std;
        cerr << "lockShared "
             << this << " index " << index
             << ": now " << entry.print() << " data "
             << data->print() << endl;
#endif
    }

    void unlockShared(GcInfo::PerThreadInfo * info = 0, 
                      RunDefer runDefer = RD_YES)
    {
        ThreadGcInfoEntry & entry = getEntry(info);

        entry.unlockShared(runDefer);

#if GC_LOCK_DEBUG
        using namespace std;
        cerr << "unlockShared "
             << this << " index " << index
             << ": now " << entry.print() << " data "
             << data->print() << endl;
#endif
    }

    /** Speculative critical sections should be used for hot loops doing
        repeated but short reads on shared objects where it's acceptable to keep
        hold of the section in between read operations because you're likely to
        need it again soon.

        This is an optimization of lockShared since it can avoid repeated entry
        and exit of the CS when it's likely to be reused shortly after. It also
        has the effect of heavily reducing contention on the lock under heavy
        contention scenarios.

        Usage example:

            GcLock gc;
            for (condition) {
                gc.enterSpeculative();
                // In critical section

                gc.exitSpeculative();
                // After the call, gc might or might not be unlocked
            }

            gc.forceUnlock();

        Note the call to forceUnlock() after the loop which ensure that we've
        exited the critical section. Also note that the speculative functions
        are called directly in this example for illustrative purposes. In actual
        code, use the SpeculativeGuard class which provides full RAII
        guarantees.
    */
    void lockSpeculative(GcInfo::PerThreadInfo * info = 0,
                         RunDefer runDefer = RD_YES)
    {
        ThreadGcInfoEntry & entry = getEntry(info); 

        entry.lockSpeculative(runDefer);
    }

    void unlockSpeculative(GcInfo::PerThreadInfo * info = 0,
                           RunDefer runDefer = RD_YES)
    {
        ThreadGcInfoEntry & entry = getEntry(info);

        entry.unlockSpeculative(runDefer);
    }

    /** Ensures that after the call, the Gc is "unlocked".

        This should be used in conjunction with the speculative lock to notify
        the Gc Lock to exit any leftover speculative sections for the current
        thread. If multiple threads can hold a speculative region, this function
        has to be called in each thread respectively. Note that it will be
        called automatically when a thread is destroyed.
     */
    void forceUnlock(GcInfo::PerThreadInfo * info = 0,
                     RunDefer runDefer = RD_YES) {
        ThreadGcInfoEntry & entry = getEntry(info);

        entry.forceUnlock(runDefer);
    }
        
    int isLockedShared(GcInfo::PerThreadInfo * info = 0) const
    {
        ThreadGcInfoEntry & entry = getEntry(info);

        return entry.isLockedShared();
    }

    int lockedInEpoch(GcInfo::PerThreadInfo * info = 0) const
    {
        ThreadGcInfoEntry & entry = getEntry(info);

        return entry.inEpoch;
    }

    void lockExclusive(GcInfo::PerThreadInfo * info = 0)
    {
        ThreadGcInfoEntry & entry = getEntry(info);

        entry.lockExclusive();
#if GC_LOCK_DEBUG
        using namespace std;
        cerr << "lockExclusive "
             << this << " index " << index
             << ": now " << entry.print() << " data "
             << data->print() << endl;
#endif
    }

    void unlockExclusive(GcInfo::PerThreadInfo * info = 0)
    {
        ThreadGcInfoEntry & entry = getEntry(info);

        entry.unlockExclusive();

#if GC_LOCK_DEBUG
        using namespace std;
        cerr << "unlockExclusive"
             << this << " index " << index
             << ": now " << entry.print()
             << " data " << data->print() << endl;
#endif
    }

    int isLockedExclusive(GcInfo::PerThreadInfo * info = 0) const
    {
        ThreadGcInfoEntry & entry = getEntry(info);

        return entry.writeLocked;
    }

    enum DoLock {
        DONT_LOCK = 0,
        DO_LOCK = 1
    };

    struct SharedGuard {
        SharedGuard(GcLockBase & lock,
                    RunDefer runDefer = RD_YES,
                    DoLock doLock = DO_LOCK)
            : lock_(lock),
              runDefer_(runDefer),
              doLock_(doLock)
        {
            if (doLock_)
                lock_.lockShared(0, runDefer_);
        }

        ~SharedGuard()
        {
            if (doLock_)
                lock_.unlockShared(0, runDefer_);
        }
        
        void lock()
        {
            if (doLock_)
                return;
            lock_.lockShared(0, runDefer_);
            doLock_ = DO_LOCK;
        }

        void unlock()
        {
            if (!doLock_)
                return;
            lock_.unlockShared(0, runDefer_);
            doLock_ = DONT_LOCK;
        }

        GcLockBase & lock_;
        const RunDefer runDefer_;  ///< Can this do deferred work?
        DoLock doLock_;      ///< Do we really lock?
    };

    struct ExclusiveGuard {
        ExclusiveGuard(GcLockBase & lock)
            : lock(lock)
        {
            lock.lockExclusive();
        }

        ~ExclusiveGuard()
        {
            lock.unlockExclusive();
        }

        GcLockBase & lock;
    };

    struct SpeculativeGuard {
        SpeculativeGuard(GcLockBase &lock,
                         RunDefer runDefer = RD_YES) :
            lock(lock),
            runDefer_(runDefer) 
        {
            lock.lockSpeculative(0, runDefer_);
        }

        ~SpeculativeGuard() 
        {
            lock.unlockSpeculative(0, runDefer_);
        }

        GcLockBase & lock;
        const RunDefer runDefer_;
    };


    /** Wait until everything that's currently visible is no longer
        accessible.
        
        You can't call this if a guard is held, as it would deadlock waiting
        for itself to exit from the critical section.
    */
    void visibleBarrier();

    /** Wait until all defer functions that have been registered have been
        run.
    
        You can't call this if a guard is held, as it would deadlock waiting
        for itself to exit from the critical section.
    */
    void deferBarrier();

    void defer(boost::function<void ()> work);

    typedef void (WorkFn1) (void *);
    typedef void (WorkFn2) (void *, void *);
    typedef void (WorkFn3) (void *, void *, void *);

    void defer(void (work) (void *), void * arg);
    void defer(void (work) (void *, void *), void * arg1, void * arg2);
    void defer(void (work) (void *, void *, void *), void * arg1, void * arg2, void * arg3);

    template<typename T>
    void defer(void (*work) (T *), T * arg)
    {
        defer((WorkFn1 *)work, (void *)arg);
    }

    template<typename T>
    static void doDelete(T * arg)
    {
        delete arg;
    }

    template<typename T>
    void deferDelete(T * toDelete)
    {
        if (!toDelete) return;
        defer(doDelete<T>, toDelete);
    }

    template<typename... Args>
    void doDefer(void (fn) (Args...), Args...);

    template<typename Fn, typename... Args>
    void deferBind(Fn fn, Args... args)
    {
        boost::function<void ()> bound = boost::bind<void>(fn, args...);
        this->defer(bound);
    }

    void dump();

protected:
    Data* data;

private:
    struct Deferred;
    struct DeferredList;

    GcInfo gcInfo;

    Deferred * deferred;   ///< Deferred workloads (hidden structure)

    /** Update with the new value after first checking that the current
        value is the same as the old value.  Returns true if it
        succeeded; otherwise oldValue is updated with the new old
        value.

        As part of doing this, it will calculate the correct value for
        visibleEpoch() and, if it has changed, wake up anything waiting
        on that value, and will run any deferred handlers registered for
        that value.
    */
    bool updateData(Data & oldValue, Data & newValue, RunDefer runDefer);

    /** Executes any available deferred work. */
    void runDefers();

    /** Check what deferred updates need to be run and do them.  Must be
        called with deferred locked.
    */
    std::vector<DeferredList *> checkDefers();
};


/*****************************************************************************/
/* GC LOCK                                                                   */
/*****************************************************************************/

/** GcLock for use within a single process. */

struct GcLock : public GcLockBase
{
    GcLock();
    virtual ~GcLock();

    virtual void unlink();

private:

    Data localData;

};


/*****************************************************************************/
/* SHARED GC LOCK                                                            */
/*****************************************************************************/

/** Constants that can be used to control how resources are opened.
    Note that these are structs so we can more easily overload constructors.
*/
extern struct GcCreate {} GC_CREATE; ///< Open and initialize a new resource.
extern struct GcOpen {} GC_OPEN;     ///< Open an existing resource.


/** GcLock to be shared among multiple processes. */

struct SharedGcLock : public GcLockBase
{
    SharedGcLock(GcCreate, const std::string& name);
    SharedGcLock(GcOpen, const std::string& name);
    virtual ~SharedGcLock();

    /** Permanently deletes any resources associated with the gc lock. */
    virtual void unlink();

private:

    /** mmap an shm file into memory and set the data member of GcLock. */
    void doOpen(bool create);

    std::string name;
    int fd;
    void* addr;

};

} // namespace Datacratic

#endif /* __mmap__gc_lock_h__ */


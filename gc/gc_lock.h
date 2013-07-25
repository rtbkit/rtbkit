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

#if GC_LOCK_DEBUG
#  include <iostream>
#endif

namespace Datacratic {


/*****************************************************************************/
/* GC LOCK BASE                                                              */
/*****************************************************************************/

struct GcLockBase : public boost::noncopyable {

    struct Deferred;
    struct DeferredList;

    /// A thread's bookkeeping info about each GC area
    struct ThreadGcInfoEntry {
        ThreadGcInfoEntry()
            : inEpoch(-1), readLocked(0), writeLocked(0)
        {
        }

        int inEpoch;  // 0, 1, -1 = not in 
        int readLocked;
        int writeLocked;

        std::string print() const;
    };

    typedef ML::ThreadSpecificInstanceInfo<ThreadGcInfoEntry, GcLockBase>
        GcInfo;
    typedef typename GcInfo::PerThreadInfo ThreadGcInfo;

    GcInfo gcInfo;

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

    Data* data;

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
    bool updateData(Data & oldValue, Data & newValue, bool runDefer = true);

    /** Executes any available deferred work. */
    void runDefers();

    /** Check what deferred updates need to be run and do them.  Must be
        called with deferred locked.
    */
    std::vector<DeferredList *> checkDefers();

    void enterCS(ThreadGcInfoEntry * entry = 0, bool runDefer = true);
    void exitCS(ThreadGcInfoEntry * entry = 0, bool runDefer = true);
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
        return *gcInfo.get(info);
    }

    GcLockBase();

    virtual ~GcLockBase();

    /** Permanently deletes any resources associated with this lock. */
    virtual void unlink() = 0;

    void lockShared(GcInfo::PerThreadInfo * info = 0,
                    bool runDefer = true)
    {
        ThreadGcInfoEntry & entry = getEntry(info);

        if (!entry.readLocked && !entry.writeLocked)
            enterCS(&entry, runDefer);

        ++entry.readLocked;

#if GC_LOCK_DEBUG
        using namespace std;
        cerr << "lockShared "
             << this << " index " << index
             << ": now " << entry.print() << " data "
             << data->print() << endl;
#endif
    }

    void unlockShared(GcInfo::PerThreadInfo * info = 0, 
                      bool runDefer = true)
    {
        ThreadGcInfoEntry & entry = getEntry(info);

        if (entry.readLocked <= 0)
            throw ML::Exception("bad read lock nesting");
        --entry.readLocked;
        if (!entry.readLocked && !entry.writeLocked)
            exitCS(&entry, runDefer);

#if GC_LOCK_DEBUG
        using namespace std;
        cerr << "unlockShared "
             << this << " index " << index
             << ": now " << entry.print() << " data "
             << data->print() << endl;
#endif
    }

    int isLockedShared(GcInfo::PerThreadInfo * info = 0) const
    {
        ThreadGcInfoEntry & entry = getEntry(info);
        return entry.readLocked + entry.writeLocked;
    }

    int lockedInEpoch(GcInfo::PerThreadInfo * info = 0) const
    {
        ThreadGcInfoEntry & entry = getEntry(info);
        return entry.inEpoch;
    }

    void lockExclusive(GcInfo::PerThreadInfo * info = 0)
    {
        ThreadGcInfoEntry & entry = getEntry(info);

        if (entry.readLocked)
            throw ML::Exception("can't acquire write lock with read lock held");

        if (!entry.writeLocked)
            enterCSExclusive(&entry);

        ++entry.writeLocked;

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

        if (entry.writeLocked <= 0)
            throw ML::Exception("bad write lock nesting");
        --entry.writeLocked;
        if (!entry.writeLocked)
            exitCSExclusive(&entry);

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

    struct SharedGuard {
        SharedGuard(GcLockBase & lock, bool runDefer = true)
            : lock(lock),
              runDefer_(runDefer)
        {
            lock.lockShared(0, runDefer_);
        }

        ~SharedGuard()
        {
            lock.unlockShared(0, runDefer_);
        }
        
        GcLockBase & lock;
        const bool runDefer_;
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


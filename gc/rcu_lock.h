/* rcu_lock.h                                                      -*- C++ -*-
   Jeremy Barnes, 20 November 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Garbage collection lock using userspace RCU.
*/

#ifndef __mmap__rcu_lock_h__
#define __mmap__rcu_lock_h__

#include <urcu.h>
#include <vector>
#include <boost/function.hpp>
#include "jml/compiler/compiler.h"
#include "jml/utils/exc_assert.h"
#include "jml/arch/rwlock.h"
#include "jml/arch/thread_specific.h"

namespace Datacratic {

struct RcuLock {

    /// A thread's bookkeeping info about each GC area
    struct ThreadGcInfoEntry {
        ThreadGcInfoEntry()
            : readLocked(0), lockedExclusive(false), writeLocked(0)
        {
        }

        int readLocked;
        bool lockedExclusive;
        int writeLocked;
    };

    /// A thread's overall bookkeeping information
    struct ThreadGcInfo {
        ThreadGcInfo()
        {
            static int threadInfoNum = 0;
            threadNum = __sync_fetch_and_add(&threadInfoNum, 1);
            rcu_register_thread();
            rcu_defer_register_thread();
        }

        ~ThreadGcInfo()
        {
            rcu_unregister_thread();
            rcu_defer_unregister_thread();
        }

        int threadNum;

        std::vector<ThreadGcInfoEntry> info;

        ThreadGcInfoEntry & operator [] (int index)
        {
            ExcAssertGreaterEqual(index, 0);
            if (info.size() <= index)
                info.resize(index + 1);
            return info[index];
        }
    };

    static ML::Thread_Specific<ThreadGcInfo> gcInfo;  ///< Thread-specific bookkeeping
    bool exclusiveMode;  ///< If true we fall back to a RW mutex
    ML::RWLock exclusionMutex;  ///< The RW mutex for exclusive mode
    int index;     ///< Which global thread number we are

    void enterCS()
    {
        rcu_read_lock();
    }

    void exitCS()
    {
        rcu_read_unlock();
    }

    int myEpoch() const
    {
        return -1;
    }

    int currentEpoch() const
    {
        return -1;
    }

    JML_ALWAYS_INLINE ThreadGcInfoEntry & getEntry() const
    {
        ThreadGcInfo & info = *gcInfo;
        return info[index];
    }

    static int currentIndex;

    RcuLock()
        : exclusiveMode(false),
          index(currentIndex + 1)
    {
    }

    void lockShared()
    {
        ThreadGcInfoEntry & entry = getEntry();

        //cerr << "lockShared: readLocked " << entry.readLocked
        //     << " writeLocked: " << entry.writeLocked
        //     << " exclusive " << exclusiveMode << endl;

        while (!entry.readLocked && !entry.writeLocked) {
            entry.lockedExclusive = exclusiveMode;

            // If something else wanted an exclusive lock then we are in
            // exclusive mode and we have to acquire the RW lock beore we
            // can continue
            if (JML_UNLIKELY(entry.lockedExclusive))
                exclusionMutex.lock_shared();

            //cerr << "entering" << endl;
            enterCS();

            // Avoid racing with the update of the exlusive lock...
            // TODO: this needs to be well tested
            if (entry.lockedExclusive != exclusiveMode) {
                //cerr << "reexiting" << endl;
                exitCS();
                continue;
            }
            
            break;
        }

        ++entry.readLocked;
    }

    void unlockShared()
    {
        ThreadGcInfoEntry & entry = getEntry();

        if (entry.readLocked <= 0)
            throw ML::Exception("bad read lock nesting");
        --entry.readLocked;
        if (!entry.readLocked && !entry.writeLocked) {
            if (entry.lockedExclusive) exclusionMutex.unlock_shared();
            exitCS();
        }
    }

    bool isLockedShared()
    {
        ThreadGcInfoEntry & entry = getEntry();
        return entry.readLocked || entry.writeLocked;
    }

    void lockExclusive()
    {
        ThreadGcInfoEntry & entry = getEntry();

        if (entry.readLocked)
            throw ML::Exception("can't acquire write lock with read lock held");

        if (!entry.writeLocked) {
            exclusionMutex.lock();
            //cerr << "entering exclusive mode numInRead = " << numInRead << endl;
            ExcAssert(!exclusiveMode);
            exclusiveMode = true;
            //ML::memory_barrier();
            visibleBarrier();

            //ExcAssertEqual(numInRead, 0);
        }

        ++entry.writeLocked;
    }

    void unlockExclusive()
    {
        ThreadGcInfoEntry & entry = getEntry();

        if (entry.writeLocked <= 0)
            throw ML::Exception("bad write lock nesting");
        --entry.writeLocked;
        if (!entry.writeLocked) {
            exclusiveMode = false;
            exclusionMutex.unlock();
        }
    }

    bool isLockedExclusive()
    {
        ThreadGcInfoEntry & entry = getEntry();
        return entry.writeLocked;
    }

    void visibleBarrier()
    {
        synchronize_rcu();
    }

    void deferBarrier()
    {
        rcu_defer_barrier();
    }

    static void callFn(void * arg)
    {
        boost::function<void ()> * fn
            = reinterpret_cast<boost::function<void ()> *>(arg);
        try {
            (*fn)();
        } catch (...) {
            delete fn;
            throw;
        }
        delete fn;
    }

    void defer(boost::function<void ()> work)
    {
        defer(callFn, new boost::function<void ()>(work));
    }

    void defer(void (work) (void *), void * arg)
    {
        getEntry();  // make sure thread is registered
        defer_rcu(work, arg);
    }

    void defer(void (work) (void *, void *), void * arg1, void * arg2)
    {
        defer([=] () { work(arg1, arg2); });
    }

    void defer(void (work) (void *, void *, void *), void * arg1, void * arg2, void * arg3)
    {
        defer([=] () { work(arg1, arg2, arg3); });
    }

    void dump()
    {
    }
};

} // namespace Datacratic


#endif /* __mmap__rcu_lock_h__ */

/* gc.cc
   Jeremy Barnes, 26 September 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

*/

#include "gc.h"
#include <urcu.h>
#include <boost/thread.hpp>
#include <iostream>
#include "jml/utils/exc_assert.h"
#include "jml/arch/atomic_ops.h"
#include "jml/arch/rwlock.h"
#include "jml/arch/thread_specific.h"

using namespace std;


namespace Datacratic {
namespace MMap {


struct doInit {
    doInit()
    {
        rcu_init();
    }
} init;

void registerThread()
{
    rcu_register_thread();
    rcu_defer_register_thread();
}

void unregisterThread()
{
    rcu_defer_unregister_thread();
    rcu_unregister_thread();
}

/** If true, then we are in exclusive mode which means that only one thing
    can hold a lock at once. */
static volatile bool exclusiveMode = false;

/** The mutex for exclusive mode. */
//static boost::shared_mutex exclusionMutex;
static ML::RWLock exclusionMutex;

#if 0

/** Thread-specific data for the gc functionality.  Boost's thread specific data
    class has internal locking which makes it unsuitable.  We use gcc's support
    for thread specific data (which is much faster), with Boost's stuff there
    simply to make sure that everything gets cleaned up when each thread exits.
*/

struct ThreadData;

struct ThreadDataDeleter {
    ThreadDataDeleter(ThreadData * data)
        : data(data)
    {
    }

    ThreadData * data;

    ~ThreadDataDeleter();
};

boost::thread_specific_ptr<ThreadDataDeleter> threadDataDeleter;

#endif

struct ThreadData {

    static int numInRead;

    ThreadData()
        : readLocked(0), writeLocked(0)
    {
        lockedExclusive = false;
        registerThread();
        initialized = true;
        threadNum = random();

        //threadDataDeleter.reset(new ThreadDataDeleter(this));

        //cerr << "initialized thread " << threadNum << " at " << this << endl;
    }

    ~ThreadData()
    {
        //cerr << "destroyThread " << threadNum << " at " << this << endl;

        if (writeLocked) {
            cerr << "warning: thread exiting with write lock held" << endl;
        }

        if (readLocked) {
            cerr << "warning: thread exiting with read lock held" << endl;
        }
        
        if (initialized)
            unregisterThread();
        initialized = false;
    }
    
    void readLock()
    {
        //cerr << "readLock" << " " << this << " readLocked " << readLocked << " writeLocked "
        //     << writeLocked << " excl " << exclusiveMode << " lexcl " << lockedExclusive
        //     << endl;
        while (!readLocked && !writeLocked) {
            lockedExclusive = exclusiveMode;

            if (lockedExclusive)
                exclusionMutex.lock_shared();

            rcu_read_lock();  // this also does a memory barrier...

            // Avoid racing with the update of the exlusive lock...
            // TODO: this needs to be well tested
            if (lockedExclusive != exclusiveMode) {
                rcu_read_unlock();
                continue;
            }

            //ML::atomic_inc(numInRead);

            break;
        }

        ++readLocked;

    }

    bool isReadLocked()
    {
        return readLocked || writeLocked;
    }
    
    void readUnlock()
    {
        //cerr << "readUnlock" << " " << this << " readLocked " << readLocked << " writeLocked "
        //     << writeLocked << " excl " << exclusiveMode << " lexcl " << lockedExclusive
        //     << endl;

        if (readLocked <= 0)
            throw ML::Exception("bad read lock nesting");
        --readLocked;
        if (!readLocked && !writeLocked) {
            if (lockedExclusive) exclusionMutex.unlock_shared();
            //ML::atomic_dec(numInRead);
            rcu_read_unlock();
        }
    }
    
    void writeLock()
    {
        //cerr << "writeLock" << " " << this << " readLocked " << readLocked << " writeLocked "
        //     << writeLocked << " excl " << exclusiveMode << " lexcl " << lockedExclusive
        //     << endl;
        if (readLocked)
            throw ML::Exception("can't acquire write lock with read lock held");

        if (!writeLocked) {
            exclusionMutex.lock();
            //cerr << "entering exclusive mode numInRead = " << numInRead << endl;
            ExcAssert(!exclusiveMode);
            exclusiveMode = true;
            //ML::memory_barrier();
            synchronize_rcu();  // wait for all readers to stop and block on lock

            //ExcAssertEqual(numInRead, 0);
        }

        ++writeLocked;
    }

    void writeUnlock()
    {
        //cerr << "writeUnlock" << " " << this << " readLocked " << readLocked << " writeLocked "
        //     << writeLocked << " excl " << exclusiveMode << " lexcl " << lockedExclusive
        //     << endl;

        if (writeLocked <= 0)
            throw ML::Exception("bad write lock nesting");
        --writeLocked;
        if (!writeLocked) {
            exclusiveMode = false;
            exclusionMutex.unlock();
        }
    }

    bool isWriteLocked()
    {
        return writeLocked;
    }

    void stopTheWorld()
    {
        writeLock();
        rcu_defer_barrier();
    }
    
    void restartTheWorld()
    {
        writeUnlock();
    }

    bool initialized;
    int readLocked;
    bool lockedExclusive;
    int writeLocked;
    int threadNum;

    pthread_key_t tssKey;
};

int ThreadData::numInRead = 0;

#if 0
ThreadDataDeleter::
~ThreadDataDeleter()
{
    delete data;
}

// Will be leaked...
 __thread ThreadData * threadData = 0;


#endif

ML::Thread_Specific<ThreadData> threadData;

ThreadData & getThreadData()
{
    return *threadData;

#if 0
    if (!threadData.get())
        threadData.reset(new ThreadData());
    return *threadData;
#endif
}

int getThreadNum()
{
    return getThreadData().threadNum;
}

void readLock()
{
    getThreadData().readLock();
}

void readUnlock()
{
    getThreadData().readUnlock();
}

bool isReadLocked()
{
    return getThreadData().isReadLocked();
}

void waitForGC()
{
    rcu_defer_barrier();
}

// These are stubs at the moment but will have to become something better...

void writeLock()
{
    getThreadData().writeLock();
}

void writeUnlock()
{
    getThreadData().writeUnlock();
}

bool isWriteLocked()
{
    return getThreadData().isWriteLocked();
}

void stopTheWorld()
{
    getThreadData().stopTheWorld();
}

void restartTheWorld()
{
    getThreadData().restartTheWorld();
}

void doDeferredGc(void * workBlock)
{
    boost::function<void ()> * work
        = reinterpret_cast<boost::function<void ()> *>(workBlock);
    try {
        (*work)();
    } catch (...) {
        delete work;
        throw;
    }
    
    delete work;
}

void deferGC(const boost::function<void ()> & work)
{
    defer_rcu(&doDeferredGc,
              new boost::function<void ()>(work));
}


//    boost::lock_guard<boost::mutex> lock(m);


} // namespace MMap
} // namespace Datacratic

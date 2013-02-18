/* gc.h                                                            -*- C++ -*-
   Jeremy Barnes, 26 September 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Garbage collection basics.
*/

#ifndef __mmap__gc_h__
#define __mmap__gc_h__

#include "jml/arch/exception.h"
#include <boost/function.hpp>
#include "jml/compiler/compiler.h"

namespace Datacratic {
namespace MMap {

int getThreadNum() JML_PURE_FN;

#if 0


void readLock();
void readUnlock();
bool isReadLocked();

struct ReadGuard {
    ReadGuard()
    {
        readLock();
    }

    ~ReadGuard()
    {
        readUnlock();
    }
};

struct FancyReadGuard {

    FancyReadGuard(bool doLock = true)
        : locked_(false)
    {
        if (doLock) lock();
    }

    ~FancyReadGuard()
    {
        if (locked_) unlock();
    }

    void unlock()
    {
        if (!locked_)
            throw ML::Exception("attempt to unlock unlocked FancyReadGuard");
        readUnlock();
        locked_ = false;
    }

    void lock()
    {
        if (locked_)
            throw ML::Exception("attempt to lock locked FancyReadGuard");
        readLock();
        locked_ = true;
    }

    bool locked() const { return locked_; }

    bool locked_;
};

void writeLock();
void writeUnlock();
bool isWriteLocked();

struct WriteGuard {
    WriteGuard()
    {
        writeLock();
    }
    
    ~WriteGuard()
    {
        writeUnlock();
    }
};

// Stop the whole world, including dealing with all deferred updates
void stopTheWorld();
void restartTheWorld();



struct StopTheWorldGuard {
    StopTheWorldGuard()
    {
        stopTheWorld();
    }
    
    ~StopTheWorldGuard()
    {
        restartTheWorld();
    }
};

/** Wait until everything that's pending for GC has been run.  Runs like a
    barrier.
*/
void waitForGC();

/** Do some deferred work to garbage collect. */
void deferGC(const boost::function<void ()> & work);

#endif

} // namespace MMap
} // namespace Datacratic

#endif /* __mmap__gc_h__ */



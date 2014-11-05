/** spinlock.h                                                     -*- C++ -*-
    Jeremy Barnes, 13 December 2009.  All rights reserved.
    Implementation of a spinlock.
*/

#ifndef __jml__arch__spinlock_h__
#define __jml__arch__spinlock_h__

#include <sched.h>

namespace ML {

struct Spinlock {
    Spinlock(bool yield = true)
        : value(0), yield(yield)
    {
    }

    void lock()
    {
        acquire();
    }

    void unlock()
    {
        release();
    }

    bool locked() const
    {
        return value;
    }

    int try_acquire()
    {
        if (__sync_bool_compare_and_swap(&value, 0, 1))
            return 0;
        return -1;
    }
    
    bool try_lock()
    {
        return try_acquire() == 0;
    }

    int acquire()
    {
        for (int tries = 0; true;  ++tries) {
            if (!__sync_lock_test_and_set(&value, 1))
                return 0;
            if (tries == 100 && yield) {
                tries = 0;
                sched_yield();
            }
        }
    }

    int release()
    {
        __sync_lock_release(&value);
        return 0;
    }

    volatile int value;
    bool yield;
};

} // namespace ML

#endif /* __jml__arch__spinlock_h__ */


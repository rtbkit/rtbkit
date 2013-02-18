/** rwlock.h                                                       -*- C++ -*-
    Jeremy Barnes, 13 November 2011
    Copyright (c) 2011 Datacratic.  All rights reserved.

    Placed under the BSD license.
*/

#ifndef __arch__rwlock_h__
#define __arch__rwlock_h__

#include <pthread.h>
#include "exception.h"
#include <iostream>

namespace ML {

/** Boost's RW mutex uses a condition variable underneath and is way slower
    than it needs to be... hence this data structure.
*/
struct RWLock {
    RWLock()
    {
        pthread_rwlockattr_t lockattr;
        int res = pthread_rwlockattr_init(&lockattr);
        check_err(res);
        pthread_rwlockattr_setkind_np(&lockattr,
                                      PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP);
        check_err(res);

        res = pthread_rwlock_init(&rwlock, &lockattr);
        check_err(res);

        //res = pthread_rwlockattr_destroy(&lockattr); 
        //check_err(res);
    }
    
    ~RWLock()
    {
        int res = pthread_rwlock_destroy(&rwlock);
        if (res == -1)
            std::cerr << "warning: couldn't destroy rwlock" << std::endl;
}
    
    void lock_shared()
    {
        int res = pthread_rwlock_rdlock(&rwlock);
        check_err(res);
    }

    void unlock_shared()
    {
        int res = pthread_rwlock_unlock(&rwlock);
        check_err(res);
    }

    void lock()
    {
        int res = pthread_rwlock_wrlock(&rwlock);
        check_err(res);
    }

    void unlock()
    {
        int res = pthread_rwlock_unlock(&rwlock);
        check_err(res);
    }

    void check_err(int res)
    {
        if (res == -1)
            throw ML::Exception("couldn't perform rwlock operation");
    }

    pthread_rwlock_t rwlock;
};

} // namespace ML

#endif /* __arch__rwlock_h__ */

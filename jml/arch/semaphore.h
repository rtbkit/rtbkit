/* semaphore.h                                                     -*- C++ -*-
   Jeremy Barnes, 7 September 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Implementation of a semaphore; nominally on top of the futex.
*/

#pragma once

#include "jml/arch/exception.h"
#include <iostream>

#if 0

#include <ace/Semaphore.h>

namespace ML {


/** The ACE semaphores are not useful, as two tryacquire operations performed
    in parallel with 2 free semaphores (eg, enough for both to acquire the
    semaphore) can cause only one to succeed.  This is because the tryacquire
    operation performs a tryacquire on the semaphore's mutex, which just
    protects the internal state, and doesn't retry afterwards.

    We create this class here with a useful tryacquire behaviour.

    TODO: this is suboptimal due to the calls to gettimeofday(); we should
    use something that directly implements semaphores using the pthread
    primitives, or attempt to re-write the ACE implementation.
*/
struct Semaphore : public ACE_Semaphore {
    Semaphore(int initial_count = 1)
        : ACE_Semaphore(initial_count)
    {
    }

    int acquire()
    {
        return ACE_Semaphore::acquire();
    }

    int tryacquire()
    {
        return ACE_Semaphore::tryacquire();

        ACE_Time_Value tv = ACE_OS::gettimeofday();
        int result = ACE_Semaphore::acquire(tv);
        if (result == 0) return result;
        if (result == -1 && errno == ETIME) return 0;
        return -1;
    }

    int release()
    {
        return ACE_Semaphore::release();
    }
};

#else // no ACE

#include "jml/arch/futex.h"
#include <semaphore.h>

namespace ML {

struct Semaphore {
    sem_t val;

    Semaphore(int initialVal = 1)
    {
        if (sem_init(&val, 0, initialVal))
            throw ML::Exception(errno, "sem_init");
    }

    ~Semaphore()
    {
        if (sem_destroy(&val))
            throw ML::Exception(errno, "sem_destroy");
    }

    void acquire()
    {
        int res;
        while ((res = sem_wait(&val)) && errno == EINTR) ;
        if (res)
            throw ML::Exception(errno, "sem_wait");
    }

    int tryacquire()
    {
        int res;
        while ((res = sem_trywait(&val)) && errno == EINTR) ;

        if (res && (errno == EAGAIN))
            return -1;
        if (res)
            throw ML::Exception(errno, "sem_trywait");

        return 0;
    }

    void release()
    {
        int res;
        while ((res = sem_post(&val)) && errno == EINTR) ;
        if (res)
            throw ML::Exception(errno, "sem_post");
    }
};


#endif

struct Release_Sem {
    Release_Sem(Semaphore & sem)
        : sem(&sem)
    {
    }

    void operator () ()
    {
        sem->release();
    }

    Semaphore * sem;
};


} // namespace ML

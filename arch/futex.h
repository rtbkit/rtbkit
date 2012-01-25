/* futex.h                                                         -*- C++ -*-
   Jeremy Barnes, 25 January 2012
   Copyright (c) 2012 Recoset.  All rights reserved.

   Basic futex function wrappers.
*/

#ifndef __jml__arch__futex_h__
#define __jml__arch__futex_h__

#include <sys/syscall.h>
#include <linux/futex.h>


namespace ML {

inline long sys_futex(void *addr1, int op, int val1, struct timespec *timeout = 0, void *addr2 = 0, int val3 = 0)
{
    return syscall(SYS_futex, addr1, op, val1, timeout, addr2, val3);
}

inline void futex_wake(int & futex, int nToWake = INT_MAX)
{
    sys_futex(&futex, FUTEX_WAKE, nToWake, 0, 0, 0);
}

inline long futex_wait(int & futex, int oldValue)
{
    return sys_futex(&futex, FUTEX_WAIT, oldValue, 0, 0, 0);
}

inline void futex_unlock(void * futex)
{
    int & lock = *reinterpret_cast<int *>(futex);
    int newLock = __sync_add_and_fetch(&lock, 1);
    if (newLock == 0)
        futex_wake(lock);
}


} // namespace ML


#endif /* __jml__arch__futex_h__ */

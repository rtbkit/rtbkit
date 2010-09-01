/* atomic_init.h                                                   -*- C++ -*-
   Jeremy Barnes, 23 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.
   
   Function to atomically initialize a variable in a thread-safe manner.
*/

#ifndef __arch__atomic_init_h__
#define __arch__atomic_init_h__

#include "arch.h"
#include "cmp_xchg.h"

namespace ML {

/** Atomically initialize the given pointer to the given value.  In the case
    that the value was already initialized or simultaneously initialized by
    another thread, the object pointed to by val will be freed.
*/
template<class X>
bool atomic_init(X * & ptr, X * val)
{
    X * null = 0;
    if (!cmp_xchg(ptr, null, val)) {
        delete val;
        return false;
    }
    return true;
}

} // namespace ML

#endif /* __arch__atomic_init_h__ */

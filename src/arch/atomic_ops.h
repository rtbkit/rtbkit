/* atomic_ops.h                                                    -*- C++ -*-
   Jeremy Barnes,24 October 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Atomic operations.
*/

#ifndef __arch__atomic_ops_h__
#define __arch__atomic_ops_h__

#include "cmp_xchg.h"

namespace ML {


template<typename Val>
void atomic_accumulate(Val & value, const Val & increment)
{
    Val old_val = value, new_val;
    do {
        new_val = old_val + increment;
    } while (!JML_LIKELY(cmp_xchg(value, old_val, new_val)));
}

template<typename Val>
void atomic_accumulate(Val * old, const Val * increment, int n)
{
    for (unsigned i = 0;  i < n;  ++i)
        atomic_accumulate(old[i], increment[i]);
}


} // file scope


#endif /* __arch__atomic_ops_h__ */

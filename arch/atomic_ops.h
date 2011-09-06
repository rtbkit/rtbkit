/* atomic_ops.h                                                    -*- C++ -*-
   Jeremy Barnes,24 October 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Atomic operations.
*/

#ifndef __arch__atomic_ops_h__
#define __arch__atomic_ops_h__

#include "cmp_xchg.h"
#include "jml/compiler/compiler.h"
#include <algorithm>

namespace ML {


template<typename Val1, typename Val2>
void atomic_accumulate(Val1 & value, const Val2 & increment)
{
    Val1 old_val = value, new_val;
    do {
        new_val = old_val + increment;
    } while (!JML_LIKELY(cmp_xchg(value, old_val, new_val)));
}

template<typename Val1, typename Val2>
void atomic_accumulate(Val1 * old, const Val2 * increment, int n)
{
    for (unsigned i = 0;  i < n;  ++i)
        atomic_accumulate(old[i], increment[i]);
}

template<typename Val1, typename Val2>
void atomic_add(Val1 & val, const Val2 & amount)
{
    Val1 amt = amount;
    asm volatile ("lock add %[amount], %[val]\n\t"
         : [val] "=m" (val)
                  : [amount] "r" (amt)
         : "cc");
}

template<typename Val1, typename Val2>
void atomic_set_bits(Val1 & val, Val2 amount)
{
    Val1 bits = amount;
    asm volatile ("lock or %[bits], %[val]\n\t"
         : [val] "=m" (val)
         : [bits] "r" (bits)
         : "cc");
}

template<typename Val1, typename Val2>
void atomic_clear_bits(Val1 & val, Val2 amount)
{
    Val1 bits = ~amount;
    asm volatile ("lock and %[bits], %[val]\n\t"
         : [val] "=m" (val)
         : [bits] "r" (bits)
         : "cc");
}

// Maximum that works atomically.  It's safe against any kind of change
// in old_val.
template<typename Val1, typename Val2>
void atomic_max(Val1 & val1, const Val2 & val2)
{
    Val1 old_val = val1, new_val;
    do {
        new_val = std::max<Val1>(old_val, val2);
        if (new_val == old_val) return;
    } while (!JML_LIKELY(cmp_xchg(val1, old_val, new_val)));
}

JML_ALWAYS_INLINE void memory_barrier()
{
    // GCC < 4.4 doesn't do this properly
    // See http://gcc.gnu.org/bugzilla/show_bug.cgi?id=36793
#if 0
    __sync_synchronize();
#else
    asm volatile ( "mfence; \n" );
#endif
}

} // file scope


#endif /* __arch__atomic_ops_h__ */

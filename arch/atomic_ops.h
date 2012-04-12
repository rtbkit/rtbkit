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
         : [val] "+m" (val)
                  : [amount] "r" (amt)
         : "cc");
}

template<int Width>
struct IncDecSwitch {
};

template<>
struct IncDecSwitch<1> {
    template<typename Val1>
    static void atomic_inc(Val1 & val)
    {
        asm volatile ("lock incb %[val]\n\t"
                      : [val] "+m" (val)
                      :
                      : "cc");
    }

    template<typename Val1>
    static void atomic_dec(Val1 & val)
    {
        asm volatile ("lock decb %[val]\n\t"
                      : [val] "+m" (val)
                      :
                      : "cc");
    }
};

template<>
struct IncDecSwitch<2> {
    template<typename Val1>
    static void atomic_inc(Val1 & val)
    {
        asm volatile ("lock incs %[val]\n\t"
                      : [val] "+m" (val)
                      :
                      : "cc");
    }

    template<typename Val1>
    static void atomic_dec(Val1 & val)
    {
        asm volatile ("lock decs %[val]\n\t"
                      : [val] "+m" (val)
                      :
                      : "cc");
    }
};

template<>
struct IncDecSwitch<4> {
    template<typename Val1>
    static void atomic_inc(Val1 & val)
    {
        asm volatile ("lock incl %[val]\n\t"
                      : [val] "+m" (val)
                      :
                      : "cc");
    }

    template<typename Val1>
    static void atomic_dec(Val1 & val)
    {
        asm volatile ("lock decl %[val]\n\t"
                      : [val] "+m" (val)
                      :
                      : "cc");
    }
};

template<>
struct IncDecSwitch<8> {
    template<typename Val1>
    static void atomic_inc(Val1 & val)
    {
        asm volatile ("lock incq %[val]\n\t"
                      : [val] "+m" (val)
                      :
                      : "cc");
    }

    template<typename Val1>
    static void atomic_dec(Val1 & val)
    {
        asm volatile ("lock decq %[val]\n\t"
                      : [val] "+m" (val)
                      :
                      : "cc");
    }
};

template<typename Val1>
void atomic_inc(Val1 & val)
{
    IncDecSwitch<sizeof(Val1)>::atomic_inc(val);
}

template<typename Val1>
void atomic_dec(Val1 & val)
{
    IncDecSwitch<sizeof(Val1)>::atomic_dec(val);
}

template<typename Val1, typename Val2>
void atomic_set_bits(Val1 & val, Val2 amount)
{
    Val1 bits = amount;
    asm volatile ("lock or %[bits], %[val]\n\t"
         : [val] "+m" (val)
         : [bits] "r" (bits)
         : "cc");
}

template<typename Val1, typename Val2>
void atomic_clear_bits(Val1 & val, Val2 amount)
{
    Val1 bits = ~amount;
    asm volatile ("lock and %[bits], %[val]\n\t"
         : [val] "+m" (val)
         : [bits] "r" (bits)
         : "cc");
}

template<typename Val1>
uint32_t atomic_test_and_set(Val1 & val, uint32_t bitNum)
{
    uint32_t result = 0;
    asm volatile
        ("lock bts %[bitnum], %[val]\n\t"
         "adc      $0, %[result]\n\t"
         : [val] "+m,m" (val), [result] "+r,r" (result)
         : [bitnum] "J,c" ((uint32_t)bitNum)
         : "cc");
    return result;
}

template<typename Val1>
bool atomic_test_and_clear(Val1 & val, uint32_t bitNum)
{
    uint32_t result = 0;
    asm volatile
        ("lock btr %[bitnum], %[val]\n\t"
         "adc      $0, %[result]\n\t"
         : [val] "+m,m" (val), [result] "+r,r" (result)
         : [bitnum] "J,c" ((uint32_t)bitNum)
         : "cc");
    return result;
}

template<typename Val1>
bool atomic_test_and_toggle(Val1 & val, uint32_t bitNum)
{
    uint32_t result = 0;
    asm volatile
        ("lock btc %[bitnum], %[val]\n\t"
         "adc      $0, %[result]\n\t"
         : [val] "+m,m" (val), [result] "+r,r" (result)
         : [bitnum] "J,c" ((uint32_t)bitNum)
         : "cc");
    return result;
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

// Maximum that works atomically.  It's safe against any kind of change
// in old_val.
template<typename Val1, typename Val2>
void atomic_min(Val1 & val1, const Val2 & val2)
{
    Val1 old_val = val1, new_val;
    do {
        new_val = std::min<Val1>(old_val, val2);
        if (new_val == old_val) return;
    } while (!JML_LIKELY(cmp_xchg(val1, old_val, new_val)));
}

template<typename X>
JML_ALWAYS_INLINE X atomic_xchg(X & val, X new_val)
{
    asm volatile ("xchg %[new_val], %[val]\n\t"
                  : [new_val] "+a" (new_val), [val] "+m" (val)
                  :
                  : "cc", "memory");
    return new_val;
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

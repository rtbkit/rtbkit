/* cmp_xchg.h                                                      -*- C++ -*-
   Jeremy Barnes, 21 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

   Compare/exchange instructions.
*/

#ifndef __arch__cmp_xchg_h__
#define __arch__cmp_xchg_h__

#include <stdint.h>
#include "jml/compiler/compiler.h"
#include "jml/arch/arch.h"

namespace ML {

#ifdef JML_INTEL_ISA

template<unsigned Size>
struct cmp_xchg_switch {

    /** Required to prevent compiler "optimizations" in gcc 4.8. My guess is
        that doing atomic operands over float values is undefined because there
        are values that may multiple binary representations.
     */
    JML_ALWAYS_INLINE bool cmp_xchg(double & val, double & old,
                                    const double & new_val)
    {
        auto _val = reinterpret_cast<volatile uint64_t *>(&val);
        auto _old = reinterpret_cast<volatile uint64_t *>(&old);
        auto _new_val = reinterpret_cast<const volatile uint64_t *>(&new_val);

        return cmp_xchg(*_val, *_old, *_new_val);
    }

    /** Required to prevent compiler "optimizations" in gcc 4.8. My guess is
        that doing atomic operands over float values is undefined because there
        are values that may multiple binary representations.
     */
    JML_ALWAYS_INLINE bool cmp_xchg(volatile double & val,
                                    volatile double & old,
                                    volatile const double & new_val)
    {
        auto _val = reinterpret_cast<volatile uint64_t *>(&val);
        auto _old = reinterpret_cast<volatile uint64_t *>(&old);
        auto _new_val = reinterpret_cast<volatile const uint64_t *>(&new_val);

        return cmp_xchg(*_val, *_old, *_new_val);
    }

    template<class X>
    JML_ALWAYS_INLINE bool cmp_xchg(X & val, X & old, const X & new_val)
    {
        uint8_t result;
        asm volatile ("lock cmpxchg %[new_val], (%[val])\n\t"
                      "     setz    %[result]\n\t"
                      : "+&a" (old),
                        [result] "=q" (result)
                      : [val] "r" (&val),
                        [new_val] "r" (new_val)
                      : "cc", "memory");
        return result;
    }

    template<class X>
    JML_ALWAYS_INLINE bool cmp_xchg(volatile X & val, X & old, const X & new_val)
    {
        uint8_t result;
        asm volatile ("lock cmpxchg %[new_val], (%[val])\n\t"
                      "     setz    %[result]\n\t"
                      : "+&a" (old),
                        [result] "=q" (result)
                      : [val] "r" (&val),
                        [new_val] "r" (new_val)
                      : "cc", "memory");
        return result;
    }

};


#if JML_BITS == 32

template<>
struct cmp_xchg_switch<1> {

    template<class X>
    JML_ALWAYS_INLINE bool cmp_xchg(X & val, X & old, const X & new_val)
    {
        uint8_t result;
        asm volatile ("lock cmpxchg %[new_val], (%[val])\n\t"
                      "     setz    %[result]\n\t"
                      : "+&a" (old),
                        [result] "=q" (result)
                      : [val] "r" (&val),
                        [new_val] "q" (new_val)
                      : "cc", "memory");
        return result;
    }

    template<class X>
    JML_ALWAYS_INLINE bool cmp_xchg(volatile X & val, X & old, const X & new_val)
    {
        uint8_t result;
        asm volatile ("lock cmpxchg %[new_val], (%[val])\n\t"
                      "     setz    %[result]\n\t"
                      : "+&a" (old),
                        [result] "=q" (result)
                      : [val] "r" (&val),
                        [new_val] "q" (new_val)
                      : "cc", "memory");
        return result;
    }

};

template<>
struct cmp_xchg_switch<8> {

    template<class X>
    JML_ALWAYS_INLINE bool cmp_xchg(X & val, X & old, const X & new_val)
    {
        /* Split new_val into low and high parts. */
        union {
            struct {
                uint32_t l, h;
            };
            X val;
        } split;
        split.val = new_val;

        uint8_t result;
        /* Note that we have to be careful here as GCC can't handle ebx being
           destroyed very well.  We pretend instead that it isn't destroyed
           and use it to push.  Note that we would still have a problem if
           val was accessed using the ebx register, but since it's the PIC
           register it's not likely. */
        asm volatile ("push %%ebx\n\t"
                      "mov  %[splitl], %%ebx\n\t"
                      "lock cmpxchg8b %[val]\n\t"
                      "     setz      %[result]\n\t"
                      "pop  %%ebx\n\t"
                      : "+A" (old), [result] "=c" (result)
                      : [val] "m" (val), [splitl] "r" (split.l), "c" (split.h)
                      : "cc");
        return result;
    }

    template<class X>
    JML_ALWAYS_INLINE bool cmp_xchg(volatile X & val, X & old, const X & new_val)
    {
        /* Split new_val into low and high parts. */
        union {
            struct {
                uint32_t l, h;
            };
            X val;
        } split;
        split.val = new_val;

        uint8_t result;
        /* Note that we have to be careful here as GCC can't handle ebx being
           destroyed very well.  We pretend instead that it isn't destroyed
           and use it to push.  Note that we would still have a problem if
           val was accessed using the ebx register, but since it's the PIC
           register it's not likely. */
        asm volatile ("push %%ebx\n\t"
                      "mov  %[splitl], %%ebx\n\t"
                      "lock cmpxchg8b %[val]\n\t"
                      "     setz      %[result]\n\t"
                      "pop  %%ebx\n\t"
                      : "+A" (old), [result] "=c" (result)
                      : [val] "m" (val), [splitl] "r" (split.l), "c" (split.h)
                      : "cc");
        return result;
    }
};

#else // 64 bits

template<>
struct cmp_xchg_switch<16> {
    template<class X>
    JML_ALWAYS_INLINE bool cmp_xchg(X & val, X & old, const X & new_val)
    {
        /* Split new_val into low and high parts. */
        uint64_t * pold = reinterpret_cast<uint64_t *>(&old);
        const uint64_t * pnew = reinterpret_cast<const uint64_t *>(&new_val);

        uint8_t result;
        asm volatile ("lock cmpxchg16b  %[val]\n\t"
                      "     setz       %[result]\n\t"
                      : "+a" (pold[0]), "+d" (pold[1]), [result] "=c" (result)
                      : [val] "m" (val), "b" (pnew[0]), "c" (pnew[1])
                      : "cc");
        return result;
    }

    template<class X>
    JML_ALWAYS_INLINE bool cmp_xchg(volatile X & val, X & old, const X & new_val)
    {
        /* Split new_val into low and high parts. */
        uint64_t * pold = reinterpret_cast<uint64_t *>(&old);
        const uint64_t * pnew = reinterpret_cast<const uint64_t *>(&new_val);

        uint8_t result;
        asm volatile ("lock cmpxchg16b  %[val]\n\t"
                      "     setz       %[result]\n\t"
                      : "+a" (pold[0]), "+d" (pold[1]), [result] "=c" (result)
                      : [val] "m" (val), "b" (pnew[0]), "c" (pnew[1])
                      : "cc");
        return result;
    }
};

#endif // 32/64 bits

template<class X>
JML_ALWAYS_INLINE bool cmp_xchg(X & val, X & old, const X & new_val)
{
    return cmp_xchg_switch<sizeof(X)>().cmp_xchg(val, old, new_val);
}

template<class X>
JML_ALWAYS_INLINE bool cmp_xchg(volatile X & val, X & old, const X & new_val)
{
    return cmp_xchg_switch<sizeof(X)>().cmp_xchg(val, old, new_val);
}

#endif // Intel ISA

} // namespace ML

#endif /* __arch__cmp_xchg_h__ */

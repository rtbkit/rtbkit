/* cache.h                                                         -*- C++ -*-
   Jeremy Barnes, 21 January 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Cache control functions.
*/

#ifndef __jml__arch__cache_h__
#define __jml__arch__cache_h__

#include "sse2.h"
#include "jml/compiler/compiler.h"

namespace ML {

static const size_t l1_cache_size = 32 * 1024;


inline void warmup_cache_all_levels(const float * mem, size_t n)
{
    // TODO: assumes 64 byte cache lines
    // TODO: prefetch?
    float total JML_UNUSED = 0.0;
    for (unsigned i = 0;  i < n;  i += 16)
        total += mem[n];
}

inline void warmup_cache_all_levels(const double * mem, size_t n)
{
    // TODO: assumes 64 byte cache lines
    // TODO: prefetch?
    double total JML_UNUSED = 0.0;
    for (unsigned i = 0;  i < n;  i += 8)
        total += mem[n];
}

inline void store_non_temporal(float & addr, float val)
{
    // TODO: use intel compiler intrinsics?
    __asm__ ("movnti %[val], %[mem]\n\t"
             : [mem] "=m" (addr)
             : [val] "r" (val));
}

inline void store_non_temporal(double & addr, double val)
{
    // TODO: use intel compiler intrinsics?
    __asm__ ("movntiq %[val], %[mem]\n\t"
             : [mem] "=m" (addr)
             : [val] "r" (val));
}

inline bool aligned(void * ptr, int bits)
{
    size_t x = reinterpret_cast<size_t>(ptr);
    return ((x & ((1 << bits) - 1)) == 0);
}

inline void streaming_copy_from_strided(float * output, const float * input,
                                        size_t stride, size_t n)
{
    unsigned i = 0;

#if 1
    for (; i < n && !aligned(output + i, 4);  ++i)
        store_non_temporal(*(output + i), input[i * stride]);

    for (; i + 4 <= n;  i += 4) {
        using namespace SIMD;
        const float * addr = input + i * stride;

        // TODO: do something smarter
        //v4sf v0 = __builtin_ia32_loaduss(addr + stride * 0);
        //v4sf v1 = __builtin_ia32_loaduss(addr + stride * 1);
        //v4sf v2 = __builtin_ia32_loaduss(addr + stride * 2);
        //v4sf v3 = __builtin_ia32_loaduss(addr + stride * 3);

        v4sf v = { addr[stride * 0], addr[stride * 1], addr[stride * 2],
                   addr[stride * 3] };

        __builtin_ia32_movntps(output + i, v);
    }
#endif
    

    for (; i < n;  ++i)
        store_non_temporal(*(output + i), input[i * stride]);
}

inline void streaming_copy_from_strided(double * output, const double * input,
                                        size_t stride, size_t n)
{
    unsigned i = 0;

#if 1
    for (; i < n && !aligned(output + i, 4);  ++i)
        store_non_temporal(*(output + i), input[i * stride]);
    
    for (; i + 2 <= n;  i += 2) {
        using namespace SIMD;
        const double * addr = input + i * stride;
        v2df v = { addr[stride * 0], addr[stride * 1] };

        __builtin_ia32_movntpd(output + i, v);
    }
#endif
    

    for (; i < n;  ++i)
        store_non_temporal(*(output + i), input[i * stride]);
}

} // namespace ML

#endif /* __jml__arch__cache_h__ */

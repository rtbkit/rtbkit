/* bitops.h                                                        -*- C++ -*-
   Jeremy Barnes, 23 March 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.

   This file is part of "Jeremy's Machine Learning Library", copyright (c)
   1999-2005 Jeremy Barnes.
   
   This program is available under the GNU General Public License, the terms
   of which are given by the file "license.txt" in the top level directory of
   the source code distribution.  If this file is missing, you have no right
   to use the program; please contact the author.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
   for more details.

   ---

   Bitwise operations.
*/

#ifndef __arch__bitops_h__
#define __arch__bitops_h__

#include "jml/compiler/compiler.h"
#include <stdint.h>
#include <cassert>

namespace ML {

/* NOTE: there are single instructions that can do all this... */

template<unsigned Size>
struct highest_bit_switch;

template<>
struct highest_bit_switch<1> {
    template<class T>
    JML_ALWAYS_INLINE JML_PURE_FN
    int operator () (T arg, int none_set = -1) const
    {
        //return arg ? (sizeof(unsigned) * 8) - __builtin_clz(arg) - 1 : none_set;
#if (defined(__i386__) || defined(__amd64__)) && ! JML_COMPILER_NVCC
        uint32_t in2 = arg & 0xff;
        uint32_t tmp;
        int32_t result = none_set;
        __asm__
            ("bsr    %[in], %[tmp]    \n\t"
             "cmovnz %[tmp], %[out]   \n\t"
             : [out] "+r" (result), [tmp] "=r" (tmp)
             : [in] "r" (in2)
             : "cc"
             );
        return result;
#else
        int res = -1;
        if (arg & 0xf0) { res += 4;  arg >>= 4; }
        if (arg & 0x0c) { res += 2;  arg >>= 2; }
        if (arg & 0x03) { res += 1;  arg >>= 1; }
        assert(arg <= 1);
        res += (arg != 0);
        return res;
#endif
    }
};

template<>
struct highest_bit_switch<2> {
    template<class T>
    JML_ALWAYS_INLINE JML_PURE_FN
    int operator () (T arg, int none_set = -1) const
    {
        //return arg ? (sizeof(unsigned) * 8) - __builtin_clz(arg) - 1 : none_set;
#if (defined(__i386__) || defined(__amd64__)) && ! JML_COMPILER_NVCC
        uint32_t in2 = arg & 0xffff;
        uint32_t tmp;
        int32_t result = none_set;
        __asm__
            ("bsr    %[in], %[tmp]    \n\t"
             "cmovnz %[tmp], %[out]   \n\t"
             : [out] "+r" (result), [tmp] "=r" (tmp)
             : [in] "r" (in2)
             : "cc"
             );
        return result;
#else
        int res = -1;
        if (arg & 0xff00) { res += 8;  arg >>= 8; }
        if (arg & 0x00f0) { res += 4;  arg >>= 4; }
        if (arg & 0x000c) { res += 2;  arg >>= 2; }
        if (arg & 0x0003) { res += 1;  arg >>= 1; }
        assert(arg <= 1);
        res += (arg != 0);
        return res;
#endif
    }
};

template<>
struct highest_bit_switch<4> {
    template<class T>
    JML_ALWAYS_INLINE JML_PURE_FN
    int operator () (T arg, int none_set = -1) const
    {
        //return arg ? (sizeof(unsigned) * 8) - __builtin_clz(arg) - 1 : none_set;
#if (defined(__i386__) || defined(__amd64__)) && ! JML_COMPILER_NVCC
        uint32_t in2 = arg;
        uint32_t tmp;
        int32_t result = none_set;
        __asm__
            ("bsr    %[in], %[tmp]    \n\t"
             "cmovnz %[tmp], %[out]   \n\t"
             : [out] "+r" (result), [tmp] "=r" (tmp)
             : [in] "r" (in2)
             : "cc"
             );
        return result;
#else
        int res = -1;
        if (arg & 0xffff0000) { res += 16;  arg >>= 16; }
        if (arg & 0x0000ff00) { res += 8;   arg >>=  8; }
        if (arg & 0x000000f0) { res += 4;   arg >>=  4; }
        if (arg & 0x0000000c) { res += 2;   arg >>=  2; }
        if (arg & 0x00000003) { res += 1;   arg >>=  1; }
        assert(arg <= 1);
        res += (arg != 0);
        return res;
#endif
    }
};

template<>
struct highest_bit_switch<8> {
    template<class T>
    JML_ALWAYS_INLINE JML_PURE_FN
    int operator () (T arg, int none_set = -1) const
    {
#if defined(__amd64__) && ! JML_COMPILER_NVCC
        uint64_t in2 = arg;
        uint64_t tmp;
        int64_t result = none_set;
        __asm__
            ("bsr    %[in], %[tmp]    \n\t"
             "cmovnz %[tmp], %[out]   \n\t"
             : [out] "+r" (result), [tmp] "=r" (tmp)
             : [in] "r" (in2)
             : "cc"
             );
        return result;
#else
# if 1
        int result = highest_bit_switch<4>()(arg >> 32);
        if (result != -1) return result + 32;
        return result = highest_bit_switch<4>()(arg, none_set);
# else
        int res = -1;
        if (arg & 0xffffffff00000000ULL) { res += 32;  arg >>= 32; }
        if (arg & 0x00000000ffff0000ULL) { res += 16;  arg >>= 16; }
        if (arg & 0x000000000000ff00ULL) { res +=  8;  arg >>= 8; }
        if (arg & 0x00000000000000f0ULL) { res +=  4;  arg >>= 4; }
        if (arg & 0x000000000000000CULL) { res +=  2;  arg >>= 2; }
        if (arg & 0x0000000000000003ULL) { res +=  1;  arg >>= 1; }
        assert(arg <= 1);
        res += (arg != 0);
        
        return res;
# endif
#endif
    }
};

template<class T>
JML_PURE_FN int highest_bit(T arg, int none_set = -1)
{
    // This funky code allows for GCC to calculate the result directly where
    // it knows that the arguments are all constants
    if (__builtin_constant_p(arg) && __builtin_constant_p(none_set)
        && arg >= 0) {
        return arg
            ? sizeof(unsigned long long) * 8 - __builtin_clzll(arg) - 1
            : none_set;
    }
    return highest_bit_switch<sizeof(T)>()(arg, none_set);
}


/*****************************************************************************/
/* LOWEST BIT                                                                */
/*****************************************************************************/

template<unsigned Size>
struct lowest_bit_switch;

template<>
struct lowest_bit_switch<1> {
    template<class T>
    JML_ALWAYS_INLINE JML_PURE_FN
    int operator () (T arg, int none_set = -1) const
    {
        //return arg ? (sizeof(unsigned) * 8) - __builtin_clz(arg) - 1 : none_set;
#if (defined(__i386__) || defined(__amd64__)) && ! JML_COMPILER_NVCC
        uint32_t in2 = arg & 0xff;
        uint32_t tmp;
        int32_t result = none_set;
        __asm__
            ("bsf    %[in], %[tmp]    \n\t"
             "cmovnz %[tmp], %[out]   \n\t"
             : [out] "+r" (result), [tmp] "=r" (tmp)
             : [in] "r" (in2)
             : "cc"
             );
        return result;
#else
        int res = -1;
        if (arg & 0xf0) { res += 4;  arg >>= 4; }
        if (arg & 0x0c) { res += 2;  arg >>= 2; }
        if (arg & 0x03) { res += 1;  arg >>= 1; }
        assert(arg <= 1);
        res += (arg != 0);
        return res;
#endif
    }
};

template<>
struct lowest_bit_switch<2> {
    template<class T>
    JML_ALWAYS_INLINE JML_PURE_FN
    int operator () (T arg, int none_set = -1) const
    {
        //return arg ? (sizeof(unsigned) * 8) - __builtin_clz(arg) - 1 : none_set;
#if (defined(__i386__) || defined(__amd64__)) && ! JML_COMPILER_NVCC
        uint32_t in2 = arg & 0xffff;
        uint32_t tmp;
        int32_t result = none_set;
        __asm__
            ("bsf    %[in], %[tmp]    \n\t"
             "cmovnz %[tmp], %[out]   \n\t"
             : [out] "+r" (result), [tmp] "=r" (tmp)
             : [in] "r" (in2)
             : "cc"
             );
        return result;
#else
        int res = -1;
        if (arg & 0xff00) { res += 8;  arg >>= 8; }
        if (arg & 0x00f0) { res += 4;  arg >>= 4; }
        if (arg & 0x000c) { res += 2;  arg >>= 2; }
        if (arg & 0x0003) { res += 1;  arg >>= 1; }
        assert(arg <= 1);
        res += (arg != 0);
        return res;
#endif
    }
};

template<>
struct lowest_bit_switch<4> {
    template<class T>
    JML_ALWAYS_INLINE JML_PURE_FN
    int operator () (T arg, int none_set = -1) const
    {
        //return arg ? (sizeof(unsigned) * 8) - __builtin_clz(arg) - 1 : none_set;
#if (defined(__i386__) || defined(__amd64__)) && ! JML_COMPILER_NVCC
        uint32_t in2 = arg;
        uint32_t tmp;
        int32_t result = none_set;
        __asm__
            ("bsf    %[in], %[tmp]    \n\t"
             "cmovnz %[tmp], %[out]   \n\t"
             : [out] "+r" (result), [tmp] "=r" (tmp)
             : [in] "r" (in2)
             : "cc"
             );
        return result;
#else
        int res = -1;
        if (arg & 0xffff0000) { res += 16;  arg >>= 16; }
        if (arg & 0x0000ff00) { res += 8;   arg >>=  8; }
        if (arg & 0x000000f0) { res += 4;   arg >>=  4; }
        if (arg & 0x0000000c) { res += 2;   arg >>=  2; }
        if (arg & 0x00000003) { res += 1;   arg >>=  1; }
        assert(arg <= 1);
        res += (arg != 0);
        return res;
#endif
    }
};

template<>
struct lowest_bit_switch<8> {
    template<class T>
    JML_ALWAYS_INLINE JML_PURE_FN
    int operator () (T arg, int none_set = -1) const
    {
#if defined(__amd64__) && ! JML_COMPILER_NVCC
        uint64_t in2 = arg;
        uint64_t tmp;
        int64_t result = none_set;
        __asm__
            ("bsf    %[in], %[tmp]    \n\t"
             "cmovnz %[tmp], %[out]   \n\t"
             : [out] "+r" (result), [tmp] "=r" (tmp)
             : [in] "r" (in2)
             : "cc"
             );
        return result;
#else
# if 1
        int result = lowest_bit_switch<4>()(arg >> 32);
        if (result != -1) return result + 32;
        return result = lowest_bit_switch<4>()(arg, none_set);
# else
        int res = -1;
        if (arg & 0xffffffff00000000ULL) { res += 32;  arg >>= 32; }
        if (arg & 0x00000000ffff0000ULL) { res += 16;  arg >>= 16; }
        if (arg & 0x000000000000ff00ULL) { res +=  8;  arg >>= 8; }
        if (arg & 0x00000000000000f0ULL) { res +=  4;  arg >>= 4; }
        if (arg & 0x000000000000000CULL) { res +=  2;  arg >>= 2; }
        if (arg & 0x0000000000000003ULL) { res +=  1;  arg >>= 1; }
        assert(arg <= 1);
        res += (arg != 0);
        
        return res;
# endif
#endif
    }
};

template<class T>
JML_PURE_FN int lowest_bit(T arg, int none_set = -1)
{
    // This funky code allows for GCC to calculate the result directly where
    // it knows that the arguments are all constants
    if (__builtin_constant_p(arg) && __builtin_constant_p(none_set)
        && arg >= 0) {
        return arg
            ? __builtin_ctzll(arg)
            : none_set;
    }

    return lowest_bit_switch<sizeof(T)>()(arg, none_set);
}

template<class T>
JML_PURE_FN int num_bits_set(T arg)
{
    return __builtin_popcount(arg);
}

JML_PURE_FN JML_ALWAYS_INLINE int num_bits_set(unsigned long arg)
{
    return __builtin_popcountl(arg);
}

JML_PURE_FN JML_ALWAYS_INLINE int num_bits_set(signed long arg)
{
    return __builtin_popcountl(arg);
}

JML_PURE_FN JML_ALWAYS_INLINE int num_bits_set(unsigned long long arg)
{
    return __builtin_popcountll(arg);
}

JML_PURE_FN JML_ALWAYS_INLINE int num_bits_set(signed long long arg)
{
    return __builtin_popcountll(arg);
}

template<int Width>
struct RotateSwitch {
};

template<>
struct RotateSwitch<1> {
    template<typename Val>
    JML_ALWAYS_INLINE JML_PURE_FN JML_COMPUTE_METHOD
    static Val rotate_right(Val val, uint32_t bits)
    {
        __asm__("rorb   %[bits], %[val] \n\t"
                : [val] "+r,r" (val)
                : [bits] "J,c" ((uint8_t)bits)
                : "cc"
                );
        return val;
    }

    template<typename Val>
    JML_ALWAYS_INLINE JML_PURE_FN JML_COMPUTE_METHOD
    static Val rotate_left(Val val, uint32_t bits)
    {
        __asm__("rolb   %[bits], %[val] \n\t"
                : [val] "+r,r" (val)
                : [bits] "J,c" ((uint8_t)bits)
                : "cc"
                );
        return val;
    }
};

template<>
struct RotateSwitch<2> {
    template<typename Val>
    JML_ALWAYS_INLINE JML_PURE_FN JML_COMPUTE_METHOD
    static Val rotate_right(Val val, uint32_t bits)
    {
        __asm__("rorw   %[bits], %[val] \n\t"
                : [val] "+r,r" (val)
                : [bits] "J,c" ((uint8_t)bits)
                : "cc"
                );
        return val;
    }

    template<typename Val>
    JML_ALWAYS_INLINE JML_PURE_FN JML_COMPUTE_METHOD
    static Val rotate_left(Val val, uint32_t bits)
    {
        __asm__("rolw   %[bits], %[val] \n\t"
                : [val] "+r,r" (val)
                : [bits] "J,c" ((uint8_t)bits)
                : "cc"
                );
        return val;
    }
};

template<>
struct RotateSwitch<4> {
    template<typename Val>
    JML_ALWAYS_INLINE JML_PURE_FN JML_COMPUTE_METHOD
    static Val rotate_right(Val val, uint32_t bits)
    {
        __asm__("rorl   %[bits], %[val] \n\t"
                : [val] "+r,r" (val)
                : [bits] "J,c" ((uint8_t)bits)
                : "cc"
                );
        return val;
    }

    template<typename Val>
    JML_ALWAYS_INLINE JML_PURE_FN JML_COMPUTE_METHOD
    static Val rotate_left(Val val, uint32_t bits)
    {
        __asm__("roll   %[bits], %[val] \n\t"
                : [val] "+r,r" (val)
                : [bits] "J,c" ((uint8_t)bits)
                : "cc"
                );
        return val;
    }
};

template<>
struct RotateSwitch<8> {
    template<typename Val>
    JML_ALWAYS_INLINE JML_PURE_FN JML_COMPUTE_METHOD
    static Val rotate_right(Val val, uint32_t bits)
    {
        __asm__("rorq   %[bits], %[val] \n\t"
                : [val] "+r,r" (val)
                : [bits] "J,c" ((uint8_t)bits)
                : "cc"
                );
        return val;
    }

    template<typename Val>
    JML_ALWAYS_INLINE JML_PURE_FN JML_COMPUTE_METHOD
    static Val rotate_left(Val val, uint32_t bits)
    {
        __asm__("rolq   %[bits], %[val] \n\t"
                : [val] "+r,r" (val)
                : [bits] "J,c" ((uint8_t)bits)
                : "cc"
                );
        return val;
    }
};

template<typename Val>
JML_ALWAYS_INLINE JML_PURE_FN JML_COMPUTE_METHOD
Val rotate_right(Val val, uint32_t bits)
{
    // This funky code allows for GCC to calculate the result directly where
    // it knows that the arguments are all constants
    if (__builtin_constant_p(val) && __builtin_constant_p(bits)
        && val >= 0 && bits <= sizeof(Val) * 8) {
        return (val >> bits) | (val << (sizeof(Val) * 8 - bits));
    }
    return RotateSwitch<sizeof(Val)>::rotate_right(val, bits);
}

template<typename Val>
JML_ALWAYS_INLINE JML_PURE_FN JML_COMPUTE_METHOD
Val rotate_left(Val val, uint32_t bits)
{
    // This funky code allows for GCC to calculate the result directly where
    // it knows that the arguments are all constants
    if (__builtin_constant_p(val) && __builtin_constant_p(bits)
        && val >= 0 && bits <= sizeof(Val) * 8) {
        return (val << bits) | (val >> (sizeof(Val) * 8 - bits));
    }
    return RotateSwitch<sizeof(Val)>::rotate_left(val, bits);
}

} // namespace ML

#endif /* __arch__bitops_h__ */

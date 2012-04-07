/* fixed_point_accum.h                                             -*- C++ -*-
   Jeremy Barnes, 1 April 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Fixed-point accurate accumulators.
*/

#ifndef __boosting__fixed_point_accum_h__
#define __boosting__fixed_point_accum_h__

#include "jml/compiler/compiler.h"
#include "jml/utils/float_traits.h"

namespace ML {

/** A structure to accumulate values between zero and one in a single 32
    bit integer. */

struct FixedPointAccum32Unsigned {
    unsigned rep;

    static constexpr float VAL_2_REP = 1ULL << 32;
    static constexpr float REP_2_VAL = 1.0f / (1ULL << 32);
    static constexpr float ADD_TO_ROUND = 1.0f / (1ULL << 33);
    static constexpr unsigned MAX_REP = (unsigned)-1;

    FixedPointAccum32Unsigned()
        : rep(0)
    {
    }

    FixedPointAccum32Unsigned(float value)
        : rep((value + ADD_TO_ROUND)* VAL_2_REP)
    {
    }

    operator float() const { return rep * REP_2_VAL; }

    FixedPointAccum32Unsigned &
    operator += (const FixedPointAccum32Unsigned & other)
    {
        unsigned new_rep = rep + other.rep;
        rep = (new_rep < rep ? MAX_REP : new_rep);
        return *this;
    }

    FixedPointAccum32Unsigned
    operator + (const FixedPointAccum32Unsigned & other) const
    {
        FixedPointAccum32Unsigned result = *this;
        result += other;
        return result;
    }
};

struct FixedPointAccum32 {
    int rep;

    static constexpr float VAL_2_REP = 1ULL << 31;
    static constexpr float REP_2_VAL = 1.0f / (1ULL << 31);
    static constexpr float ADD_TO_ROUND = 0.5f / (1ULL << 31);

    FixedPointAccum32()
        : rep(0)
    {
    }

    FixedPointAccum32(float value)
        : rep((value + ADD_TO_ROUND)* VAL_2_REP)
    {
    }

    operator float() const { return rep * REP_2_VAL; }

    FixedPointAccum32 & operator += (const FixedPointAccum32 & other)
    {
        rep += other.rep;
        return *this;
    }

    FixedPointAccum32 operator + (const FixedPointAccum32 & other) const
    {
        FixedPointAccum32 result = *this;
        result += other;
        return result;
    }
};

#ifdef JML_COMPILER_NVCC
__device__
void
atomic_add(FixedPointAccum32 & result, FixedPointAccum32 other)
{
    unsigned old = atomicAdd(&result.rep, other.rep);
    //if (result < old)
    //    result.rep = FixedPointAccum::MAX_REP;
}
__device__
void
atomic_add_shared(FixedPointAccum32 & result, FixedPointAccum32 other)
{
    atomic_add(result, other);
}
#endif // JML_COMPILER_NVCC

struct FixedPointAccum64 {
#ifndef JML_COMPILER_NVCC
    union {
        struct {
            unsigned l;
            int h;
        };
        long long hl;
    };
#else // JML_COMPILER_NVCC
    // Note that using a union like this for NVCC forces the structure into
    // local memory, which adversely affects the speed.

    long long hl;
    __host__ __device__ unsigned h() const { return hl >> 32; }
    __host__ __device__ unsigned l() const { return hl; }
#endif // JML_COMPILER_NVCC
    static constexpr float VAL_2_HL = 1.0f * (1ULL << 63);
    static constexpr float HL_2_VAL = 1.0f / (1ULL << 63);
    static constexpr float VAL_2_H = (1ULL << 31);
    static constexpr float H_2_VAL = 1.0f / (1ULL << 31);
    static constexpr float ADD_TO_ROUND = 0.5f / (1ULL << 63);
    
    FixedPointAccum64()
        : hl(0)
    {
    }

    FixedPointAccum64(float value)
        : hl((value + ADD_TO_ROUND)* VAL_2_HL)
    {
    }

#ifndef JML_COMPILER_NVCC
    operator float() const { return h * H_2_VAL; }
#else
    operator float() const { return h() * H_2_VAL; }
#endif

    FixedPointAccum64 & operator += (const FixedPointAccum64 & other)
    {
        hl += other.hl;
        return *this;
    }

    FixedPointAccum64 & operator -= (const FixedPointAccum64 & other)
    {
        hl -= other.hl;
        return *this;
    }

    FixedPointAccum64 operator + (const FixedPointAccum64 & other) const
    {
        FixedPointAccum64 result = *this;
        result += other;
        return result;
    }

    FixedPointAccum64 operator + (float other) const
    {
        FixedPointAccum64 result = *this;
        result += FixedPointAccum64(other);
        return result;
    }

    FixedPointAccum64 operator + (double other) const
    {
        FixedPointAccum64 result = *this;
        result += FixedPointAccum64(other);
        return result;
    }
};

#ifdef JML_COMPILER_NVCC
__device__
void
atomic_add(FixedPointAccum64 & result, const FixedPointAccum64 & other)
{
    atomicAdd((unsigned long long *)&result.hl, other.hl);
}

__device__
void
atomic_add_shared(FixedPointAccum64 & result, const FixedPointAccum64 & other)
{
    // We have to to it in two 32 bit operations
    unsigned old = atomicAdd((unsigned *)(&result), other.l());

    // Do we have carry?
    unsigned carry = (((unsigned)-1) - old) < other.l();

    // What to we add to the high value?
    unsigned toadd = carry + other.h();

    // Do the high bit
    if (toadd != 0) atomicAdd(((unsigned *)(&result)) + 1, toadd);
}
#endif // JML_COMPILER_NVCC

template<typename F>
float operator / (const FixedPointAccum64 & f1, const F & f2)
{
    return f1.operator float() / float(f2);
}

template<typename F>
float operator / (const F & f1, const FixedPointAccum64 & f2)
{
    return float(f1) / f2.operator float();
}

inline float operator / (const FixedPointAccum64 & f1,
                          const FixedPointAccum64 & f2)
{
    return f1.operator float() / f2.operator float();
}


} // namespace ML

#endif /* __boosting__fixed_point_accum_h__ */

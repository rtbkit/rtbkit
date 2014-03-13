/* dual_float_accum.h                                              -*- C++ -*-
   Jeremy Barnes, 1 April 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Accumulator for two floats.
*/

#ifndef __boosting__dual_float_accum_h__
#define __boosting__dual_float_accum_h__

#include "jml/compiler/compiler.h"

namespace ML {

/** In this structure, we store a double as two floats, with one estimating
    the error of the other.  Allows a more accuate way of accumulating lots
    of small values together.
*/

struct DualFloatAccum {
    float val, err;
    
#ifdef JML_COMPILER_NVCC
    DualFloatAccum(float2 other)
        : val(other.x), err(other.y)
    {
    }
#endif // JML_COMPILER_NVCC

    DualFloatAccum(float other = 0.0f)
    {
        val = other;
        err = 0.0f;
    }

    DualFloatAccum(double other)
    {
        val = (float)other;
        err = other - (double)val;  // estimate error
    }

    operator float() const { return val + err; }

    //operator double() const { return (double)x + (double)y; }
    
    DualFloatAccum & operator += (const DualFloatAccum & other)
    {
        volatile float toadd = other.val + err;
        volatile float newx  = val + toadd;
        volatile float added = newx - val;
        err       = toadd - added + other.err;
        val       = newx;
        return *this;
    }

    DualFloatAccum & operator += (float other)
    {
        volatile float toadd = other + err;
        volatile float newx  = val + toadd;
        volatile float added = newx - val;
        err       = toadd - added;
        val       = newx;
        return *this;
    }

    DualFloatAccum operator + (const DualFloatAccum & other) const
    {
        DualFloatAccum result = *this;
        result += other;
        return result;
    }

    DualFloatAccum & operator - ()
    {
        val = -val;
        err = -err;
        return *this;
    }

    DualFloatAccum & operator -= (const DualFloatAccum & other)
    {
        operator += (-other);
        return *this;
    }

    DualFloatAccum operator - (const DualFloatAccum & other) const
    {
        DualFloatAccum result = *this;
        result -= other;
        return result;
    }

#if 0
    DualFloatAccum & operator *= (const DualFloatAccum & other)
    {
        y   = (x * other.y)
            + (y * other.x)
            + (y * other.y);  // should always be insignificant...
        
        x   = (x * other.x);

        return *this;
    }
#endif
} __align__(8);


#ifdef JML_COMPILER_NVCC

__device__
void
atomic_add(FloatAccum & result, FloatAccum other)
{
#if 1
    // First, get the old y and replace it with zero.  That way we're sure
    // that we don't count it twice.  0 unsigned is also 0.0 floating point.
    volatile float oldy = atomic_exchange(result.rep.y, 0.0f);
    
    // Now, calculate the amount we need to add including the y
    volatile float toadd = other.rep.x + oldy;

    // Perform the addition, returning the old and new values
    float2 oldnew = atomic_add(result.rep.x, toadd);
    volatile float oldx = oldnew.x;
    volatile float newx = oldnew.y;
    
    // Calculate the new y (error) term
    volatile float added = newx - oldx;
    volatile float newy  = toadd - added + other.rep.y;

    // Add in the new y term to any existing term that might have ended up
    // there
    atomic_add(result.rep.y, newy);
#elif 1
    atomic_add(result.rep.x, other.rep.x);
#else
    result += to_add;
#endif
}

__device__
void
atomic_add_shared(FloatAccum & result, FloatAccum other)
{
    return atomic_add(result, other);
}

#endif // JML_COMPILER_NVCC

} // namespace ML

#endif /* __boosting__dual_float_accum_h__ */

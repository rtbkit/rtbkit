/* atomic.h                                                        -*- C++ -*-
   Jeremy Barnes, 1 April 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Atomic operations for CUDA.
*/

#ifndef __cuda__atomic_h__
#define __cuda__atomic_h__

#include "jml/compiler/compiler.h"

#if (! defined(JML_COMPILER_NVCC) ) || (! JML_COMPILER_NVCC)
# error "This file should only be included for CUDA"
#endif

// No namespaces since a CUDA file...

__device__
void
atomic_add(double & result, double to_add)
{
    // We need to do an atomic update
    unsigned long long old_val;
    unsigned long long * value
        = (unsigned long long *)&result;
    bool done = false;
    do {
        old_val = *value;
        double new_val = __longlong_as_double(old_val) + to_add;
        unsigned long long new_val2 = __double_as_longlong(new_val);
        unsigned long long old_val_seen = atomicCAS(value, old_val, new_val2);
        done = (old_val_seen == old_val);
    } while (!done);
}

__device__
float2
atomic_add(float & result, float to_add)
{
    // We need to do an atomic update
    int old_val;
    int * value = (int *)&result;
    bool done = false;
    float new_val;
    do {
        old_val = *value;
        new_val = __int_as_float(old_val) + to_add;
        int new_val2 = __float_as_int(new_val);
        int old_val_seen = atomicCAS(value, old_val, new_val2);
        done = (old_val_seen == old_val);
    } while (!done);

    return make_float2(__int_as_float(old_val), new_val);
}

__device__
float2
atomic_add_shared(float & result, float to_add)
{
    return atomic_add(result, to_add);
}

__device__
float atomic_exchange(float & val, float new_val)
{
    int * value = (int *)&val;
    int new_val2 = __float_as_int(new_val);
    int old_val_int = atomicExch(value, new_val2);
    return __int_as_float(old_val_int);
}

#endif /* __cuda__atomic_h__ */

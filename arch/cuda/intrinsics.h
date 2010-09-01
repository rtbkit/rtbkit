/* cuda_intrinsics.h                                               -*- C++ -*-
   Jeremy Barnes, 20 March 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Intrinsic functions for CUDA.
*/

#ifndef __jml__compiler__cuda_intrinsics_h__
#define __jml__compiler__cuda_intrinsics_h__

#include "jml/compiler/compiler.h"

#if (! defined(JML_COMPILER_NVCC) ) || (! JML_COMPILER_NVCC)
# error "This file should only be included for CUDA"
#endif

// No namespaces since a CUDA file...

#include <stdint.h>

namespace ML {

JML_ALWAYS_INLINE JML_COMPUTE_METHOD
bool isnanf(float f)
{
    return isnan(f);
}

JML_ALWAYS_INLINE JML_COMPUTE_METHOD
float min(float f1, float f2)
{
    return fmin(f1, f2);
}

JML_ALWAYS_INLINE JML_COMPUTE_METHOD
float max(float f1, float f2)
{
    return fmax(f1, f2);
}

JML_ALWAYS_INLINE JML_COMPUTE_METHOD
double min(double f1, double f2)
{
    return fmin(f1, f2);
}

JML_ALWAYS_INLINE JML_COMPUTE_METHOD
double max(double f1, double f2)
{
    return fmax(f1, f2);
}
JML_ALWAYS_INLINE JML_COMPUTE_METHOD
float min(uint32_t f1, uint32_t f2)
{
    return ::min(f1, f2);
}

JML_ALWAYS_INLINE JML_COMPUTE_METHOD
float max(uint32_t f1, uint32_t f2)
{
    return ::max(f1, f2);
}

JML_ALWAYS_INLINE JML_COMPUTE_METHOD
float min(int32_t f1, int32_t f2)
{
    return ::min(f1, f2);
}

JML_ALWAYS_INLINE JML_COMPUTE_METHOD
float max(int32_t f1, int32_t f2)
{
    return ::max(f1, f2);
}

template<typename T>
JML_ALWAYS_INLINE JML_COMPUTE_METHOD
static void swap(T & val1, T & val2)
{
    T tmp = val1;
    val1 = val2;
    val2 = tmp;
}

} // namespace ML

#endif /* __jml__compiler__cuda_intrinsics_h__ */

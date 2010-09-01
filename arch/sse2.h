/* sse2.h                                                           -*- C++ -*-
   Jeremy Barnes, 15 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

   Wrappers around SSE functions.
*/

#ifndef __arch__sse2_h__
#define __arch__sse2_h__

#include "jml/compiler/compiler.h"
#include <iostream>

#define USE_SIMD_SSE2 1

namespace ML {
namespace SIMD {

typedef float v4sf __attribute__((__vector_size__(16)));

JML_ALWAYS_INLINE v4sf vec_splat(float val)
{
    v4sf result = {val, val, val, val};

    return result;
}

inline std::ostream & operator << (std::ostream & stream, const v4sf & val)
{
    float vals[4];
    *((v4sf *)vals) = val;
    return stream << "{ " << vals[0] << ", " << vals[1] << ", " << vals[2]
                  << ", " << vals[3] << " }";
}

typedef double v2df __attribute__((__vector_size__(16)));

JML_ALWAYS_INLINE v2df vec_splat(double val)
{
    v2df result = {val, val};
    return result;
}

JML_ALWAYS_INLINE v4sf vec_d2f(v2df low, v2df high)
{
    v4sf rr0a  = __builtin_ia32_cvtpd2ps(low);
    v4sf rr0b  = __builtin_ia32_cvtpd2ps(high);
    return __builtin_ia32_shufps(rr0a, rr0b, 0x44);
}

JML_ALWAYS_INLINE void vec_f2d(v4sf ffff, v2df & low, v2df & high)
{
    low  = __builtin_ia32_cvtps2pd(ffff);
    ffff = __builtin_ia32_shufps(ffff, ffff, 14);
    high = __builtin_ia32_cvtps2pd(ffff);
}

inline std::ostream & operator << (std::ostream & stream, const v2df & val)
{
    double vals[2];
    *((v2df *)vals) = val;
    return stream << "{ " << vals[0] << ", " << vals[1] << " }";
}

typedef int v4si __attribute__((__vector_size__(16)));

JML_ALWAYS_INLINE v4si vec_splat(int val)
{
    v4si result = {val, val, val, val};
    return result;
}

inline std::ostream & operator << (std::ostream & stream, const v4si & val)
{
    int vals[4];
    *((v4si *)vals) = val;
    return stream << "{ " << vals[0] << ", " << vals[1] << ", " << vals[2]
                  << ", " << vals[3] << " }";
}

typedef long long int v2di __attribute__((__vector_size__(16)));

inline std::ostream & operator << (std::ostream & stream, const v2di & val)
{
    long long int vals[2];
    *((v2di *)vals) = val;
    return stream << "{ " << vals[0] << ", " << vals[1] << " }";
}


inline void unpack(v2df val, double * where)
{
    (*(v2df *)where) = val;
}

inline v2df pack(double * where)
{
    return *(v2df *)where;
}

inline void unpack(v4sf val, float * where)
{
    (*(v4sf *)where) = val;
}

inline v4sf pack(float * where)
{
    return *(v4sf *)where;
}

inline void unpack(v4si val, int * where)
{
    (*(v4si *)where) = val;
}

inline v4si pack(int * where)
{
    return *(v4si *)where;
}


} // namespace SIMD
} // namespace ML


#endif /* __arch__sse2_h__ */

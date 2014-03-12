/* sse2_poly.h                                                     -*- C++ -*-
   Jeremy Barnes, 23 January 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

   SSE2 polynomial evalution functions.
*/

#ifndef __jml__arch__sse2_poly_h__
#define __jml__arch__sse2_poly_h__

namespace ML {
namespace SIMD {

inline v4sf polevl( v4sf x, float * coef, int N )
{
    float * p = coef;
    v4sf ans = vec_splat(*p++);
    int i = N;

    do ans = ans * x + vec_splat(*p++);
    while (--i);

    return ans;
}

inline v4sf polevl_1( v4sf x, float * coef, int N )
{
    float * p = coef;
    v4sf ans = vec_splat(*p++) * x;
    int i = N;

    do ans = (ans + vec_splat(*p++)) * x;
    while (--i);
    
    return ans;
}

inline float polevl( float x, float * coef, int N )
{
    float * p = coef;
    float ans = *p++;
    int i = N;

    do ans = ans * x + *p++;
    while (--i);

    return ans;
}


inline v2df polevl( v2df x, double * coef, int N )
{
    double * p = coef;
    v2df ans = vec_splat(*p++);
    int i = N;

    do ans = ans * x + vec_splat(*p++);
    while (--i);

    return ans;
}

inline double polevl( double x, double * coef, int N )
{
    double * p = coef;
    double ans = *p++;
    int i = N;

    do ans = ans * x + *p++;
    while (--i);

    return ans;
}


} // namespace SIMD
} // namespace ML

#endif /* __jml__arch__sse2_poly_h__ */


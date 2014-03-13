/* sse2_log.h                                                      -*- C++ -*-
   Jeremy Barnes, 23 January 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

   SSE2 logarithm functions.
*/

#ifndef __jml__arch__sse2_log_h__
#define __jml__arch__sse2_log_h__

#include "sse2.h"
#include "sse2_poly.h"
#include "sse2_math.h"
#include "sse2_misc.h"

namespace ML {
namespace SIMD {

/*****************************************************************************/
/* SINGLE PRECISION LOG                                                      */
/*****************************************************************************/

/*							logf.c
 *
 *	Natural logarithm
 *
 *
 *
 * SYNOPSIS:
 *
 * float x, y, logf();
 *
 * y = logf( x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns the base e (2.718...) logarithm of x.
 *
 * The argument is separated into its exponent and fractional
 * parts.  If the exponent is between -1 and +1, the logarithm
 * of the fraction is approximated by
 *
 *     log(1+x) = x - 0.5 x**2 + x**3 P(x)
 *
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    IEEE      0.5, 2.0    100000       7.6e-8     2.7e-8
 *    IEEE      1, MAXNUMF  100000                  2.6e-8
 *
 * In the tests over the interval [1, MAXNUM], the logarithms
 * of the random arguments were uniformly distributed over
 * [0, MAXLOGF].
 *
 * ERROR MESSAGES:
 *
 * logf singularity:  x = 0; returns MINLOG
 * logf domain:       x < 0; returns MINLOG
 */

/*
Cephes Math Library Release 2.2:  June, 1992
Copyright 1984, 1987, 1988, 1992 by Stephen L. Moshier
Direct inquiries to 30 Frost Street, Cambridge, MA 02140
*/

/* Single precision natural logarithm
 * test interval: [sqrt(2)/2, sqrt(2)]
 * trials: 10000
 * peak relative error: 7.1e-8
 * rms relative error: 2.7e-8
 */

static const float SQRTHF = 0.707106781186547524f;

// Doesn't handle NaN, exp, inf, zeros or negative numbers

float logf_coef[9] = { 
      7.0376836292E-2f,
    - 1.1514610310E-1f,
    + 1.1676998740E-1f,
    - 1.2420140846E-1f,
    + 1.4249322787E-1f,
    - 1.6668057665E-1f,
    + 2.0000714765E-1f,
    - 2.4999993993E-1f,
    + 3.3333331174E-1f };


inline v4sf sse2_logf_unsafe(v4sf xx)
{
    v4sf y, x, z;
    v4si e;

    x = xx;
    x = sse2_frexpf_unsafe_nz_nodn( x, e );

    v4sf ltsqrthf = (v4sf)__builtin_ia32_cmpltps(x, vec_splat(SQRTHF));
    v4si minusone = (v4si)__builtin_ia32_andps(ltsqrthf, (v4sf)vec_splat(-1));
    v4sf xmasked  = __builtin_ia32_andps(ltsqrthf, x);

    x = x + xmasked;
    x = x - vec_splat(1.0f);
    e = e + minusone;

    z = x * x;

    y = polevl_1(x, logf_coef, 8) * z;

    v4sf fe = __builtin_ia32_cvtdq2ps(e);
    y += vec_splat(-2.12194440e-4f) * fe;
    y += vec_splat(-0.5f) * z;
    z = x + y;
    z += vec_splat(0.693359375f) * fe;

    return z;
}

inline v4sf sse2_logf(v4sf x)
{
    int mask = 0;

    // For out of range results, we have to use the other values
    if (JML_UNLIKELY(mask = out_of_range_mask_oo(x, 0.0f, INFINITY))) {
        //using namespace std;
        //cerr << "mask = " << mask << " x = " << x << endl;

        v4sf unsafe_result = vec_splat(0.0f);
        if (mask != 15)
            unsafe_result = sse2_logf_unsafe(x);
        float xin[4];
        unpack(x, xin);
        
        float xout[4];
        unpack(unsafe_result, xout);
        
        for (unsigned i = 0;  i < 4;  ++i)
            if (mask & (1 << i))
                xout[i] = logf(xin[i]);

        return pass_nan(x, pack(xout));
    }

    return pass_nan(x, sse2_logf_unsafe(x));
}

v2df sse2_log(v2df val)
{
    double f[2];
    *(v2df *)f = val;

    f[0] = log(f[0]);
    f[1] = log(f[1]);

    return *(v2df *)f;
}

} // namespace SIMD
} // namespace ML

#endif /* __jml__arch__sse2_log_h__ */

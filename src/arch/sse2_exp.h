/* sse_expf.h                                                      -*- C++ -*-
   Jeremy Barnes, 18 January 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.
*/


/* SIMD (SSE1+MMX or SSE2) implementation of exp

   Inspired by Intel Approximate Math library, and based on the
   corresponding algorithms of the cephes math library
*/

/* Copyright (C) 2007  Julien Pommier

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

#include <xmmintrin.h>
#include <emmintrin.h>
#include "sse2.h"
#include "sse2_math.h"
#include "sse2_misc.h"
#include "sse2_poly.h"
#include <cmath>

namespace ML {
namespace SIMD {

/*****************************************************************************/
/* SINGLE PRECISION EXP                                                      */
/*****************************************************************************/

static const v4sf float_1             = vec_splat(1.0f);
static const v4sf float_0p5           = vec_splat(0.5f);
static const v4si int_0x7f            = vec_splat(0x7f);


static const v4sf float_exp_hi        = vec_splat(88.3762626647949f);
static const v4sf float_exp_lo        = vec_splat(-88.3762626647949f);

static const v4sf float_cephes_LOG2EF = vec_splat(1.44269504088896341f);
static const v4sf float_cephes_exp_C1 = vec_splat(0.693359375f);
static const v4sf float_cephes_exp_C2 = vec_splat(-2.12194440e-4f);

static const v4sf float_cephes_exp_p0 = vec_splat(1.9875691500E-4f);
static const v4sf float_cephes_exp_p1 = vec_splat(1.3981999507E-3f);
static const v4sf float_cephes_exp_p2 = vec_splat(8.3334519073E-3f);
static const v4sf float_cephes_exp_p3 = vec_splat(4.1665795894E-2f);
static const v4sf float_cephes_exp_p4 = vec_splat(1.6666665459E-1f);
static const v4sf float_cephes_exp_p5 = vec_splat(5.0000001201E-1f);

//static const float MAXLOGF = 88.72283905206835f;
//static const float MINLOGF = -103.278929903431851103f; /* log(2^-149) */

static const float MAXLOGF = 88.3762626647949f;
static const float MINLOGF = -87.5;

static const v4sf float_cephes_MAXLOGF = vec_splat(MAXLOGF);
static const v4sf float_cephes_MINLOGF = vec_splat(MINLOGF);

/* TODO: problems to fix up some day:
   1.  If we remove the clamping to float_exp_lo, then we get some crazy
       values (eg, -4e38 for an input of -100.0).
   2.  The floor function doesn't do the same thing as cephes.
*/

inline v4sf sse2_expf_unsafe(v4sf x)
{
    v4sf tmp = _mm_setzero_ps(), fx;
    v4si emm0;
    v4sf one = float_1;

    //x = _mm_min_ps(x, float_exp_hi);
    x = _mm_max_ps(x, float_exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
    fx = x * float_cephes_LOG2EF;
    fx = fx + float_0p5;

    /* how to perform a floorf with SSE: just below */
    tmp  = sse2_floor_unsafe(fx);

    /* if greater, subtract 1 */
    v4sf mask = _mm_cmpgt_ps(tmp, fx);    
    mask = _mm_and_ps(mask, one);
    fx = _mm_sub_ps(tmp, mask);

    tmp = fx * float_cephes_exp_C1;
    v4sf z = fx * float_cephes_exp_C2;
    x = x - tmp;
    x = x - z;

    z = x * x;
  
    v4sf y = float_cephes_exp_p0;
    y = y * x;
    y = y + float_cephes_exp_p1;
    y = y * x;
    y = y + float_cephes_exp_p2;
    y = y * x;
    y = y + float_cephes_exp_p3;
    y = y * x;
    y = y + float_cephes_exp_p4;
    y = y * x;
    y = y + float_cephes_exp_p5;
    y = y * z;
    y = y + x;
    y = y + one;

    /* build 2^n */
    emm0 = (v4si)_mm_cvttps_epi32(fx);
    emm0 = emm0 + int_0x7f;
    emm0 = (v4si)_mm_slli_epi32((v2di)emm0, 23);
    v4sf pow2n = _mm_castsi128_ps((v2di)emm0);

    y = y * pow2n;

    return y;
}

inline v4sf sse2_expf(v4sf x)
{
    int mask = 0;

    // For out of range results, we have to use the other values
    if (JML_UNLIKELY(mask = out_of_range_mask_cc(x, MINLOGF, MAXLOGF))) {
        using namespace std;
        //cerr << "mask = " << mask << " x = " << x << endl;

        v4sf unsafe_result = vec_splat(0.0f);
        if (mask != 15)
            unsafe_result = sse2_expf_unsafe(x);
        float xin[4];
        unpack(x, xin);

        float xout[4];
        unpack(unsafe_result, xout);
        
        for (unsigned i = 0;  i < 4;  ++i)
            if (mask & (1 << i))
                xout[i] = expf(xin[i]);

        return pass_nan(x, pack(xout));
    }

    return pass_nan(x, sse2_expf_unsafe(x));
}

static double P[] = {
 1.26177193074810590878E-4,
 3.02994407707441961300E-2,
 9.99999999999999999910E-1,
};
static double Q[] = {
 3.00198505138664455042E-6,
 2.52448340349684104192E-3,
 2.27265548208155028766E-1,
 2.00000000000000000009E0,
};
static double C1 = 6.93145751953125E-1;
static double C2 = 1.42860682030941723212E-6;

double LOG2E  =  1.4426950408889634073599;     /* 1/log(2) */

#ifdef DENORMAL
double MAXLOG =  7.09782712893383996732E2;     /* log(MAXNUM) */
double MINLOG = -7.451332191019412076235E2;     /* log(2**-1075) */
#else
double MAXLOG =  7.08396418532264106224E2;     /* log 2**1022 */
double MINLOG = -7.08396418532264106224E2;     /* log 2**-1022 */
#endif


/*****************************************************************************/
/* DOUBLE PRECISION EXP                                                      */
/*****************************************************************************/

/*							exp.c
 *
 *	Exponential function
 *
 *
 *
 * SYNOPSIS:
 *
 * double x, y, exp();
 *
 * y = exp( x );
 *
 *
 *
 * DESCRIPTION:
 *
 * Returns e (2.71828...) raised to the x power.
 *
 * Range reduction is accomplished by separating the argument
 * into an integer k and fraction f such that
 *
 *     x    k  f
 *    e  = 2  e.
 *
 * A Pade' form  1 + 2x P(x**2)/( Q(x**2) - P(x**2) )
 * of degree 2/3 is used to approximate exp(f) in the basic
 * interval [-0.5, 0.5].
 *
 *
 * ACCURACY:
 *
 *                      Relative error:
 * arithmetic   domain     # trials      peak         rms
 *    DEC       +- 88       50000       2.8e-17     7.0e-18
 *    IEEE      +- 708      40000       2.0e-16     5.6e-17
 *
 *
 * Error amplification in the exponential function can be
 * a serious matter.  The error propagation involves
 * exp( X(1+delta) ) = exp(X) ( 1 + X*delta + ... ),
 * which shows that a 1 lsb error in representing X produces
 * a relative error of X times 1 lsb in the function.
 * While the routine gives an accurate result for arguments
 * that are exactly represented by a double precision
 * computer number, the result contains amplified roundoff
 * error for large arguments not exactly represented.
 *
 *
 * ERROR MESSAGES:
 *
 *   message         condition      value returned
 * exp underflow    x < MINLOG         0.0
 * exp overflow     x > MAXLOG         INFINITY
 *
 */
/*
Cephes Math Library Release 2.8:  June, 2000
Copyright 1984, 1995, 2000 by Stephen L. Moshier
*/

inline v2df sse2_exp_unsafe(v2df x)
{
    /* Express e**x = e**g 2**n
     *   = e**g e**( n loge(2) )
     *   = e**( g + n loge(2) )
     */
    /* floor() truncates toward -infinity. */
    v2df px = sse2_floor_unsafe( vec_splat(LOG2E) * x + vec_splat(0.5) );

    v4si n = __builtin_ia32_cvtpd2dq(px);

    x -= px * vec_splat(C1);
    x -= px * vec_splat(C2);

    /* rational approximation for exponential
     * of the fractional part:
     * e**x = 1 + 2x P(x**2)/( Q(x**2) - P(x**2) )
     */
    v2df xx = x * x;

    px = x * polevl( xx, P, 2 );
    x =  px/( polevl( xx, Q, 3 ) - px );
    x = vec_splat(1.0) + (x + x);

    /* multiply by power of 2 */
    x = ldexp( x, n );
    return x;
}

inline v2df sse2_exp(v2df x)
{
    int mask = 0;

    // For out of range results, we have to use the other values
    if (JML_UNLIKELY(mask = out_of_range_mask_cc(x, MINLOG, MAXLOG))) {
        //using namespace std;
        //cerr << "mask = " << mask << " x = " << x << endl;

        v2df unsafe_result = vec_splat(0.0);
        if (mask != 3)
            unsafe_result = sse2_exp_unsafe(x);
        double xin[2];
        unpack(x, xin);

        double xout[2];
        unpack(unsafe_result, xout);
        
        if (mask & 1) xout[0] = exp(xin[0]);
        if (mask & 2) xout[1] = exp(xin[1]);

        return pass_nan(x, pack(xout));
    }

    return pass_nan(x, sse2_exp_unsafe(x));
}


} // namespace SIMD
} // namespace ML

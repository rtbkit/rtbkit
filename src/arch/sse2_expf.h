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
#include <cmath>

namespace ML {
namespace SIMD {

inline v4sf pass_nan(v4sf input, v4sf result)
{
    v4sf mask_nan = (v4sf)__builtin_ia32_cmpunordps(input, input);
    input = __builtin_ia32_andps(mask_nan, input);
    result = __builtin_ia32_andnps(mask_nan, result);
    result = __builtin_ia32_orps(result, input);

    return result;
}


inline v4sf sse2_trunc_unsafe(v4sf x)
{
    return __builtin_ia32_cvtdq2ps(__builtin_ia32_cvttps2dq(x));
}

inline v4sf sse2_floor_unsafe(v4sf x)
{
    return __builtin_ia32_cvtdq2ps(__builtin_ia32_cvtps2dq(x));
}

inline v4sf sse2_floor(v4sf x)
{
    return pass_nan(x, sse2_floor_unsafe(x));
}

inline v4sf sse2_trunc(v4sf x)
{
    return pass_nan(x, sse2_trunc_unsafe(x));
}


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

inline int out_of_range_mask(v4sf input, v4sf min_val, v4sf max_val)
{
    v4sf mask_too_low  = (v4sf)__builtin_ia32_cmpltps(input, min_val);
    v4sf mask_too_high = (v4sf)__builtin_ia32_cmpgtps(input, max_val);

    return __builtin_ia32_movmskps(__builtin_ia32_orps(mask_too_low,
                                                       mask_too_high));
}

inline int out_of_range_mask(v4sf input, float min_val, float max_val)
{
    v4sf mask_too_low  = (v4sf)__builtin_ia32_cmpltps(input, vec_splat(min_val));
    v4sf mask_too_high = (v4sf)__builtin_ia32_cmpgtps(input, vec_splat(max_val));

    return __builtin_ia32_movmskps(__builtin_ia32_orps(mask_too_low,
                                                       mask_too_high));
}

inline void unpack(v4sf val, float * where)
{
    (*(v4sf *)where) = val;
}

inline v4sf pack(float * where)
{
    return *(v4sf *)where;
}

inline v4sf sse2_expf(v4sf x)
{
    int mask = 0;

    // For out of range results, we have to use the other values
    if (JML_UNLIKELY(mask = out_of_range_mask(x, MINLOGF, MAXLOGF))) {
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
        

#if 0

inline v2df sse2_floor(v2df x)
{
    return __builtin_ia32_cvtdq2pd(__builtin_ia32_cvttpd2dq(x));
}

inline v2df sse2_exp(v2df x)
{
    v2df tmp = _mm_setzero_pd(), fx;
    v4si emm0;
    v2df one = float_1;

    x = _mm_min_ps(x, float_exp_hi);
    x = _mm_max_ps(x, float_exp_lo);

    /* express exp(x) as exp(g + n*log(2)) */
    fx = x * float_cephes_LOG2EF;
    fx = fx + float_0p5;

    /* how to perform a floorf with SSE: just below */
    tmp  = sse2_floor(fx);

    /* if greater, substract 1 */
    v2df mask = _mm_cmpgt_ps(tmp, fx);    
    mask = _mm_and_ps(mask, one);
    fx = _mm_sub_ps(tmp, mask);

    tmp = fx * float_cephes_exp_C1;
    v2df z = fx * float_cephes_exp_C2;
    x = x - tmp;
    x = x - z;

    z = x * x;
  
    v2df y = float_cephes_exp_p0;
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
    emm0 = emm0 + _pi32_0x7f;
    emm0 = (v4si)_mm_slli_epi32((v2di)emm0, 23);
    v2df pow2n = _mm_castsi128_ps((v2di)emm0);

    y = y * pow2n;

    return y;
}

#endif

} // namespace SIMD
} // namespace ML

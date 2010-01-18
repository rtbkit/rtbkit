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

namespace ML {
namespace SIMD {


#define _PS_CONST(Name, Val) \
    static const v4sf _ps_##Name = vec_splat(Val)


#if 0
/* declare some SSE constants -- why can't I figure a better way to do that? */
#define _PS_CONST(Name, Val)                                            \
  static const ALIGN16_BEG float _ps_##Name[4] ALIGN16_END = { Val, Val, Val, Val }
#define _PI32_CONST(Name, Val)                                            \
  static const ALIGN16_BEG int _pi32_##Name[4] ALIGN16_END = { Val, Val, Val, Val }
#define _PS_CONST_TYPE(Name, Type, Val)                                 \
  static const ALIGN16_BEG Type _ps_##Name[4] ALIGN16_END = { Val, Val, Val, Val }
#endif



_PS_CONST(1  , 1.0f);
_PS_CONST(0p5, 0.5f);
static const v4si _pi32_0x7f = vec_splat(0x7f);

#if 0
/* the smallest non denormalized float number */
_PS_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS_CONST_TYPE(sign_mask, int, 0x80000000);
_PS_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PI32_CONST(1, 1);
_PI32_CONST(inv1, ~1);
_PI32_CONST(2, 2);
_PI32_CONST(4, 4);

_PS_CONST(cephes_SQRTHF, 0.707106781186547524);
_PS_CONST(cephes_log_p0, 7.0376836292E-2);
_PS_CONST(cephes_log_p1, - 1.1514610310E-1);
_PS_CONST(cephes_log_p2, 1.1676998740E-1);
_PS_CONST(cephes_log_p3, - 1.2420140846E-1);
_PS_CONST(cephes_log_p4, + 1.4249322787E-1);
_PS_CONST(cephes_log_p5, - 1.6668057665E-1);
_PS_CONST(cephes_log_p6, + 2.0000714765E-1);
_PS_CONST(cephes_log_p7, - 2.4999993993E-1);
_PS_CONST(cephes_log_p8, + 3.3333331174E-1);
_PS_CONST(cephes_log_q1, -2.12194440e-4);
_PS_CONST(cephes_log_q2, 0.693359375);

#endif




_PS_CONST(exp_hi,	88.3762626647949f);
_PS_CONST(exp_lo,	-88.3762626647949f);

_PS_CONST(cephes_LOG2EF, 1.44269504088896341f);
_PS_CONST(cephes_exp_C1, 0.693359375f);
_PS_CONST(cephes_exp_C2, -2.12194440e-4f);

_PS_CONST(cephes_exp_p0, 1.9875691500E-4f);
_PS_CONST(cephes_exp_p1, 1.3981999507E-3f);
_PS_CONST(cephes_exp_p2, 8.3334519073E-3f);
_PS_CONST(cephes_exp_p3, 4.1665795894E-2f);
_PS_CONST(cephes_exp_p4, 1.6666665459E-1f);
_PS_CONST(cephes_exp_p5, 5.0000001201E-1f);

inline v4sf sse2_floor(v4sf x)
{
    return __builtin_ia32_cvtdq2ps(__builtin_ia32_cvttps2dq(x));
}

inline v4sf sse2_expf(v4sf x)
{
  v4sf tmp = _mm_setzero_ps(), fx;
  v4si emm0;
  v4sf one = _ps_1;

  x = _mm_min_ps(x, _ps_exp_hi);
  x = _mm_max_ps(x, _ps_exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
  fx = x * _ps_cephes_LOG2EF;
  fx = fx + _ps_0p5;

  /* how to perform a floorf with SSE: just below */
  tmp  = sse2_floor(fx);

  /* if greater, substract 1 */
  v4sf mask = _mm_cmpgt_ps(tmp, fx);    
  mask = _mm_and_ps(mask, one);
  fx = _mm_sub_ps(tmp, mask);

  tmp = fx * _ps_cephes_exp_C1;
  v4sf z = fx * _ps_cephes_exp_C2;
  x = x - tmp;
  x = x - z;

  z = x * x;
  
  v4sf y = _ps_cephes_exp_p0;
  y = y * x;
  y = y + _ps_cephes_exp_p1;
  y = y * x;
  y = y + _ps_cephes_exp_p2;
  y = y * x;
  y = y + _ps_cephes_exp_p3;
  y = y * x;
  y = y + _ps_cephes_exp_p4;
  y = y * x;
  y = y + _ps_cephes_exp_p5;
  y = y * z;
  y = y + x;
  y = y + one;

  /* build 2^n */
  emm0 = (v4si)_mm_cvttps_epi32(fx);
  emm0 = emm0 + _pi32_0x7f;
  emm0 = (v4si)_mm_slli_epi32((v2di)emm0, 23);
  v4sf pow2n = _mm_castsi128_ps((v2di)emm0);

  y = y * pow2n;

  return y;
}


} // namespace SIMD
} // namespace ML

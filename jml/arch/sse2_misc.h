/* sse2_misc.h                                                     -*- C++ -*-
   Jeremy Barnes, 23 January 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

   SSE2 miscellaneous functions.
*/

#ifndef __jml__arch__sse2_misc_h__
#define __jml__arch__sse2_misc_h__

#include "sse2.h"
#include <cmath>

namespace ML {
namespace SIMD {


inline v2df pass_nan(v2df input, v2df result)
{
    v2df mask_nan = (v2df)__builtin_ia32_cmpunordpd(input, input);
    input = __builtin_ia32_andpd(mask_nan, input);
    result = __builtin_ia32_andnpd(mask_nan, result);
    result = __builtin_ia32_orpd(result, input);
    return result;
}

inline v2df pass_nan_inf_zero(v2df input, v2df result)
{
    v2df mask = (v2df)__builtin_ia32_cmpunordpd(input, input);
    mask = __builtin_ia32_orpd(mask, (v2df)__builtin_ia32_cmpeqpd(input, vec_splat(double(-INFINITY))));
    mask = __builtin_ia32_orpd(mask, (v2df)__builtin_ia32_cmpeqpd(input, vec_splat(double(INFINITY))));
    mask = __builtin_ia32_orpd(mask, (v2df)__builtin_ia32_cmpeqpd(input, vec_splat(0.0)));

    input = __builtin_ia32_andpd(mask, input);
    result = __builtin_ia32_andnpd(mask, result);
    result = __builtin_ia32_orpd(result, input);
    return result;
}

inline int out_of_range_mask_cc(v2df input, v2df min_val, v2df max_val)
{
    v2df mask_too_low  = (v2df)__builtin_ia32_cmpltpd(input, min_val);
    v2df mask_too_high = (v2df)__builtin_ia32_cmpgtpd(input, max_val);

    return __builtin_ia32_movmskpd(__builtin_ia32_orpd(mask_too_low,
                                                       mask_too_high));
}

inline int out_of_range_mask_cc(v2df input, double min_val, double max_val)
{
    return out_of_range_mask_cc(input, vec_splat(min_val), vec_splat(max_val));
}

inline int out_of_range_mask_oo(v2df input, v2df min_val, v2df max_val)
{
    v2df mask_too_low  = (v2df)__builtin_ia32_cmplepd(input, min_val);
    v2df mask_too_high = (v2df)__builtin_ia32_cmpgepd(input, max_val);

    return __builtin_ia32_movmskpd(__builtin_ia32_orpd(mask_too_low,
                                                       mask_too_high));
}

inline int out_of_range_mask_oo(v2df input, double min_val, double max_val)
{
    return out_of_range_mask_oo(input, vec_splat(min_val), vec_splat(max_val));
}


inline int out_of_range_mask_cc(v4sf input, v4sf min_val, v4sf max_val)
{
    v4sf mask_too_low  = (v4sf)__builtin_ia32_cmpltps(input, min_val);
    v4sf mask_too_high = (v4sf)__builtin_ia32_cmpgtps(input, max_val);

    return __builtin_ia32_movmskps(__builtin_ia32_orps(mask_too_low,
                                                       mask_too_high));
}

inline int out_of_range_mask_cc(v4sf input, float min_val, float max_val)
{
    return out_of_range_mask_cc(input, vec_splat(min_val), vec_splat(max_val));
}

inline int out_of_range_mask_oo(v4sf input, v4sf min_val, v4sf max_val)
{
    v4sf mask_too_low  = (v4sf)__builtin_ia32_cmpleps(input, min_val);
    v4sf mask_too_high = (v4sf)__builtin_ia32_cmpgeps(input, max_val);
    v4sf mask_or = __builtin_ia32_orps(mask_too_low, mask_too_high);
    int result = __builtin_ia32_movmskps(mask_or);
    return result;
}

inline int out_of_range_mask_oo(v4sf input, float min_val, float max_val)
{
    return out_of_range_mask_oo(input, vec_splat(min_val), vec_splat(max_val));
}

inline v4sf pass_nan(v4sf input, v4sf result)
{
    v4sf mask_nan = (v4sf)__builtin_ia32_cmpunordps(input, input);
    input = __builtin_ia32_andps(mask_nan, input);
    result = __builtin_ia32_andnps(mask_nan, result);
    result = __builtin_ia32_orps(result, input);

    return result;
}

inline v4sf pass_nan_inf(v4sf input, v4sf result)
{
    v4sf mask = (v4sf)__builtin_ia32_cmpunordps(input, input);
    mask = __builtin_ia32_orps(mask, (v4sf)__builtin_ia32_cmpeqps(input, vec_splat(-INFINITY)));
    mask = __builtin_ia32_orps(mask, (v4sf)__builtin_ia32_cmpeqps(input, vec_splat(INFINITY)));

    input = __builtin_ia32_andps(mask, input);
    result = __builtin_ia32_andnps(mask, result);
    result = __builtin_ia32_orps(result, input);
    return result;
}

inline v4sf pass_nan_inf_zero(v4sf input, v4sf result)
{
    v4sf mask = (v4sf)__builtin_ia32_cmpunordps(input, input);
    mask = __builtin_ia32_orps(mask, (v4sf)__builtin_ia32_cmpeqps(input, vec_splat(-INFINITY)));
    mask = __builtin_ia32_orps(mask, (v4sf)__builtin_ia32_cmpeqps(input, vec_splat(INFINITY)));
    mask = __builtin_ia32_orps(mask, (v4sf)__builtin_ia32_cmpeqps(input, vec_splat(0.0f)));

    input = __builtin_ia32_andps(mask, input);
    result = __builtin_ia32_andnps(mask, result);
    result = __builtin_ia32_orps(result, input);
    return result;
}

} // namespace SIMD
} // namespace ML

#endif /* __jml__arch__sse2_misc_h__ */

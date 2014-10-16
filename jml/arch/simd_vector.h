/* simd_vector.h                                                   -*- C++ -*-
   Jeremy Barnes, 1 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   
   This file is part of "Jeremy's Machine Learning Library", copyright (c)
   1999-2005 Jeremy Barnes.
   
   This program is available under the GNU General Public License, the terms
   of which are given by the file "license.txt" in the top level directory of
   the source code distribution.  If this file is missing, you have no right
   to use the program; please contact the author.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
   for more details.

   ---

   Generic SIMD vectorized loop kernels.
*/

#ifndef __arch__simd_vector_h__
#define __arch__simd_vector_h__

#include "simd.h"
#include "jml/arch/arch.h"

namespace ML {

namespace SIMD {
namespace Generic {

/* Float versions */
void vec_scale(const float * x, float factor, float * r, size_t n);
void vec_add(const float * x, const float * y, float * r, size_t n);
void vec_add(const float * x, float k, const float * y, float * r, size_t n);

// r = x + k y^2
void vec_add_sqr(const float * x, float k, const float * y, float * r, size_t n);

// r = x + k y
void vec_add(const float * x, const float * k, const float * y, float * r,
             size_t n);
void vec_add(const float * x, const double * k, const double * y, float * r,
             size_t n);
void vec_add(const float * x, const float * k, const double * y, float * r,
             size_t n);
inline void vec_add(const float * x, const double * k, const float * y,
                    float * r, size_t n)
{
    vec_add(x, y, k, r, n);
}

void vec_prod(const float * x, const float * y, float * r, size_t n);
float vec_dotprod(const float * x, const float * y, size_t n);
void vec_minus(const float * x, const float * y, float * r, size_t n);
double vec_accum_prod3(const float * x, const float * y, const float * z,
                       size_t n);

double vec_accum_prod3(const float * x, const float * y, const double * z,
                       size_t n);

// set r = k1 x + k2 y z
void vec_k1_x_plus_k2_y_z(float k1, const float * x,
                          float k2, const float * y, const float * z,
                          float * r, size_t n);

/* Double versions */
void vec_scale(const double * x, double factor, double * r, size_t n);
void vec_add(const double * x, const double * y, double * r, size_t n);
void vec_add(const double * x, double k, const double * y, double * r,
             size_t n);

// r = x + k y^2
void vec_add_sqr(const double * x, double k, const double * y, double * r,
                 size_t n);
// r = x + k y
void vec_add(const double * x, const double * k, const double * y, double * r,
             size_t n);
void vec_add(const double * x, const float * k, const float * y, double * r,
             size_t n);
void vec_add(const double * x, const float * k, const double * y, double * r,
             size_t n);
inline void vec_add(const double * x, const double * k, const float * y,
                    double * r, size_t n)
{
    return vec_add(x, y, k, r, n);
}

void vec_prod(const double * x, const double * y, double * r, size_t n);
double vec_dotprod(const double * x, const double * y, size_t n);
void vec_minus(const double * x, const double * y, double * r, size_t n);
double vec_accum_prod3(const double * x, const double * y, const double * z,
                       size_t n);
double vec_accum_prod3(const double * x, const double * y, const float * z,
                       size_t n);

// set r = k1 x + k2 y z
void vec_k1_x_plus_k2_y_z(double k1, const double * x,
                          double k2, const double * y, const double * z,
                          double * r, size_t n);

double vec_sum(const double * x, size_t n);

/* Mixed versions */
void vec_add(const float * x, float k, const double * y, float * r, size_t n);
void vec_add_sqr(const float * x, float k, const double * y, float * r, size_t n);

double vec_dotprod_dp(const double * x, const float * y, size_t n);
JML_ALWAYS_INLINE
double vec_dotprod_dp(const float * x, const double * y, size_t n)
{
    return vec_dotprod_dp(y, x, n);
}

void vec_prod(const double * x, const float * y, double * r, size_t n);
JML_ALWAYS_INLINE void vec_prod(const float * x, const double * y, double * r, size_t n)
{
    vec_prod(y, x, r, n);
}

void vec_prod(const float * x, const double * y, float * r, size_t n);

void vec_prod(const double * x, const double * y, float * r, size_t n);

void vec_add(const double * x, const float * y, double * r, size_t n);


/* Floating point using double precision accumulation */
double vec_dotprod_dp(const float * x, const float * y, size_t n);
double vec_sum_dp(const float * x, size_t n);
void vec_add(const double * x, double k, const float * y, double * r,
             size_t n);

void vec_add_sqr(const double * x, double k, const float * y, double * r,
                 size_t n);

inline double vec_sum_dp(const double * x, size_t n)
{
    return vec_sum(x, n);
}

inline double vec_dotprod_dp(const double * x, const double * y, size_t n)
{
    return vec_dotprod(x, y, n);
}

// Exponential.  The versions with a k parameter return exp(kx).  The type of
// the result controls the precision in which the calculation is performed.
void vec_exp(const float * x, float * r, size_t n);
void vec_exp(const float * x, float k, float * r, size_t n);
void vec_exp(const float * x, double * r, size_t n);
void vec_exp(const float * x, double k, double * r, size_t n);
void vec_exp(const double * x, double * r, size_t n);
void vec_exp(const double * x, double k, double * r, size_t n);

// Maximum
void vec_max(const float * x, const float * y, float * r, size_t n);
void vec_max(const float * x, float y, float * r, size_t n);
void vec_max(const double * x, const double * y, double * r, size_t n);
void vec_max(const double * x, double y, double * r, size_t n);

// Minimum
void vec_min(const float * x, const float * y, float * r, size_t n);
void vec_min(const float * x, float y, float * r, size_t n);
void vec_min(const double * x, const double * y, double * r, size_t n);
void vec_min(const double * x, double y, double * r, size_t n);

// Square of the two norm (sum x_i^2)
// Equivalent to vec_dotprod(x, x, n)
float vec_twonorm_sqr(const float * x, size_t n);
double vec_twonorm_sqr_dp(const float * x, size_t n);
double vec_twonorm_sqr(const double * x, size_t n);
inline double vec_twonorm_sqr_dp(const double * x, size_t n)
{
    return vec_twonorm_sqr(x, n);
}

// KL divergence: kl = sum(p * log(p / q))
double vec_kl(const float * p, const float * q, size_t n);

// Simultaneous min and max
void vec_min_max_el(const float * x, float * mins, float * maxs, size_t n);

} // namespace Generic

#if JML_USE_SSE1

namespace SSE1 {
} // namespace SSE1

namespace SSE2 {
} // namespace SSE2

namespace SSE3 {
} // namespace SSE3

#endif /* __i686 */

using namespace Generic;

} // namespace SIMD
} // namespace ML


#endif /* __arch__simd_vector_h__ */


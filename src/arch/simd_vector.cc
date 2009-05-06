/* simd_vector.cc
   Jeremy Barnes, 15 March 2005
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

   Contains generic implementations (non-vectorised) of SIMD functionality.
*/

#include "simd_vector.h"

namespace ML {
namespace SIMD {
namespace Generic {

void vec_scale(const float * x, float factor, float * r, size_t n)
{
    for (unsigned i = 0;  i < n;  ++i) r[i] = x[i] * factor;
}

void vec_add(const float * x, const float * y, float * r, size_t n)
{
    for (unsigned i = 0;  i < n;  ++i) r[i] = x[i] + y[i];
}

typedef float v4sf __attribute__((__vector_size__(16)));

JML_ALWAYS_INLINE v4sf vec_splat(float val)
{
    v4sf result = {val, val, val, val};

    return result;
}

void vec_add(const float * x, float k, const float * y, float * r, size_t n)
{
    v4sf kkkk = vec_splat(k);

    while (n > 16) {
        const v4sf * xx = reinterpret_cast<const v4sf *>(x);
        const v4sf * yy = reinterpret_cast<const v4sf *>(y);
        v4sf * rr = reinterpret_cast<v4sf *>(r);

        v4sf yyyy0 = yy[0];
        v4sf xxxx0 = xx[0];
        yyyy0 *= kkkk;
        v4sf yyyy1 = yy[1];
        yyyy0 += xxxx0;
        v4sf xxxx1 = xx[1];
        rr[0] = yyyy0;
        yyyy1 *= kkkk;
        v4sf yyyy2 = yy[2];
        yyyy1 += xxxx1;
        v4sf xxxx2 = xx[2];
        rr[1] = yyyy1;
        yyyy2 *= kkkk;
        v4sf yyyy3 = yy[3];
        yyyy2 += xxxx2;
        v4sf xxxx3 = xx[3];
        rr[2] = yyyy2;
        yyyy3 *= kkkk;
        yyyy3 += xxxx3;
        rr[3] = yyyy3;

#if 0
        r[0]  = x[0]  + k * y[0];
        r[1]  = x[1]  + k * y[1];
        r[2]  = x[2]  + k * y[2];
        r[3]  = x[3]  + k * y[3];

        r[4]  = x[4]  + k * y[4];
        r[5]  = x[5]  + k * y[5];
        r[6]  = x[6]  + k * y[6];
        r[7]  = x[7]  + k * y[7];

        r[8]  = x[8]  + k * y[8];
        r[9]  = x[9]  + k * y[9];
        r[10] = x[10] + k * y[10];
        r[11] = x[11] + k * y[11];

        r[12] = x[12] + k * y[12];
        r[13] = x[13] + k * y[13];
        r[14] = x[14] + k * y[14];
        r[15] = x[15] + k * y[15];
#endif

        r += 16;  x += 16;  y += 16;  n -= 16;
    }

    for (unsigned i = 0;  i < n;  ++i) r[i] = x[i] + k * y[i];
}

float vec_dotprod(const float * x, const float * y, size_t n)
{
    double res = 0.0;
    for (unsigned i = 0;  i < n;  ++i) res += x[i] * y[i];
    return res;
}

void vec_scale(const double * x, double factor, double * r, size_t n)
{
    for (unsigned i = 0;  i < n;  ++i) r[i] = x[i] * factor;
}

void vec_add(const double * x, const double * y, double * r, size_t n)
{
    for (unsigned i = 0;  i < n;  ++i) r[i] = x[i] + y[i];
}

void vec_add(const double * x, double k, const double * y, double * r,
             size_t n)
{
    for (unsigned i = 0;  i < n;  ++i) r[i] = x[i] + k * y[i];
}

double vec_dotprod(const double * x, const double * y, size_t n)
{
    double res = 0.0;
    for (unsigned i = 0;  i < n;  ++i) res += x[i] * y[i];
    return res;
}

void vec_minus(const float * x, const float * y, float * r, size_t n)
{
    for (unsigned i = 0;  i < n;  ++i) r[i] = x[i] - y[i];
}

double vec_accum_prod3(const float * x, const float * y, const float * z,
                       size_t n)
{
    double res = 0.0;
    for (unsigned i = 0;  i < n;  ++i) res += x[i] * y[i] * z[i];
    return res;
}

void vec_minus(const double * x, const double * y, double * r, size_t n)
{
    for (unsigned i = 0;  i < n;  ++i) r[i] = x[i] - y[i];
}

double vec_accum_prod3(const double * x, const double * y, const double * z,
                      size_t n)
{
    double res = 0.0;
    for (unsigned i = 0;  i < n;  ++i) res += x[i] * y[i] * z[i];
    return res;
}

double vec_sum(const double * x, size_t n)
{
    double res = 0.0;
    for (unsigned i = 0;  i < n;  ++i)
        res += x[i];
    return res;
}

double vec_dotprod_dp(const float * x, const float * y, size_t n)
{
    double res = 0.0;
    for (unsigned i = 0;  i < n;  ++i) res += x[i] * y[i];
    return res;
}

double vec_sum_dp(const float * x, size_t n)
{
    double res = 0.0;
    for (unsigned i = 0;  i < n;  ++i)
        res += x[i];
    return res;
}

void vec_add(const double * x, double k, const float * y, double * r,
             size_t n)
{
    for (unsigned i = 0;  i < n;  ++i) r[i] = x[i] + k * y[i];
}

} // namespace Generic

} // namespace SIMD
} // namespace ML

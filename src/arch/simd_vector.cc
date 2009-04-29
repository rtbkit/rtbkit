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

void vec_add(const float * x, float k, const float * y, float * r, size_t n)
{
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

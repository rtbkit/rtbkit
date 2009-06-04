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
#include "compiler/compiler.h"
#include <iostream>

using namespace std;


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

template<typename X>
int ptr_align(const X * p) 
{
    return size_t(p) & 15;
} JML_PURE_FN

void vec_add(const float * x, float k, const float * y, float * r, size_t n)
{
    v4sf kkkk = vec_splat(k);
    unsigned i = 0;

    //bool alignment_unimportant = true;  // nehalem?

    if (false && n >= 16 && (ptr_align(x) == ptr_align(y) && ptr_align(y) == ptr_align(r))) {

        /* Align everything on 16 byte boundaries */
        if (ptr_align(x) != 0) {
            int needed_to_align = (16 - ptr_align(x)) / 4;
            
            for (unsigned i = 0;  i < needed_to_align;  ++i)
                r[i] = x[i] + k * y[i];

            r += needed_to_align;  x += needed_to_align;  y += needed_to_align;
            n -= needed_to_align;
        }

        //cerr << "optimized" << endl;

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

            r += 16;  x += 16;  y += 16;  n -= 16;
        }

        for (unsigned i = 0;  i < n;  ++i) r[i] = x[i] + k * y[i];
    }
    else {
        //cerr << "unoptimized" << endl;

        for (; i + 16 <= n;  i += 16) {
            v4sf yyyy0 = __builtin_ia32_loadups(y + i + 0);
            v4sf xxxx0 = __builtin_ia32_loadups(x + i + 0);
            yyyy0 *= kkkk;
            v4sf yyyy1 = __builtin_ia32_loadups(y + i + 4);
            yyyy0 += xxxx0;
            v4sf xxxx1 = __builtin_ia32_loadups(x + i + 4);
            __builtin_ia32_storeups(r + i + 0, yyyy0);
            yyyy1 *= kkkk;
            v4sf yyyy2 = __builtin_ia32_loadups(y + i + 8);
            yyyy1 += xxxx1;
            v4sf xxxx2 = __builtin_ia32_loadups(x + i + 8);
            __builtin_ia32_storeups(r + i + 4, yyyy1);
            yyyy2 *= kkkk;
            v4sf yyyy3 = __builtin_ia32_loadups(y + i + 12);
            yyyy2 += xxxx2;
            v4sf xxxx3 = __builtin_ia32_loadups(x + i + 12);
            __builtin_ia32_storeups(r + i + 8, yyyy2);
            yyyy3 *= kkkk;
            yyyy3 += xxxx3;
            __builtin_ia32_storeups(r + i + 12, yyyy3);
        }

        for (; i + 4 <= n;  i += 4) {
            v4sf yyyy0 = __builtin_ia32_loadups(y + i + 0);
            v4sf xxxx0 = __builtin_ia32_loadups(x + i + 0);
            yyyy0 *= kkkk;
            yyyy0 += xxxx0;
            __builtin_ia32_storeups(r + i + 0, yyyy0);
        }

        for (; i < n;  ++i) r[i] = x[i] + k * y[i];
    }
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

typedef double v2df __attribute__((__vector_size__(16)));

JML_ALWAYS_INLINE v2df vec_splat(double val)
{
    v2df result = {val, val};
    return result;
}

double vec_dotprod_dp(const float * x, const float * y, size_t n)
{
    if (true) {
        v2df rr = vec_splat(0.0);

        unsigned i = 0;

        for (; i + 16 <= n;  i += 16) {
            v4sf yyyy0 = __builtin_ia32_loadups(y + i + 0);
            v4sf xxxx0 = __builtin_ia32_loadups(x + i + 0);
            yyyy0 *= xxxx0;
            v2df dd0a = __builtin_ia32_cvtps2pd(yyyy0);
            yyyy0 = __builtin_ia32_shufps(yyyy0, yyyy0, 14);
            v2df dd0b = __builtin_ia32_cvtps2pd(yyyy0);
            rr += dd0a;
            rr += dd0b;

            v4sf yyyy1 = __builtin_ia32_loadups(y + i + 4);
            v4sf xxxx1 = __builtin_ia32_loadups(x + i + 4);
            yyyy1 *= xxxx1;
            v2df dd1a = __builtin_ia32_cvtps2pd(yyyy1);
            yyyy1 = __builtin_ia32_shufps(yyyy1, yyyy1, 14);
            v2df dd1b = __builtin_ia32_cvtps2pd(yyyy1);
            rr += dd1a;
            rr += dd1b;
            
            v4sf yyyy2 = __builtin_ia32_loadups(y + i + 8);
            v4sf xxxx2 = __builtin_ia32_loadups(x + i + 8);
            yyyy2 *= xxxx2;
            v2df dd2a = __builtin_ia32_cvtps2pd(yyyy2);
            yyyy2 = __builtin_ia32_shufps(yyyy2, yyyy2, 14);
            v2df dd2b = __builtin_ia32_cvtps2pd(yyyy2);
            rr += dd2a;
            rr += dd2b;

            v4sf yyyy3 = __builtin_ia32_loadups(y + i + 12);
            v4sf xxxx3 = __builtin_ia32_loadups(x + i + 12);
            yyyy3 *= xxxx3;
            v2df dd3a = __builtin_ia32_cvtps2pd(yyyy3);
            yyyy3 = __builtin_ia32_shufps(yyyy3, yyyy3, 14);
            v2df dd3b = __builtin_ia32_cvtps2pd(yyyy3);
            rr += dd3a;
            rr += dd3b;
        }

        for (; i + 4 <= n;  i += 4) {
            v4sf yyyy0 = __builtin_ia32_loadups(y + i + 0);
            v4sf xxxx0 = __builtin_ia32_loadups(x + i + 0);
            yyyy0 *= xxxx0;
            
            v2df dd1 = __builtin_ia32_cvtps2pd(yyyy0);
            yyyy0 = __builtin_ia32_shufps(yyyy0, yyyy0, 14);
            v2df dd2 = __builtin_ia32_cvtps2pd(yyyy0);
            rr += dd1;
            rr += dd2;
        }

        double results[2];
        *(v2df *)results = rr;

        double result = results[0] + results[1];

        for (; i < n;  ++i) result += x[i] * y[i];

        return result;
    }
        

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

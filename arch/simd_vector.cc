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

#include "exception.h"
#include "simd_vector.h"
#include "jml/compiler/compiler.h"
#include <iostream>
#include <cmath>
#include "sse2.h"
#include "sse2_exp.h"
#include "sse2_log.h"

using namespace std;


namespace ML {
namespace SIMD {
namespace Generic {

template<typename X>
int ptr_align(const X * p) 
{
    return size_t(p) & 15;
} JML_PURE_FN


void vec_scale(const float * x, float k, float * r, size_t n)
{
    v4sf kkkk = vec_splat(k);
    unsigned i = 0;

    if (false) ;
    else {
        for (; i + 16 <= n;  i += 16) {
            v4sf xxxx0 = __builtin_ia32_loadups(x + i + 0);
            xxxx0 *= kkkk;
            v4sf xxxx1 = __builtin_ia32_loadups(x + i + 4);
            __builtin_ia32_storeups(r + i + 0, xxxx0);
            xxxx1 *= kkkk;
            v4sf xxxx2 = __builtin_ia32_loadups(x + i + 8);
            __builtin_ia32_storeups(r + i + 4, xxxx1);
            xxxx2 *= kkkk;
            v4sf xxxx3 = __builtin_ia32_loadups(x + i + 12);
            __builtin_ia32_storeups(r + i + 8, xxxx2);
            xxxx3 *= kkkk;
            __builtin_ia32_storeups(r + i + 12, xxxx3);
        }
    
        for (; i + 4 <= n;  i += 4) {
            v4sf xxxx0 = __builtin_ia32_loadups(x + i + 0);
            xxxx0 *= kkkk;
            __builtin_ia32_storeups(r + i + 0, xxxx0);
        }
    }
    
    for (; i < n;  ++i) r[i] = k * x[i];
}

void vec_add(const float * x, const float * y, float * r, size_t n)
{
    unsigned i = 0;

    if (false) ;
    else {
        //cerr << "unoptimized" << endl;

        for (; i + 16 <= n;  i += 16) {
            v4sf yyyy0 = __builtin_ia32_loadups(y + i + 0);
            v4sf xxxx0 = __builtin_ia32_loadups(x + i + 0);
            v4sf yyyy1 = __builtin_ia32_loadups(y + i + 4);
            yyyy0 += xxxx0;
            v4sf xxxx1 = __builtin_ia32_loadups(x + i + 4);
            __builtin_ia32_storeups(r + i + 0, yyyy0);
            v4sf yyyy2 = __builtin_ia32_loadups(y + i + 8);
            yyyy1 += xxxx1;
            v4sf xxxx2 = __builtin_ia32_loadups(x + i + 8);
            __builtin_ia32_storeups(r + i + 4, yyyy1);
            v4sf yyyy3 = __builtin_ia32_loadups(y + i + 12);
            yyyy2 += xxxx2;
            v4sf xxxx3 = __builtin_ia32_loadups(x + i + 12);
            __builtin_ia32_storeups(r + i + 8, yyyy2);
            yyyy3 += xxxx3;
            __builtin_ia32_storeups(r + i + 12, yyyy3);
        }

        for (; i + 4 <= n;  i += 4) {
            v4sf yyyy0 = __builtin_ia32_loadups(y + i + 0);
            v4sf xxxx0 = __builtin_ia32_loadups(x + i + 0);
            yyyy0 += xxxx0;
            __builtin_ia32_storeups(r + i + 0, yyyy0);
        }
        
        for (; i < n;  ++i) r[i] = x[i] + y[i];
    }
}

void vec_prod(const float * x, const float * y, float * r, size_t n)
{
    unsigned i = 0;

    if (false) ;
    else {
        //cerr << "unoptimized" << endl;
        
        for (; i + 16 <= n;  i += 16) {
            v4sf yyyy0 = __builtin_ia32_loadups(y + i + 0);
            v4sf xxxx0 = __builtin_ia32_loadups(x + i + 0);
            v4sf yyyy1 = __builtin_ia32_loadups(y + i + 4);
            yyyy0 *= xxxx0;
            v4sf xxxx1 = __builtin_ia32_loadups(x + i + 4);
            __builtin_ia32_storeups(r + i + 0, yyyy0);
            v4sf yyyy2 = __builtin_ia32_loadups(y + i + 8);
            yyyy1 *= xxxx1;
            v4sf xxxx2 = __builtin_ia32_loadups(x + i + 8);
            __builtin_ia32_storeups(r + i + 4, yyyy1);
            v4sf yyyy3 = __builtin_ia32_loadups(y + i + 12);
            yyyy2 *= xxxx2;
            v4sf xxxx3 = __builtin_ia32_loadups(x + i + 12);
            __builtin_ia32_storeups(r + i + 8, yyyy2);
            yyyy3 *= xxxx3;
            __builtin_ia32_storeups(r + i + 12, yyyy3);
        }

        for (; i + 4 <= n;  i += 4) {
            v4sf yyyy0 = __builtin_ia32_loadups(y + i + 0);
            v4sf xxxx0 = __builtin_ia32_loadups(x + i + 0);
            yyyy0 *= xxxx0;
            __builtin_ia32_storeups(r + i + 0, yyyy0);
        }
        
        for (; i < n;  ++i) r[i] = x[i] * y[i];
    }
}

void vec_prod(const float * x, const double * y, float * r, size_t n)
{
    unsigned i = 0;

#if 0 // TODO: do
    if (false) ;
    else {
        //cerr << "unoptimized" << endl;
        
        for (; i + 16 <= n;  i += 16) {
            v2df yy0a  = __builtin_ia32_loadupd(y + i + 0);
            v2df yy0b  = __builtin_ia32_loadupd(y + i + 2);
            v4sf yyyy0 = __builtin_ia32_cvtpd2ps(yy0a);



            v4sf xxxx0 = __builtin_ia32_loadups(x + i + 0);
            v4sf yyyy1 = __builtin_ia32_loadups(y + i + 4);
            yyyy0 *= xxxx0;
            v4sf xxxx1 = __builtin_ia32_loadups(x + i + 4);
            __builtin_ia32_storeups(r + i + 0, yyyy0);
            v4sf yyyy2 = __builtin_ia32_loadups(y + i + 8);
            yyyy1 *= xxxx1;
            v4sf xxxx2 = __builtin_ia32_loadups(x + i + 8);
            __builtin_ia32_storeups(r + i + 4, yyyy1);
            v4sf yyyy3 = __builtin_ia32_loadups(y + i + 12);
            yyyy2 *= xxxx2;
            v4sf xxxx3 = __builtin_ia32_loadups(x + i + 12);
            __builtin_ia32_storeups(r + i + 8, yyyy2);
            yyyy3 *= xxxx3;
            __builtin_ia32_storeups(r + i + 12, yyyy3);
        }

        for (; i + 4 <= n;  i += 4) {
            v4sf yyyy0 = __builtin_ia32_loadups(y + i + 0);
            v4sf xxxx0 = __builtin_ia32_loadups(x + i + 0);
            yyyy0 *= xxxx0;
            __builtin_ia32_storeups(r + i + 0, yyyy0);
        }
        
    }
#endif

    for (; i < n;  ++i) r[i] = x[i] * y[i];
}

void vec_prod(const double * x, const double * y, float * r, size_t n)
{
    unsigned i = 0;
    for (; i < n;  ++i) r[i] = x[i] * y[i];
}

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

void vec_add(const float * x, const float * k, const float * y, float * r,
             size_t n)
{
    unsigned i = 0;

    if (true) {
        for (; i + 16 <= n;  i += 16) {
            v4sf yyyy0 = __builtin_ia32_loadups(y + i + 0);
            v4sf kkkk0 = __builtin_ia32_loadups(k + i + 0);
            v4sf xxxx0 = __builtin_ia32_loadups(x + i + 0);
            yyyy0 *= kkkk0;
            v4sf yyyy1 = __builtin_ia32_loadups(y + i + 4);
            v4sf kkkk1 = __builtin_ia32_loadups(k + i + 4);
            yyyy0 += xxxx0;
            v4sf xxxx1 = __builtin_ia32_loadups(x + i + 4);
            __builtin_ia32_storeups(r + i + 0, yyyy0);
            yyyy1 *= kkkk1;
            v4sf yyyy2 = __builtin_ia32_loadups(y + i + 8);
            v4sf kkkk2 = __builtin_ia32_loadups(k + i + 8);
            yyyy1 += xxxx1;
            v4sf xxxx2 = __builtin_ia32_loadups(x + i + 8);
            __builtin_ia32_storeups(r + i + 4, yyyy1);
            yyyy2 *= kkkk2;
            v4sf yyyy3 = __builtin_ia32_loadups(y + i + 12);
            v4sf kkkk3 = __builtin_ia32_loadups(k + i + 12);
            yyyy2 += xxxx2;
            v4sf xxxx3 = __builtin_ia32_loadups(x + i + 12);
            __builtin_ia32_storeups(r + i + 8, yyyy2);
            yyyy3 *= kkkk3;
            yyyy3 += xxxx3;
            __builtin_ia32_storeups(r + i + 12, yyyy3);
        }

        for (; i + 4 <= n;  i += 4) {
            v4sf yyyy0 = __builtin_ia32_loadups(y + i + 0);
            v4sf xxxx0 = __builtin_ia32_loadups(x + i + 0);
            v4sf kkkk0 = __builtin_ia32_loadups(k + i + 0);
            yyyy0 *= kkkk0;
            yyyy0 += xxxx0;
            __builtin_ia32_storeups(r + i + 0, yyyy0);
        }

        for (; i < n;  ++i) r[i] = x[i] + k[i] * y[i];
    }
}

void vec_add(const float * x, float k, const double * y, float * r, size_t n)
{
    for (unsigned i = 0; i < n;  ++i) r[i] = x[i] + k * y[i];
}

float vec_dotprod(const float * x, const float * y, size_t n)
{
    double res = 0.0;
    for (unsigned i = 0;  i < n;  ++i) res += x[i] * y[i];
    return res;
}

void vec_scale(const double * x, double k, double * r, size_t n)
{
    v2df kk = vec_splat(k);
    unsigned i = 0;

    if (false) ;
    else {
        for (; i + 8 <= n;  i += 8) {
            v2df xx0 = __builtin_ia32_loadupd(x + i + 0);
            xx0 *= kk;
            v2df xx1 = __builtin_ia32_loadupd(x + i + 2);
            __builtin_ia32_storeupd(r + i + 0, xx0);
            xx1 *= kk;
            v2df xx2 = __builtin_ia32_loadupd(x + i + 4);
            __builtin_ia32_storeupd(r + i + 2, xx1);
            xx2 *= kk;
            v2df xx3 = __builtin_ia32_loadupd(x + i + 6);
            __builtin_ia32_storeupd(r + i + 4, xx2);
            xx3 *= kk;
            __builtin_ia32_storeupd(r + i + 6, xx3);
        }
    
        for (; i + 2 <= n;  i += 2) {
            v2df xx0 = __builtin_ia32_loadupd(x + i + 0);
            xx0 *= kk;
            __builtin_ia32_storeupd(r + i + 0, xx0);
        }
    }
    
    for (; i < n;  ++i) r[i] = k * x[i];
}

void vec_add(const double * x, double k, const double * y, double * r,
             size_t n)
{
    v2df kk = vec_splat(k);
    unsigned i = 0;

    if (true) {
        for (; i + 8 <= n;  i += 8) {
            v2df yy0 = __builtin_ia32_loadupd(y + i + 0);
            v2df xx0 = __builtin_ia32_loadupd(x + i + 0);
            yy0 *= kk;
            yy0 += xx0;
            __builtin_ia32_storeupd(r + i + 0, yy0);

            v2df yy1 = __builtin_ia32_loadupd(y + i + 2);
            v2df xx1 = __builtin_ia32_loadupd(x + i + 2);
            yy1 *= kk;
            yy1 += xx1;
            __builtin_ia32_storeupd(r + i + 2, yy1);
            
            v2df yy2 = __builtin_ia32_loadupd(y + i + 4);
            v2df xx2 = __builtin_ia32_loadupd(x + i + 4);
            yy2 *= kk;
            yy2 += xx2;
            __builtin_ia32_storeupd(r + i + 4, yy2);

            v2df yy3 = __builtin_ia32_loadupd(y + i + 6);
            v2df xx3 = __builtin_ia32_loadupd(x + i + 6);
            yy3 *= kk;
            yy3 += xx3;
            __builtin_ia32_storeupd(r + i + 6, yy3);

        }

        for (; i + 2 <= n;  i += 2) {
            v2df yy0 = __builtin_ia32_loadupd(y + i + 0);
            v2df xx0 = __builtin_ia32_loadupd(x + i + 0);
            yy0 *= kk;
            yy0 += xx0;
            __builtin_ia32_storeupd(r + i + 0, yy0);
        }
    }

    for (;  i < n;  ++i) r[i] = x[i] + k * y[i];
}

void vec_add(const double * x, const double * k, const double * y,
             double * r, size_t n)
{
    unsigned i = 0;
    if (true) {
        for (; i + 8 <= n;  i += 8) {
            v2df yy0 = __builtin_ia32_loadupd(y + i + 0);
            v2df xx0 = __builtin_ia32_loadupd(x + i + 0);
            v2df kk0 = __builtin_ia32_loadupd(k + i + 0);
            yy0 *= kk0;
            yy0 += xx0;
            __builtin_ia32_storeupd(r + i + 0, yy0);

            v2df yy1 = __builtin_ia32_loadupd(y + i + 2);
            v2df xx1 = __builtin_ia32_loadupd(x + i + 2);
            v2df kk1 = __builtin_ia32_loadupd(k + i + 2);
            yy1 *= kk1;
            yy1 += xx1;
            __builtin_ia32_storeupd(r + i + 2, yy1);
            
            v2df yy2 = __builtin_ia32_loadupd(y + i + 4);
            v2df xx2 = __builtin_ia32_loadupd(x + i + 4);
            v2df kk2 = __builtin_ia32_loadupd(k + i + 4);
            yy2 *= kk2;
            yy2 += xx2;
            __builtin_ia32_storeupd(r + i + 4, yy2);

            v2df yy3 = __builtin_ia32_loadupd(y + i + 6);
            v2df xx3 = __builtin_ia32_loadupd(x + i + 6);
            v2df kk3 = __builtin_ia32_loadupd(k + i + 6);
            yy3 *= kk3;
            yy3 += xx3;
            __builtin_ia32_storeupd(r + i + 6, yy3);
        }

        for (; i + 2 <= n;  i += 2) {
            v2df yy0 = __builtin_ia32_loadupd(y + i + 0);
            v2df xx0 = __builtin_ia32_loadupd(x + i + 0);
            v2df kk0 = __builtin_ia32_loadupd(k + i + 0);
            yy0 *= kk0;
            yy0 += xx0;
            __builtin_ia32_storeupd(r + i + 0, yy0);
        }
    }

    for (;  i < n;  ++i) r[i] = x[i] + k[i] * y[i];
}

double vec_dotprod(const double * x, const double * y, size_t n)
{
    unsigned i = 0;
    double result = 0.0;

    if (true) {
        v2df rr = vec_splat(0.0);

        for (; i + 8 <= n;  i += 8) {
            v2df yy0 = __builtin_ia32_loadupd(y + i + 0);
            v2df xx0 = __builtin_ia32_loadupd(x + i + 0);
            yy0 *= xx0;
            rr += yy0;

            v2df yy1 = __builtin_ia32_loadupd(y + i + 2);
            v2df xx1 = __builtin_ia32_loadupd(x + i + 2);
            yy1 *= xx1;
            rr += yy1;
            
            v2df yy2 = __builtin_ia32_loadupd(y + i + 4);
            v2df xx2 = __builtin_ia32_loadupd(x + i + 4);
            yy2 *= xx2;
            rr += yy2;

            v2df yy3 = __builtin_ia32_loadupd(y + i + 6);
            v2df xx3 = __builtin_ia32_loadupd(x + i + 6);
            yy3 *= xx3;
            rr += yy3;
        }

        for (; i + 2 <= n;  i += 2) {
            v2df yy0 = __builtin_ia32_loadupd(y + i + 0);
            v2df xx0 = __builtin_ia32_loadupd(x + i + 0);
            yy0 *= xx0;
            rr += yy0;
        }

        double results[2];
        *(v2df *)results = rr;

        result = results[0] + results[1];
    }

    for (; i < n;  ++i) result += x[i] * y[i];

    return result;
}

void vec_minus(const float * x, const float * y, float * r, size_t n)
{
    for (unsigned i = 0;  i < n;  ++i) r[i] = x[i] - y[i];
}

double vec_accum_prod3(const float * x, const float * y, const float * z,
                       size_t n)
{
    double res = 0.0;
    unsigned i = 0;

    if (true) {
        v2df rr = vec_splat(0.0);

        for (; i + 16 <= n;  i += 16) {
            v4sf yyyy0 = __builtin_ia32_loadups(y + i + 0);
            v4sf xxxx0 = __builtin_ia32_loadups(x + i + 0);
            yyyy0 *= xxxx0;
            v4sf zzzz0 = __builtin_ia32_loadups(z + i + 0);
            yyyy0 *= zzzz0;
            v2df dd0a = __builtin_ia32_cvtps2pd(yyyy0);
            yyyy0 = __builtin_ia32_shufps(yyyy0, yyyy0, 14);
            v2df dd0b = __builtin_ia32_cvtps2pd(yyyy0);
            rr += dd0a;
            rr += dd0b;

            v4sf yyyy1 = __builtin_ia32_loadups(y + i + 4);
            v4sf xxxx1 = __builtin_ia32_loadups(x + i + 4);
            yyyy1 *= xxxx1;
            v4sf zzzz1 = __builtin_ia32_loadups(z + i + 4);
            yyyy1 *= zzzz1;
            v2df dd1a = __builtin_ia32_cvtps2pd(yyyy1);
            yyyy1 = __builtin_ia32_shufps(yyyy1, yyyy1, 14);
            v2df dd1b = __builtin_ia32_cvtps2pd(yyyy1);
            rr += dd1a;
            rr += dd1b;
            
            v4sf yyyy2 = __builtin_ia32_loadups(y + i + 8);
            v4sf xxxx2 = __builtin_ia32_loadups(x + i + 8);
            yyyy2 *= xxxx2;
            v4sf zzzz2 = __builtin_ia32_loadups(z + i + 8);
            yyyy2 *= zzzz2;
            v2df dd2a = __builtin_ia32_cvtps2pd(yyyy2);
            yyyy2 = __builtin_ia32_shufps(yyyy2, yyyy2, 14);
            v2df dd2b = __builtin_ia32_cvtps2pd(yyyy2);
            rr += dd2a;
            rr += dd2b;

            v4sf yyyy3 = __builtin_ia32_loadups(y + i + 12);
            v4sf xxxx3 = __builtin_ia32_loadups(x + i + 12);
            yyyy3 *= xxxx3;
            v4sf zzzz3 = __builtin_ia32_loadups(z + i + 12);
            yyyy3 *= zzzz3;
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
            v4sf zzzz0 = __builtin_ia32_loadups(z + i + 0);
            yyyy0 *= zzzz0;

            v2df dd1 = __builtin_ia32_cvtps2pd(yyyy0);
            yyyy0 = __builtin_ia32_shufps(yyyy0, yyyy0, 14);
            v2df dd2 = __builtin_ia32_cvtps2pd(yyyy0);
            rr += dd1;
            rr += dd2;
        }

        double results[2];
        *(v2df *)results = rr;

        res = results[0] + results[1];
    }
        
    for (;  i < n;  ++i) res += x[i] * y[i] * z[i];
    return res;
}

double vec_accum_prod3(const float * x, const float * y, const double * z,
                       size_t n)
{
    double res = 0.0;
    unsigned i = 0;

    if (true) {
        v2df rr = vec_splat(0.0);

        for (; i + 4 <= n;  i += 4) {
            v4sf yyyy0 = __builtin_ia32_loadups(y + i + 0);
            v4sf xxxx0 = __builtin_ia32_loadups(x + i + 0);
            yyyy0 *= xxxx0;

            v2df zz0a  = __builtin_ia32_loadupd(z + i + 0);
            v2df zz0b  = __builtin_ia32_loadupd(z + i + 2);

            v2df dd0a = __builtin_ia32_cvtps2pd(yyyy0);
            yyyy0 = __builtin_ia32_shufps(yyyy0, yyyy0, 14);
            v2df dd0b = __builtin_ia32_cvtps2pd(yyyy0);

            dd0a     *= zz0a;
            dd0b     *= zz0b;

            rr += dd0a;
            rr += dd0b;
        }

        double results[2];
        *(v2df *)results = rr;

        res = results[0] + results[1];
    }
        
    for (;  i < n;  ++i) res += x[i] * y[i] * z[i];
    return res;
}

void vec_minus(const double * x, const double * y, double * r, size_t n)
{
    for (unsigned i = 0;  i < n;  ++i) r[i] = x[i] - y[i];
}

double vec_accum_prod3(const double * x, const double * y, const double * z,
                      size_t n)
{
    unsigned i = 0;
    double result = 0.0;

    if (true) {
        v2df rr = vec_splat(0.0);

        for (; i + 8 <= n;  i += 8) {
            v2df yy0 = __builtin_ia32_loadupd(y + i + 0);
            v2df xx0 = __builtin_ia32_loadupd(x + i + 0);
            v2df zz0 = __builtin_ia32_loadupd(z + i + 0);
            yy0 *= xx0;
            yy0 *= zz0;
            rr += yy0;

            v2df yy1 = __builtin_ia32_loadupd(y + i + 2);
            v2df xx1 = __builtin_ia32_loadupd(x + i + 2);
            v2df zz1 = __builtin_ia32_loadupd(z + i + 2);
            yy1 *= xx1;
            yy1 *= zz1;
            rr += yy1;
            
            v2df yy2 = __builtin_ia32_loadupd(y + i + 4);
            v2df xx2 = __builtin_ia32_loadupd(x + i + 4);
            v2df zz2 = __builtin_ia32_loadupd(z + i + 4);
            yy2 *= xx2;
            yy2 *= zz2;
            rr += yy2;

            v2df yy3 = __builtin_ia32_loadupd(y + i + 6);
            v2df xx3 = __builtin_ia32_loadupd(x + i + 6);
            v2df zz3 = __builtin_ia32_loadupd(z + i + 6);
            yy3 *= xx3;
            yy3 *= zz3;
            rr += yy3;
        }

        for (; i + 2 <= n;  i += 2) {
            v2df yy0 = __builtin_ia32_loadupd(y + i + 0);
            v2df xx0 = __builtin_ia32_loadupd(x + i + 0);
            v2df zz0 = __builtin_ia32_loadupd(z + i + 0);
            yy0 *= xx0;
            yy0 *= zz0;
            rr += yy0;
        }

        double results[2];
        *(v2df *)results = rr;

        result = results[0] + results[1];
    }

    for (; i < n;  ++i) result += x[i] * y[i] * z[i];

    return result;
}

double vec_accum_prod3(const double * x, const double * y, const float * z,
                      size_t n)
{
    unsigned i = 0;
    double result = 0.0;

    if (true) {
        v2df rr = vec_splat(0.0);

        for (; i + 4 <= n;  i += 4) {
            v4sf zzzz01 = __builtin_ia32_loadups(z + i + 0);
            v2df yy0 = __builtin_ia32_loadupd(y + i + 0);
            v2df zz0 = __builtin_ia32_cvtps2pd(zzzz01);
            v2df xx0 = __builtin_ia32_loadupd(x + i + 0);

            yy0 *= xx0;
            yy0 *= zz0;
            rr += yy0;

            zzzz01 = __builtin_ia32_shufps(zzzz01, zzzz01, 14);
            v2df zz1 = __builtin_ia32_cvtps2pd(zzzz01);
            v2df yy1 = __builtin_ia32_loadupd(y + i + 2);
            v2df xx1 = __builtin_ia32_loadupd(x + i + 2);

            yy1 *= xx1;
            yy1 *= zz1;
            rr += yy1;
        }

        double results[2];
        *(v2df *)results = rr;

        result = results[0] + results[1];
    }

    for (; i < n;  ++i) result += x[i] * y[i] * z[i];

    return result;
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
    unsigned i = 0;

    if (true) {
        v2df rr = vec_splat(0.0);

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

        res = results[0] + results[1];
    }
        

    for (;  i < n;  ++i) res += x[i] * y[i];
    return res;
}

double vec_dotprod_dp(const double * x, const float * y, size_t n)
{
    double res = 0.0;

    unsigned i = 0;
    if (true) {
        v2df rr0 = vec_splat(0.0), rr1 = vec_splat(0.0);
        
        for (; i + 8 <= n;  i += 8) {
            v4sf yyyy01 = __builtin_ia32_loadups(y + i + 0);
            v2df yy0    = __builtin_ia32_cvtps2pd(yyyy01);
            yyyy01      = __builtin_ia32_shufps(yyyy01, yyyy01, 14);
            v2df yy1    = __builtin_ia32_cvtps2pd(yyyy01);

            v2df xx0    = __builtin_ia32_loadupd(x + i + 0);
            rr0        += xx0 * yy0;

            v2df xx1    = __builtin_ia32_loadupd(x + i + 2);
            rr1        += xx1 * yy1;

            v4sf yyyy23 = __builtin_ia32_loadups(y + i + 4);
            v2df yy2    = __builtin_ia32_cvtps2pd(yyyy23);
            yyyy23      = __builtin_ia32_shufps(yyyy23, yyyy23, 14);
            v2df yy3    = __builtin_ia32_cvtps2pd(yyyy23);

            v2df xx2    = __builtin_ia32_loadupd(x + i + 4);
            rr0        += xx2 * yy2;

            v2df xx3    = __builtin_ia32_loadupd(x + i + 6);
            rr1        += xx3 * yy3;
        }

        for (; i + 4 <= n;  i += 4) {
            v4sf yyyy01 = __builtin_ia32_loadups(y + i + 0);
            v2df yy0    = __builtin_ia32_cvtps2pd(yyyy01);
            yyyy01      = __builtin_ia32_shufps(yyyy01, yyyy01, 14);
            v2df yy1    = __builtin_ia32_cvtps2pd(yyyy01);

            v2df xx0    = __builtin_ia32_loadupd(x + i + 0);
            rr0        += xx0 * yy0;

            v2df xx1    = __builtin_ia32_loadupd(x + i + 2);
            rr1        += xx1 * yy1;
        }

        rr0 += rr1;

        double results[2];
        *(v2df *)results = rr0;
        
        res = results[0] + results[1];
    }


    for (;  i < n;  ++i) res += x[i] * y[i];

    return res;
}

double vec_sum_dp(const float * x, size_t n)
{
    double res = 0.0;
    for (unsigned i = 0;  i < n;  ++i)
        res += x[i];
    return res;
}

void vec_add(const double * x, const double * y, double * r, size_t n)
{
    unsigned i = 0;
    if (true) {
        for (; i + 8 <= n;  i += 8) {
            v2df yy0 = __builtin_ia32_loadupd(y + i + 0);
            v2df xx0 = __builtin_ia32_loadupd(x + i + 0);
            yy0 += xx0;
            __builtin_ia32_storeupd(r + i + 0, yy0);

            v2df yy1 = __builtin_ia32_loadupd(y + i + 2);
            v2df xx1 = __builtin_ia32_loadupd(x + i + 2);
            yy1 += xx1;
            __builtin_ia32_storeupd(r + i + 2, yy1);
            
            v2df yy2 = __builtin_ia32_loadupd(y + i + 4);
            v2df xx2 = __builtin_ia32_loadupd(x + i + 4);
            yy2 += xx2;
            __builtin_ia32_storeupd(r + i + 4, yy2);

            v2df yy3 = __builtin_ia32_loadupd(y + i + 6);
            v2df xx3 = __builtin_ia32_loadupd(x + i + 6);
            yy3 += xx3;
            __builtin_ia32_storeupd(r + i + 6, yy3);
        }

        for (; i + 2 <= n;  i += 2) {
            v2df yy0 = __builtin_ia32_loadupd(y + i + 0);
            v2df xx0 = __builtin_ia32_loadupd(x + i + 0);
            yy0 += xx0;
            __builtin_ia32_storeupd(r + i + 0, yy0);
        }
    }

    for (;  i < n;  ++i) r[i] = x[i] * y[i];
}

void vec_add(const double * x, double k, const float * y, double * r, size_t n)
{
    unsigned i = 0;

    v2df kk = vec_splat(k);

    if (true) {
        for (; i + 8 <= n;  i += 8) {
            v4sf yyyy01 = __builtin_ia32_loadups(y + i + 0);
            v2df yy0    = __builtin_ia32_cvtps2pd(yyyy01);
            yyyy01      = __builtin_ia32_shufps(yyyy01, yyyy01, 14);
            v2df yy1    = __builtin_ia32_cvtps2pd(yyyy01);
            yy0        *= kk;
            yy1        *= kk;

            v2df xx0    = __builtin_ia32_loadupd(x + i + 0);
            yy0        += xx0;
            __builtin_ia32_storeupd(r + i + 0, yy0);

            v2df xx1    = __builtin_ia32_loadupd(x + i + 2);
            yy1        += xx1;
            __builtin_ia32_storeupd(r + i + 2, yy1);

            v4sf yyyy23 = __builtin_ia32_loadups(y + i + 4);
            v2df yy2    = __builtin_ia32_cvtps2pd(yyyy23);
            yyyy23      = __builtin_ia32_shufps(yyyy23, yyyy23, 14);
            v2df yy3    = __builtin_ia32_cvtps2pd(yyyy23);
            yy2        *= kk;
            yy3        *= kk;

            v2df xx2    = __builtin_ia32_loadupd(x + i + 4);
            yy2        += xx2;
            __builtin_ia32_storeupd(r + i + 4, yy2);

            v2df xx3    = __builtin_ia32_loadupd(x + i + 6);
            yy3        += xx3;
            __builtin_ia32_storeupd(r + i + 6, yy3);
        }

        for (; i + 4 <= n;  i += 4) {
            v4sf yyyy01 = __builtin_ia32_loadups(y + i + 0);
            v2df yy0    = __builtin_ia32_cvtps2pd(yyyy01);
            yyyy01      = __builtin_ia32_shufps(yyyy01, yyyy01, 14);
            v2df yy1    = __builtin_ia32_cvtps2pd(yyyy01);
            yy0        *= kk;
            yy1        *= kk;

            v2df xx0    = __builtin_ia32_loadupd(x + i + 0);
            yy0        += xx0;
            __builtin_ia32_storeupd(r + i + 0, yy0);

            v2df xx1    = __builtin_ia32_loadupd(x + i + 2);
            yy1        += xx1;
            __builtin_ia32_storeupd(r + i + 2, yy1);
        }
    }

    for (;  i < n;  ++i) r[i] = x[i] + k * y[i];
}

void vec_add(const double * x, const float * y, double * r, size_t n)
{
    for (unsigned i = 0;  i < n;  ++i) r[i] = x[i] * y[i];
}

void vec_prod(const double * x, const double * y, double * r, size_t n)
{
    unsigned i = 0;
    if (true) {
        for (; i + 8 <= n;  i += 8) {
            v2df yy0 = __builtin_ia32_loadupd(y + i + 0);
            v2df xx0 = __builtin_ia32_loadupd(x + i + 0);
            yy0 *= xx0;
            __builtin_ia32_storeupd(r + i + 0, yy0);

            v2df yy1 = __builtin_ia32_loadupd(y + i + 2);
            v2df xx1 = __builtin_ia32_loadupd(x + i + 2);
            yy1 *= xx1;
            __builtin_ia32_storeupd(r + i + 2, yy1);
            
            v2df yy2 = __builtin_ia32_loadupd(y + i + 4);
            v2df xx2 = __builtin_ia32_loadupd(x + i + 4);
            yy2 *= xx2;
            __builtin_ia32_storeupd(r + i + 4, yy2);

            v2df yy3 = __builtin_ia32_loadupd(y + i + 6);
            v2df xx3 = __builtin_ia32_loadupd(x + i + 6);
            yy3 *= xx3;
            __builtin_ia32_storeupd(r + i + 6, yy3);
        }

        for (; i + 2 <= n;  i += 2) {
            v2df yy0 = __builtin_ia32_loadupd(y + i + 0);
            v2df xx0 = __builtin_ia32_loadupd(x + i + 0);
            yy0 *= xx0;
            __builtin_ia32_storeupd(r + i + 0, yy0);
        }
    }

    for (;  i < n;  ++i) r[i] = x[i] * y[i];
}

void vec_prod(const double * x, const float * y, double * r, size_t n)
{
    unsigned i = 0;

    if (true) {
        for (; i + 8 <= n;  i += 8) {
            v4sf yyyy01 = __builtin_ia32_loadups(y + i + 0);
            v2df yy0    = __builtin_ia32_cvtps2pd(yyyy01);
            yyyy01      = __builtin_ia32_shufps(yyyy01, yyyy01, 14);
            v2df yy1    = __builtin_ia32_cvtps2pd(yyyy01);

            v2df xx0    = __builtin_ia32_loadupd(x + i + 0);
            yy0        *= xx0;
            __builtin_ia32_storeupd(r + i + 0, yy0);

            v2df xx1    = __builtin_ia32_loadupd(x + i + 2);
            yy1        *= xx1;
            __builtin_ia32_storeupd(r + i + 2, yy1);

            v4sf yyyy23 = __builtin_ia32_loadups(y + i + 4);
            v2df yy2    = __builtin_ia32_cvtps2pd(yyyy23);
            yyyy23      = __builtin_ia32_shufps(yyyy23, yyyy23, 14);
            v2df yy3    = __builtin_ia32_cvtps2pd(yyyy23);

            v2df xx2    = __builtin_ia32_loadupd(x + i + 4);
            yy2        *= xx2;
            __builtin_ia32_storeupd(r + i + 4, yy2);

            v2df xx3    = __builtin_ia32_loadupd(x + i + 6);
            yy3        *= xx3;
            __builtin_ia32_storeupd(r + i + 6, yy3);
        }

        for (; i + 4 <= n;  i += 4) {
            v4sf yyyy01 = __builtin_ia32_loadups(y + i + 0);
            v2df yy0    = __builtin_ia32_cvtps2pd(yyyy01);
            yyyy01      = __builtin_ia32_shufps(yyyy01, yyyy01, 14);
            v2df yy1    = __builtin_ia32_cvtps2pd(yyyy01);

            v2df xx0    = __builtin_ia32_loadupd(x + i + 0);
            yy0        *= xx0;
            __builtin_ia32_storeupd(r + i + 0, yy0);

            v2df xx1    = __builtin_ia32_loadupd(x + i + 2);
            yy1        *= xx1;
            __builtin_ia32_storeupd(r + i + 2, yy1);
        }
    }

    for (;  i < n;  ++i) r[i] = x[i] * y[i];
}

void vec_k1_x_plus_k2_y_z(double k1, const double * x,
                          double k2, const double * y, const double * z,
                          double * r, size_t n)
{
    unsigned i = 0;

    v2df kk1 = vec_splat(k1);
    v2df kk2 = vec_splat(k2);

    if (true) {
        for (; i + 8 <= n;  i += 8) {
            v2df yy0 = __builtin_ia32_loadupd(y + i + 0);
            v2df xx0 = __builtin_ia32_loadupd(x + i + 0);
            v2df zz0 = __builtin_ia32_loadupd(z + i + 0);
            yy0 *= kk2;
            xx0 *= kk1;
            yy0 *= zz0;
            yy0 += xx0;
            __builtin_ia32_storeupd(r + i + 0, yy0);

            v2df yy1 = __builtin_ia32_loadupd(y + i + 2);
            v2df xx1 = __builtin_ia32_loadupd(x + i + 2);
            v2df zz1 = __builtin_ia32_loadupd(z + i + 2);
            yy1 *= kk2;
            xx1 *= kk1;
            yy1 *= zz1;
            yy1 += xx1;
            __builtin_ia32_storeupd(r + i + 2, yy1);
            
            v2df yy2 = __builtin_ia32_loadupd(y + i + 4);
            v2df xx2 = __builtin_ia32_loadupd(x + i + 4);
            v2df zz2 = __builtin_ia32_loadupd(z + i + 4);
            yy2 *= kk2;
            xx2 *= kk1;
            yy2 *= zz2;
            yy2 += xx2;
            __builtin_ia32_storeupd(r + i + 4, yy2);

            v2df yy3 = __builtin_ia32_loadupd(y + i + 6);
            v2df xx3 = __builtin_ia32_loadupd(x + i + 6);
            v2df zz3 = __builtin_ia32_loadupd(z + i + 6);
            yy3 *= kk2;
            xx3 *= kk1;
            yy3 *= zz3;
            yy3 += xx3;
            __builtin_ia32_storeupd(r + i + 6, yy3);
        }

        for (; i + 2 <= n;  i += 2) {
            v2df yy0 = __builtin_ia32_loadupd(y + i + 0);
            v2df xx0 = __builtin_ia32_loadupd(x + i + 0);
            v2df zz0 = __builtin_ia32_loadupd(z + i + 0);
            yy0 *= kk2;
            xx0 *= kk1;
            yy0 *= zz0;
            yy0 += xx0;
            __builtin_ia32_storeupd(r + i + 0, yy0);
        }
    }

    for (;  i < n;  ++i) r[i] = k1 * x[i] + k2 * y[i] * z[i];
}

void vec_k1_x_plus_k2_y_z(float k1, const float * x,
                          float k2, const float * y, const float * z,
                          float * r, size_t n)
{
    unsigned i = 0;

    v4sf kkkk1 = vec_splat(k1);
    v4sf kkkk2 = vec_splat(k2);

    if (true) {
        for (; i + 16 <= n;  i += 16) {
            v4sf yyyy0 = __builtin_ia32_loadups(y + i + 0);
            v4sf xxxx0 = __builtin_ia32_loadups(x + i + 0);
            v4sf zzzz0 = __builtin_ia32_loadups(z + i + 0);
            yyyy0 *= kkkk2;
            xxxx0 *= kkkk1;
            yyyy0 *= zzzz0;
            yyyy0 += xxxx0;
            __builtin_ia32_storeups(r + i + 0, yyyy0);

            v4sf yyyy1 = __builtin_ia32_loadups(y + i + 4);
            v4sf xxxx1 = __builtin_ia32_loadups(x + i + 4);
            v4sf zzzz1 = __builtin_ia32_loadups(z + i + 4);
            yyyy1 *= kkkk2;
            xxxx1 *= kkkk1;
            yyyy1 *= zzzz1;
            yyyy1 += xxxx1;
            __builtin_ia32_storeups(r + i + 4, yyyy1);
            
            v4sf yyyy2 = __builtin_ia32_loadups(y + i + 8);
            v4sf xxxx2 = __builtin_ia32_loadups(x + i + 8);
            v4sf zzzz2 = __builtin_ia32_loadups(z + i + 8);
            yyyy2 *= kkkk2;
            xxxx2 *= kkkk1;
            yyyy2 *= zzzz2;
            yyyy2 += xxxx2;
            __builtin_ia32_storeups(r + i + 8, yyyy2);

            v4sf yyyy3 = __builtin_ia32_loadups(y + i + 12);
            v4sf xxxx3 = __builtin_ia32_loadups(x + i + 12);
            v4sf zzzz3 = __builtin_ia32_loadups(z + i + 12);
            yyyy3 *= kkkk2;
            xxxx3 *= kkkk1;
            yyyy3 *= zzzz3;
            yyyy3 += xxxx3;
            __builtin_ia32_storeups(r + i + 12, yyyy3);
        }

        for (; i + 4 <= n;  i += 4) {
            v4sf yyyy0 = __builtin_ia32_loadups(y + i + 0);
            v4sf xxxx0 = __builtin_ia32_loadups(x + i + 0);
            v4sf zzzz0 = __builtin_ia32_loadups(z + i + 0);
            yyyy0 *= kkkk2;
            xxxx0 *= kkkk1;
            yyyy0 *= zzzz0;
            yyyy0 += xxxx0;
            __builtin_ia32_storeups(r + i + 0, yyyy0);
        }
    }

    for (;  i < n;  ++i) r[i] = k1 * x[i] + k2 * y[i] * z[i];
}

void vec_add_sqr(const float * x, float k, const float * y, float * r, size_t n)
{
    unsigned i = 0;

    if (true) {
        v4sf kkkk = vec_splat(k);
        //cerr << "unoptimized" << endl;

        for (; i + 16 <= n;  i += 16) {
            v4sf yyyy0 = __builtin_ia32_loadups(y + i + 0);
            v4sf xxxx0 = __builtin_ia32_loadups(x + i + 0);
            yyyy0 *= yyyy0;
            yyyy0 *= kkkk;
            v4sf yyyy1 = __builtin_ia32_loadups(y + i + 4);
            yyyy1 *= yyyy1;
            yyyy0 += xxxx0;
            v4sf xxxx1 = __builtin_ia32_loadups(x + i + 4);
            __builtin_ia32_storeups(r + i + 0, yyyy0);
            yyyy1 *= kkkk;
            v4sf yyyy2 = __builtin_ia32_loadups(y + i + 8);
            yyyy1 += xxxx1;
            yyyy2 *= yyyy2;
            v4sf xxxx2 = __builtin_ia32_loadups(x + i + 8);
            __builtin_ia32_storeups(r + i + 4, yyyy1);
            yyyy2 *= kkkk;
            v4sf yyyy3 = __builtin_ia32_loadups(y + i + 12);
            yyyy2 += xxxx2;
            yyyy3 *= yyyy3;
            v4sf xxxx3 = __builtin_ia32_loadups(x + i + 12);
            __builtin_ia32_storeups(r + i + 8, yyyy2);
            yyyy3 *= kkkk;
            yyyy3 += xxxx3;
            __builtin_ia32_storeups(r + i + 12, yyyy3);
        }

        for (; i + 4 <= n;  i += 4) {
            v4sf yyyy0 = __builtin_ia32_loadups(y + i + 0);
            v4sf xxxx0 = __builtin_ia32_loadups(x + i + 0);
            yyyy0 *= yyyy0;
            yyyy0 *= kkkk;
            yyyy0 += xxxx0;
            __builtin_ia32_storeups(r + i + 0, yyyy0);
        }

        for (; i < n;  ++i) r[i] = x[i] + k * (y[i] * y[i]);
    }
}

void vec_add_sqr(const double * x, double k, const double * y, double * r,
                 size_t n)
{
    v2df kk = vec_splat(k);
    unsigned i = 0;

    if (true) {
        for (; i + 8 <= n;  i += 8) {
            v2df yy0 = __builtin_ia32_loadupd(y + i + 0);
            v2df xx0 = __builtin_ia32_loadupd(x + i + 0);
            yy0 *= yy0;
            yy0 *= kk;
            yy0 += xx0;
            __builtin_ia32_storeupd(r + i + 0, yy0);

            v2df yy1 = __builtin_ia32_loadupd(y + i + 2);
            v2df xx1 = __builtin_ia32_loadupd(x + i + 2);
            yy1 *= yy1;
            yy1 *= kk;
            yy1 += xx1;
            __builtin_ia32_storeupd(r + i + 2, yy1);
            
            v2df yy2 = __builtin_ia32_loadupd(y + i + 4);
            v2df xx2 = __builtin_ia32_loadupd(x + i + 4);
            yy2 *= yy2;
            yy2 *= kk;
            yy2 += xx2;
            __builtin_ia32_storeupd(r + i + 4, yy2);

            v2df yy3 = __builtin_ia32_loadupd(y + i + 6);
            v2df xx3 = __builtin_ia32_loadupd(x + i + 6);
            yy3 *= yy3;
            yy3 *= kk;
            yy3 += xx3;
            __builtin_ia32_storeupd(r + i + 6, yy3);

        }

        for (; i + 2 <= n;  i += 2) {
            v2df yy0 = __builtin_ia32_loadupd(y + i + 0);
            v2df xx0 = __builtin_ia32_loadupd(x + i + 0);
            yy0 *= yy0;
            yy0 *= kk;
            yy0 += xx0;
            __builtin_ia32_storeupd(r + i + 0, yy0);
        }
    }

    for (;  i < n;  ++i) r[i] = x[i] + k * (y[i] * y[i]);
}

void vec_add_sqr(const float * x, float k, const double * y, float * r, size_t n)
{
    for (unsigned i = 0; i < n;  ++i) r[i] = x[i] + k * (y[i] * y[i]);
}

void vec_add_sqr(const double * x, double k, const float * y, double * r,
                 size_t n)
{
    unsigned i = 0;

    if (true) {
        v2df kk = vec_splat(k);
        for (; i + 8 <= n;  i += 8) {
            v4sf yyyy01 = __builtin_ia32_loadups(y + i + 0);
            yyyy01     *= yyyy01;
            v2df yy0    = __builtin_ia32_cvtps2pd(yyyy01);
            yyyy01      = __builtin_ia32_shufps(yyyy01, yyyy01, 14);
            v2df yy1    = __builtin_ia32_cvtps2pd(yyyy01);
            yy0        *= kk;
            yy1        *= kk;

            v2df xx0    = __builtin_ia32_loadupd(x + i + 0);
            yy0        += xx0;
            __builtin_ia32_storeupd(r + i + 0, yy0);

            v2df xx1    = __builtin_ia32_loadupd(x + i + 2);
            yy1        += xx1;
            __builtin_ia32_storeupd(r + i + 2, yy1);

            v4sf yyyy23 = __builtin_ia32_loadups(y + i + 4);
            yyyy23     *= yyyy23;
            v2df yy2    = __builtin_ia32_cvtps2pd(yyyy23);
            yyyy23      = __builtin_ia32_shufps(yyyy23, yyyy23, 14);
            v2df yy3    = __builtin_ia32_cvtps2pd(yyyy23);
            yy2        *= kk;
            yy3        *= kk;

            v2df xx2    = __builtin_ia32_loadupd(x + i + 4);
            yy2        += xx2;
            __builtin_ia32_storeupd(r + i + 4, yy2);

            v2df xx3    = __builtin_ia32_loadupd(x + i + 6);
            yy3        += xx3;
            __builtin_ia32_storeupd(r + i + 6, yy3);
        }

        for (; i + 4 <= n;  i += 4) {
            v4sf yyyy01 = __builtin_ia32_loadups(y + i + 0);
            yyyy01     *= yyyy01;
            v2df yy0    = __builtin_ia32_cvtps2pd(yyyy01);
            yyyy01      = __builtin_ia32_shufps(yyyy01, yyyy01, 14);
            v2df yy1    = __builtin_ia32_cvtps2pd(yyyy01);
            yy0        *= kk;
            yy1        *= kk;

            v2df xx0    = __builtin_ia32_loadupd(x + i + 0);
            yy0        += xx0;
            __builtin_ia32_storeupd(r + i + 0, yy0);

            v2df xx1    = __builtin_ia32_loadupd(x + i + 2);
            yy1        += xx1;
            __builtin_ia32_storeupd(r + i + 2, yy1);
        }
    }

    for (;  i < n;  ++i) r[i] = x[i] + k * (y[i] * y[i]);
}

void vec_add(const float * x, const double * k, const double * y, float * r,
             size_t n)
{
    unsigned i = 0;

    if (false) ;
    else {
        for (; i + 4 <= n;  i += 4) {
            v2df yy0a  = __builtin_ia32_loadupd(y + i + 0);
            v2df yy0b  = __builtin_ia32_loadupd(y + i + 2);
            v2df kk0a  = __builtin_ia32_loadupd(k + i + 0);
            v2df kk0b  = __builtin_ia32_loadupd(k + i + 2);
            v4sf xxxx0 = __builtin_ia32_loadups(x + i + 0);

            yy0a *= kk0a;
            yy0b *= kk0b;

            v2df xx0a, xx0b;
            vec_f2d(xxxx0, xx0a, xx0b);

            xx0a += yy0a;
            xx0b += yy0b;

            v4sf rrrr0 = vec_d2f(xx0a, xx0b);

            __builtin_ia32_storeups(r + i + 0, rrrr0);
        }
    }

    for (; i < n;  ++i) r[i] = x[i] + k[i] * y[i];
}

void vec_add(const float * x, const float * k, const double * y, float * r,
             size_t n)
{
    unsigned i = 0;

    if (false) ;
    else {
        for (; i + 4 <= n;  i += 4) {
            v2df yy0a  = __builtin_ia32_loadupd(y + i + 0);
            v2df yy0b  = __builtin_ia32_loadupd(y + i + 2);
            v4sf kkkk0 = __builtin_ia32_loadups(k + i + 0);
            v4sf xxxx0 = __builtin_ia32_loadups(x + i + 0);
            
            v2df kk0a, kk0b;
            vec_f2d(kkkk0, kk0a, kk0b);

            yy0a *= kk0a;
            yy0b *= kk0b;

            v2df xx0a, xx0b;
            vec_f2d(xxxx0, xx0a, xx0b);

            xx0a += yy0a;
            xx0b += yy0b;

            v4sf rrrr0 = vec_d2f(xx0a, xx0b);

            __builtin_ia32_storeups(r + i + 0, rrrr0);
        }
    }

    for (; i < n;  ++i) r[i] = x[i] + k[i] * y[i];
}

void vec_add(const double * x, const float * k, const float * y, double * r,
             size_t n)
{
    unsigned i = 0;

    if (false) ;
    else {
        for (; i + 4 <= n;  i += 4) {
            v4sf kkkk0 = __builtin_ia32_loadups(k + i + 0);
            v4sf yyyy0 = __builtin_ia32_loadups(y + i + 0);
            v2df xx0a  = __builtin_ia32_loadupd(x + i + 0);
            v2df xx0b  = __builtin_ia32_loadupd(x + i + 2);

            v4sf ykyk0 = yyyy0 * kkkk0;
            
            v2df yk0a, yk0b;
            vec_f2d(ykyk0, yk0a, yk0b);

            xx0a += yk0a;
            xx0b += yk0b;

            __builtin_ia32_storeupd(r + i + 0, xx0a);
            __builtin_ia32_storeupd(r + i + 2, xx0b);
        }
    }

    for (; i < n;  ++i) r[i] = x[i] + k[i] * y[i];
}

void vec_add(const double * x, const float * k, const double * y, double * r,
             size_t n)
{
    unsigned i = 0;

    if (false) ;
    else {
        for (; i + 4 <= n;  i += 4) {
            v2df yy0a  = __builtin_ia32_loadupd(y + i + 0);
            v2df yy0b  = __builtin_ia32_loadupd(y + i + 2);
            v4sf kkkk0 = __builtin_ia32_loadups(k + i + 0);
            v2df xx0a  = __builtin_ia32_loadupd(x + i + 0);
            v2df xx0b  = __builtin_ia32_loadupd(x + i + 2);
            
            v2df kk0a, kk0b;
            vec_f2d(kkkk0, kk0a, kk0b);

            yy0a *= kk0a;
            yy0b *= kk0b;

            xx0a += yy0a;
            xx0b += yy0b;

            __builtin_ia32_storeupd(r + i + 0, xx0a);
            __builtin_ia32_storeupd(r + i + 2, xx0b);
        }
    }

    for (; i < n;  ++i) r[i] = x[i] + k[i] * y[i];
}

// See https://bugzilla.redhat.com/show_bug.cgi?id=521190
// for why we use double version of exp

void vec_exp(const float * x, float * r, size_t n)
{
    unsigned i = 0;
    for (; i < n;  ++i) r[i] = exp((double)x[i]);
}

void vec_exp(const float * x, float k, float * r, size_t n)
{
    unsigned i = 0;
    for (; i < n;  ++i) r[i] = exp((double)(k * x[i]));
}

void vec_exp(const float * x, double * r, size_t n)
{
    unsigned i = 0;
    for (; i < n;  ++i) r[i] = exp((double)x[i]);
}

void vec_exp(const float * x, double k, double * r, size_t n)
{
    unsigned i = 0;

    if (true) {
        v2df kk = vec_splat(k);

        for (; i + 4 <= n;  i += 4) {
            v4sf xxxx0 = __builtin_ia32_loadups(x + i + 0);
            v2df xx0a, xx0b;
            vec_f2d(xxxx0, xx0a, xx0b);

            v2df rr0a = sse2_exp(kk * xx0a);
            v2df rr0b = sse2_exp(kk * xx0b);

            __builtin_ia32_storeupd(r + i + 0, rr0a);
            __builtin_ia32_storeupd(r + i + 2, rr0b);
        }
    }

    for (; i < n;  ++i) r[i] = exp((double)(k * x[i]));
}

void vec_exp(const double * x, double * r, size_t n)
{
    unsigned i = 0;
    for (; i < n;  ++i) r[i] = exp((double)x[i]);
}

void vec_exp(const double * x, double k, double * r, size_t n)
{
    unsigned i = 0;
    for (; i < n;  ++i) r[i] = exp((double)(k * x[i]));
}

float vec_twonorm_sqr(const float * x, size_t n)
{
    unsigned i = 0;
    float result = 0.0;
    for (; i < n;  ++i) result += x[i] * x[i];
    return result;
}

double vec_twonorm_sqr_dp(const float * x, size_t n)
{
    unsigned i = 0;
    double result = 0.0;
    for (; i < n;  ++i) {
        double xd = x[i];
        result += xd * xd;
    }
    return result;
}

double vec_twonorm_sqr(const double * x, size_t n)
{
    unsigned i = 0;
    float result = 0.0;
    for (; i < n;  ++i) result += x[i] * x[i];
    return result;
}

double vec_kl(const float * p, const float * q, size_t n)
{
    unsigned i = 0;

    double total = 0.0;

    if (true) {
        v2df ttotal = vec_splat(0.0);

        for (; i + 4 <= n;  i += 4) {
            v4sf pppp0 = __builtin_ia32_loadups(p + i + 0);
            v4sf qqqq0 = __builtin_ia32_loadups(q + i + 0);
                 qqqq0 = pppp0 / qqqq0;
                 qqqq0 = sse2_logf(qqqq0);
            v4sf klkl0 = qqqq0 * pppp0;

            v2df kl0a, kl0b;
            vec_f2d(klkl0, kl0a, kl0b);

            ttotal += kl0a;
            ttotal += kl0b;
        }

        double results[2];
        *(v2df *)results = ttotal;
        
        total = results[0] + results[1];
    }

    for (; i < n;  ++i) total += p[i] * logf(p[i] / q[i]);

    return total;
}

void vec_min_max_el(const float * x, float * mins, float * maxs, size_t n)
{
    unsigned i = 0;

    if (false) ;
    else {
        for (; i + 4 <= n;  i += 4) {
            v4sf xxxx0 = __builtin_ia32_loadups(x + i + 0);
            v4sf iiii0 = __builtin_ia32_loadups(mins + i + 0);
            v4sf aaaa0 = __builtin_ia32_loadups(maxs + i + 0);
            iiii0      = __builtin_ia32_minps(iiii0, xxxx0);
            aaaa0      = __builtin_ia32_maxps(aaaa0, xxxx0);
            __builtin_ia32_storeups(mins + i + 0, iiii0);
            __builtin_ia32_storeups(maxs + i + 0, aaaa0);
        }
    }

    for (; i < n;  ++i) {
        mins[i] = std::min(mins[i], x[i]);
        maxs[i] = std::max(maxs[i], x[i]);
    }
}

} // namespace Generic

} // namespace SIMD
} // namespace ML

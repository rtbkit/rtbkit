/* floating_point.h                                                -*- C++ -*-
   Jeremy Barnes, 27 January 2005
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

   Utilities to deal with floating point numbers.
*/

#ifndef __utils__floating_point_h__
#define __utils__floating_point_h__


#include "jml/compiler/compiler.h"
#include <limits>
#include <stdint.h>

namespace ML {

namespace {

struct int_float {
    JML_ALWAYS_INLINE int_float(uint32_t x) { i = x; }
    JML_ALWAYS_INLINE int_float(float x) { f = x; }
    union {
        float f;
        uint32_t i;
    };
};

struct int_double {
    JML_ALWAYS_INLINE int_double(uint64_t x) { i = x; }
    JML_ALWAYS_INLINE int_double(double x) { f = x; }
    union {
        double f;
        uint64_t i;
    };
};

} // file scope

/** Functions to get the bit patterns of floating point numbers in and out of
    integers for twiddling.
*/
JML_ALWAYS_INLINE float reinterpret_as_float(uint32_t val)
{
    return int_float(val).f;
}

JML_ALWAYS_INLINE double reinterpret_as_double(uint64_t val)
{
    return int_double(val).f;
}

JML_ALWAYS_INLINE uint32_t reinterpret_as_int(float val)
{
    return int_float(val).i;
}

JML_ALWAYS_INLINE uint64_t reinterpret_as_int(double val)
{
    return int_double(val).i;
}

/** Like std::less<Float>, but has a well defined order for nan values, which
    allows us to sort ranges that might contain nan values without crashing.
*/
template<typename Float>
struct safe_less {
    bool operator () (Float v1, Float v2) const
    {
        bool nan1 = isnanf(v1), nan2 = isnanf(v2);
        if (nan1 && nan2) return false;
        return (nan1 > nan2)
            || ((nan1 == nan2) && v1 < v2);
    }
};

struct float_hasher {
    JML_ALWAYS_INLINE int operator () (float val) const
    {
        return reinterpret_as_int(val);
    }
};

template<class C>
struct fp_traits
    : public std::numeric_limits<C> {
};

template<>
struct fp_traits<float>
    : public std::numeric_limits<float> {
    /** Maximum argument to exp() that doesn't result in an infinity.  Found
        using exp_test program in boosting/testing. */
    static const float max_exp_arg;
};

template<>
struct fp_traits<double>
    : public std::numeric_limits<double> {
    static const double max_exp_arg;
};

template<>
struct fp_traits<long double>
    : public std::numeric_limits<long double> {
    static const long double max_exp_arg;
};

} // namespace ML

#endif /* __utils__floating_point_h__ */

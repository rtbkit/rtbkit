/* distribution_ops.h                                              -*- C++ -*-
   Jeremy Barnes, 2 Febryary 2005
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

   Operations on distributions.
*/

#ifndef __utils__distribution_ops_h__
#define __utils__distribution_ops_h__

#include "distribution.h"
#include "jml/math/bound.h"
#include "jml/math/round.h"
#include "jml/math/xdiv.h"
#include "jml/arch/math_builtins.h"
#include <cmath>

namespace ML {

template<typename F, class Underlying>
distribution<F, Underlying> bound(const distribution<F, Underlying> & dist, F min, F max)
{
    distribution<F, Underlying> result(dist.size());
    for (unsigned i = 0;  i < dist.size();  ++i)
        result[i] = ML::bound(dist[i], min, max);
    return result;
}

template<typename F, class Underlying>
distribution<F, Underlying> log(const distribution<F, Underlying> & dist)
{
    distribution<F, Underlying> result(dist.size());
    for (unsigned i = 0;  i < dist.size();  ++i)
        result[i] = std::log(dist[i]);
    return result;
}

using ::log;

template<typename F, class Underlying>
distribution<F, Underlying> abs(const distribution<F, Underlying> & dist)
{
    distribution<F, Underlying> result(dist.size());
    for (unsigned i = 0;  i < dist.size();  ++i)
        result[i] = std::abs(dist[i]);
    return result;
}

using ::abs;

template<typename F, class Underlying>
distribution<F, Underlying> sqr(const distribution<F, Underlying> & dist)
{
    distribution<F, Underlying> result(dist.size());
    for (unsigned i = 0;  i < dist.size();  ++i)
        result[i] = dist[i] * dist[i];
    return result;
}

template<typename F, class Underlying>
distribution<F, Underlying> sqrt(const distribution<F, Underlying> & dist)
{
    distribution<F, Underlying> result(dist.size());
    for (unsigned i = 0;  i < dist.size();  ++i)
        result[i] = std::sqrt(dist[i]);
    return result;
}

using ::sqrt;

template<typename F, class Underlying>
distribution<F, Underlying> tanh(const distribution<F, Underlying> & dist)
{
    distribution<F, Underlying> result(dist.size());
    for (unsigned i = 0;  i < dist.size();  ++i)
        result[i] = std::tanh(dist[i]);
    return result;
}

using ::tanh;

template<typename F, class Underlying>
distribution<F, Underlying> round(const distribution<F, Underlying> & dist)
{
    distribution<F, Underlying> result(dist.size());
    for (unsigned i = 0;  i < dist.size();  ++i)
        result[i] = ML::round(dist[i]);
    return result;
}

using ::round;

template<typename F, class Underlying>
distribution<F, Underlying> exp(const distribution<F, Underlying> & dist, F exponent);

template<typename F, class Underlying>
distribution<F, Underlying> exp(const distribution<F, Underlying> & dist)
{
    distribution<F, Underlying> result(dist.size());
    for (unsigned i = 0;  i < dist.size();  ++i)
        result[i] = ML::exp(dist[i]);
    return result;
}

template<typename F, class Underlying>
distribution<F, Underlying> max(const distribution<F, Underlying> & dist, F val)
{
    distribution<F, Underlying> result(dist.size());
    for (unsigned i = 0;  i < dist.size();  ++i)
        result[i] = std::max(dist[i], val);
    return result;
}

template<typename F, class Underlying>
distribution<F, Underlying> max(const distribution<F, Underlying> & dist1,
                                const distribution<F, Underlying> & dist2)
{
    if (dist1.size() != dist2.size())
        wrong_sizes_exception("max", dist1.size(), dist2.size());

    distribution<F, Underlying> result(dist1.size());
    for (unsigned i = 0;  i < dist1.size();  ++i)
        result[i] = std::max(dist1[i], dist2[i]);
    return result;
}

template<typename F1, typename F2, class Underlying1, class Underlying2>
distribution<decltype(F1() + F2())>
max(const distribution<F1, Underlying1> & dist1,
    const distribution<F2, Underlying2> & dist2)
{
    if (dist1.size() != dist2.size())
        wrong_sizes_exception("max", dist1.size(), dist2.size());

    distribution<decltype(F1() + F2())> result(dist1.size());
    for (unsigned i = 0;  i < dist1.size();  ++i)
        result[i] = std::max<decltype(F1() + F2())>(dist1[i], dist2[i]);
    return result;
}

using std::max;

template<typename F, class Underlying>
distribution<F, Underlying>
min(const distribution<F, Underlying> & dist, F val)
{
    distribution<F, Underlying> result(dist.size());
    for (unsigned i = 0;  i < dist.size();  ++i)
        result[i] = std::min(dist[i], val);
    return result;
}

template<typename F, class Underlying>
distribution<F, Underlying> min(const distribution<F, Underlying> & dist1,
                                const distribution<F, Underlying> & dist2)
{
    if (dist1.size() != dist2.size())
        wrong_sizes_exception("min", dist1.size(), dist2.size());

    distribution<F, Underlying> result(dist1.size());
    for (unsigned i = 0;  i < dist1.size();  ++i)
        result[i] = std::min(dist1[i], dist2[i]);
    return result;
}

template<typename F1, typename F2, class Underlying1, class Underlying2>
distribution<decltype(F1() + F2())>
min(const distribution<F1, Underlying1> & dist1,
    const distribution<F2, Underlying2> & dist2)
{
    if (dist1.size() != dist2.size())
        wrong_sizes_exception("min", dist1.size(), dist2.size());

    distribution<decltype(F1() + F2())> result(dist1.size());
    for (unsigned i = 0;  i < dist1.size();  ++i)
        result[i] = std::min<decltype(F1() + F2())>(dist1[i], dist2[i]);
    return result;
}

using std::min;

template<typename F, class Underlying>
distribution<F, Underlying> xdiv(const distribution<F, Underlying> & dist,
                                 F val)
{
    distribution<F, Underlying> result(dist.size());
    for (unsigned i = 0;  i < dist.size();  ++i)
        result[i] = ML::xdiv(dist[i], val);
    return result;
}

template<typename F, class Underlying>
distribution<F, Underlying> xdiv(F val,
                                 const distribution<F, Underlying> & dist)
{
    distribution<F, Underlying> result(dist.size());
    for (unsigned i = 0;  i < dist.size();  ++i)
        result[i] = ML::xdiv(val, dist[i]);
    return result;
}

template<typename F, class Underlying>
distribution<F, Underlying> xdiv(const distribution<F, Underlying> & dist1,
                                 const distribution<F, Underlying> & dist2)
{
    if (dist1.size() != dist2.size())
        wrong_sizes_exception("xdiv", dist1.size(), dist2.size());

    distribution<F, Underlying> result(dist1.size());
    for (unsigned i = 0;  i < dist1.size();  ++i)
        result[i] = ML::xdiv(dist1[i], dist2[i]);
    return result;
}

template<typename F, class Underlying>
distribution<bool> isnan(const distribution<F, Underlying> & dist)
{
    distribution<bool> result(dist.size());
    for (unsigned i = 0;  i < dist.size();  ++i)
        result[i] = std::isnan(dist[i]);
    return result;
}

using ::isnan;

} // namespace ML

#endif /* __utils__distribution_ops_h__ */

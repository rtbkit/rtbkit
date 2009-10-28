/* distribution.h                                                  -*- C++ -*-
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

   An algebraic class, which can be used to hold a distribution.
*/

#ifndef __stats__distribution_h__
#define __stats__distribution_h__

#include <vector>
#include "utils/float_traits.h"
#include "arch/integer.h"
#include "arch/exception.h"
#include <numeric>
#include <limits>
#include <algorithm>
#include <ostream>
#include "utils/string_functions.h"

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/type_traits/is_same.hpp>

namespace ML {
namespace Stats {


inline void wrong_sizes_exception(const char * op, int size1, int size2)
{
    throw Exception(format("distribution: operation %s between different sized "
                           "distributions %d and %d", op, size1, size2));
}


template<typename F, class Underlying= std::vector<F> >
class distribution : public Underlying {
    typedef Underlying parent;
public:
    distribution() {}

    explicit distribution(size_t size, F val = F())
        : parent(size, val)
    {
    }

    template<class Iterator>
    distribution(Iterator first, Iterator last)
        : parent(first, last)
    {
    }

    template<typename OtherF, class OtherUnderlying>
    explicit distribution(const distribution<OtherF, OtherUnderlying> & dist)
    {
        reserve(dist.size());
        for (unsigned i = 0;  i < dist.size();  ++i)
            push_back((F)dist[i]);
    }

    template<typename OtherF, class OtherUnderlying>
    distribution &
    operator = (const distribution<OtherF, OtherUnderlying> & dist)
    {
        resize(dist.size());
        std::copy(dist.begin(), dist.end(), this->begin());
        return *this;
    }

#if 0 // use fill instead
    distribution &
    operator = (const F & val)
    {
        this->fill(val);
        return *this;
    }
#endif // use fill instead

    void fill(const F & val)
    {
        std::fill(this->begin(), this->end(), val);
    }

    distribution
    operator - () const
    {
        distribution result(this->size());
        for (unsigned i = 0;  i < this->size();  ++i)
            result[i] = - this->operator [] (i);
        return result;
    }

    #define DIST_SCALAR_OP(op) \
    distribution \
    operator op (F val) const \
    { \
        distribution result(this->size()); \
        for (unsigned i = 0;  i < this->size();  ++i) \
            result[i] = this->operator [] (i) op val; \
        return result; \
    }

    DIST_SCALAR_OP(+);
    DIST_SCALAR_OP(-);
    DIST_SCALAR_OP(*);
    DIST_SCALAR_OP(/);
    DIST_SCALAR_OP(&);
    DIST_SCALAR_OP(|);
    DIST_SCALAR_OP(&&);
    DIST_SCALAR_OP(||);
    #undef DIST_SCALAR_OP


    #define UPDATE_DIST_OP(op) \
        template<class F2, class Underlying2>    \
    distribution & \
    operator op (const distribution<F2, Underlying2> & d)       \
    { \
        if (d.size() != this->size()) \
            wrong_sizes_exception(#op, this->size(), d.size()); \
        for (unsigned i = 0;  i < this->size();  ++i) \
            this->operator [] (i) op d[i]; \
        return *this; \
    }

    UPDATE_DIST_OP(+=)
    UPDATE_DIST_OP(-=)
    UPDATE_DIST_OP(*=)
    UPDATE_DIST_OP(/=)
    #undef UPDATE_DIST_OP
    
    #define UPDATE_SCALAR_OP(op) \
    template<class F2>      \
    distribution & \
    operator op (F2 val) \
    { \
        for (unsigned i = 0;  i < this->size();  ++i) \
            this->operator [] (i) op val; \
        return *this; \
    }

    UPDATE_SCALAR_OP(+=)
    UPDATE_SCALAR_OP(-=)
    UPDATE_SCALAR_OP(*=)
    UPDATE_SCALAR_OP(/=)
    #undef UPDATE_SCALAR_OP

    double dotprod(const distribution & other) const
    {
        if (this->size() != other.size())
            wrong_sizes_exception("dotprod", this->size(), other.size());
        double result = 0.0;
        for (unsigned i = 0;  i < this->size();  ++i)
            result += (*this)[i] * other[i];
        return result;
    }

    template<class OFloat, class OUnderlying>
    double dotprod(const distribution<OFloat, OUnderlying> & other) const
    {
        if (this->size() != other.size())
            wrong_sizes_exception("dotprod", this->size(), other.size());
        double result = 0.0;
        for (unsigned i = 0;  i < this->size();  ++i)
            result += (*this)[i] * other[i];
        return result;
    }

    double two_norm() const
    {
        return sqrt(dotprod(*this));
    }

    void normalize()
    {
        *this /= total();
    }

    F total() const
    {
        return std::accumulate(this->begin(), this->end(), F());
    }

    double mean() const
    {
        return this->total() / this->size();
    }

    int count() const
    {
        return std::count_if(this->begin(), this->end(),
                             std::bind2nd(std::not_equal_to<F>(), F()));
    }

    F max() const
    {
        return (this->empty()
                ? -std::numeric_limits<F>::infinity()
                : *std::max_element(this->begin(), this->end()));
    }

    F min() const
    {
        return (this->empty()
                ? std::numeric_limits<F>::infinity()
                : *std::min_element(this->begin(), this->end()));
    }

    /* Only valid for bools.  Are all of them true? */
    bool all() const
    {
        for (unsigned i = 0;  i < this->size();  ++i)
            if (!this->operator [] (i)) return false;
        return true;
    }

    /* Only valid for bools.  Is any of them true? */
    bool any() const
    {
        for (unsigned i = 0;  i < this->size();  ++i)
            if (this->operator [] (i)) return true;
        return false;
    }

    template<typename T1, typename T2>
    struct is_same_type : boost::false_type {
    };

    template<typename T>
    struct is_same_type<T, T> : boost::true_type {
    };

    // Version of cast to cast to another type
    template<class Other>
    typename boost::disable_if<is_same_type<F, Other>, distribution<Other> >::type
    cast() const
    {
        return distribution<Other>(this->begin(), this->end());
    }

    // Version of cast to cast to the same type; returns a reference which is
    // more efficient.
    template<class Other>
    typename boost::enable_if<is_same_type<F, Other>, const distribution<F, Underlying> &>::type
    cast() const
    {
        return *this;
    }

    template<class Archive>
    void serialize(Archive & archive, unsigned version)
    {
        archive & (parent *)(this);
    }

    template<class Archive>
    void serialize(Archive & archive, unsigned version) const
    {
        archive & (parent *)(this);
    }
};

#define DIST_DIST_OP(op) \
template<class F1, class F2, class Underlying1, class Underlying2>     \
distribution<F1, Underlying1>           \
operator op (const distribution<F1, Underlying1> & d1, \
                 const distribution<F2, Underlying2> & d2)             \
{ \
    if (d1.size() != d2.size()) \
        wrong_sizes_exception(#op, d1.size(), d2.size());      \
    distribution<F1, Underlying1> \
        result(d1.size()); \
    for (unsigned i = 0;  i < d1.size();  ++i) \
        result[i] = d1[i] op d2[i]; \
    return result; \
}

DIST_DIST_OP(+);
DIST_DIST_OP(-);
DIST_DIST_OP(*);
DIST_DIST_OP(/);
DIST_DIST_OP(&);
DIST_DIST_OP(|);
DIST_DIST_OP(&&);
DIST_DIST_OP(||);
#undef DIST_DIST_OP

// NOTE: cannot make the two types different here, as then
// distribution-distribution operations get caught.  Need to use a MPL
// technique to disable this if the value isn't convertible to the
// underlying type.

#define SCALAR_DIST_OP(op) \
template<typename F1, typename F2, class Underlying>                 \
typename boost::enable_if<boost::is_convertible<F1, F2>, distribution<F2> >::type \
operator op (F1 val, const distribution<F2, Underlying> & d2)       \
{ \
    distribution<F2, Underlying> result(d2.size()); \
    for (unsigned i = 0;  i < d2.size();  ++i) \
        result[i] = val op d2[i]; \
    return result; \
}

SCALAR_DIST_OP(+);
SCALAR_DIST_OP(-);
SCALAR_DIST_OP(*);
SCALAR_DIST_OP(/);
SCALAR_DIST_OP(&);
SCALAR_DIST_OP(|);
SCALAR_DIST_OP(&&);
SCALAR_DIST_OP(||);
#undef SCALAR_DIST_OP

#define DIST_DIST_COMPARE_OP(op) \
template<class F, class Underlying>           \
distribution<bool> \
 operator op (const distribution<F, Underlying> & d1,\
              const distribution<F> & d2)            \
{ \
    if (d1.size() != d2.size()) \
        throw Exception("distribution sizes don't match for compare"); \
    distribution<bool> result(d1.size()); \
    for (unsigned i = 0;  i < d1.size();  ++i) \
        result[i] = (d1[i] op d2[2]); \
    return result; \
}

DIST_DIST_COMPARE_OP(==);
DIST_DIST_COMPARE_OP(!=);
DIST_DIST_COMPARE_OP(>);
DIST_DIST_COMPARE_OP(<);
DIST_DIST_COMPARE_OP(>=);
DIST_DIST_COMPARE_OP(<=);

#define DIST_SCALAR_COMPARE_OP(op) \
template<class F, class Underlying, class Other>        \
distribution<bool> \
 operator op (const distribution<F, Underlying> & d1,\
              const Other & scalar)        \
{ \
    distribution<bool> result(d1.size()); \
    for (unsigned i = 0;  i < d1.size();  ++i) \
        result[i] = (d1[i] op scalar); \
    return result; \
}

DIST_SCALAR_COMPARE_OP(==);
DIST_SCALAR_COMPARE_OP(!=);
DIST_SCALAR_COMPARE_OP(>);
DIST_SCALAR_COMPARE_OP(<);
DIST_SCALAR_COMPARE_OP(>=);
DIST_SCALAR_COMPARE_OP(<=);

template<typename F, class Underlying>
std::ostream &
operator << (std::ostream & stream, const distribution<F, Underlying> & dist)
{
    stream << "{";
    for (unsigned i = 0;  i < dist.size();  ++i)
        stream << " " << dist[i];
    return stream << " }";
}

// TODO: replace with boost::enable_if<>
template<>
inline
distribution<float>
distribution<float>::
operator & (float f) const
{
    throw Exception("distribibution<float> doesn't support operator &");
}

// TODO: replace with boost::enable_if<>
template<>
inline
distribution<double>
distribution<double>::
operator & (double f) const
{
    throw Exception("distribibution<double> doesn't support operator &");
}

// TODO: replace with boost::enable_if<>
template<>
inline
distribution<float>
distribution<float>::
operator | (float f) const
{
    throw Exception("distribibution<float> doesn't support operator &");
}

// TODO: replace with boost::enable_if<>
template<>
inline
distribution<double>
distribution<double>::
operator | (double f) const
{
    throw Exception("distribibution<double> doesn't support operator &");
}

template<>
template<class F2>
distribution<bool> &
distribution<bool>::
operator /= (F2 val)
{
    throw Exception("distribution<bool> doesn't support operator /=");
}

} // namespace Stats

using Stats::distribution;

} // namespace ML


#endif /* __stats__distribution_h__ */

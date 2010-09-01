/* sparse_distribution.h                                           -*- C++ -*-
   Jeremy Barnes, 5 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source:$
   $Id:
   
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

   Sparse version of a distribution.
*/

#ifndef __stats__sparse_distribution_h__
#define __stats__sparse_distribution_h__

#include <map>
#include <limits>
#include <ostream>

namespace ML {

template<typename Index, typename Float,
         typename Base = std::map<Index, Float> >
class sparse_distribution : public Base {
    typedef Base base_type;

public:
    sparse_distribution() {}

    template<class Iterator>
    sparse_distribution(Iterator first, Iterator last)
        : base_type(first, last)
    {
    }

    typedef typename base_type::iterator iterator;
    typedef typename base_type::const_iterator const_iterator;

    #define DIST_SCALAR_OP(op) \
    sparse_distribution \
    operator op (Float val) const;

    DIST_SCALAR_OP(+)
    DIST_SCALAR_OP(-)
    DIST_SCALAR_OP(*)
    DIST_SCALAR_OP(/)
    DIST_SCALAR_OP(&)
    DIST_SCALAR_OP(|)
    DIST_SCALAR_OP(&&)
    DIST_SCALAR_OP(||)
    #undef DIST_SCALAR_OP


    #define UPDATE_DIST_OP(op) \
    template<class F2> \
    sparse_distribution & \
    operator op (const sparse_distribution<Index, F2> & d);

    UPDATE_DIST_OP(+=)
    UPDATE_DIST_OP(-=)
    UPDATE_DIST_OP(*=)
    UPDATE_DIST_OP(/=)
    #undef UPDATE_DIST_OP
    
    #define UPDATE_SCALAR_OP(op) \
    template<class F2> \
    sparse_distribution & \
    operator op (F2 val);

    UPDATE_SCALAR_OP(+=)
    UPDATE_SCALAR_OP(-=)
    UPDATE_SCALAR_OP(*=)
    UPDATE_SCALAR_OP(/=)
    #undef UPDATE_SCALAR_OP

    void normalize()
    {
        *this /= total();
    }

    Float total() const
    {
        double result = 0.0;
        for (const_iterator it = this->begin();  it != this->end();  ++it)
            result += it->second;
        return result;
    }

    Float max() const
    {
        Float result = -std::numeric_limits<Float>::infinity();
        for (const_iterator it = this->begin();  it != this->end();  ++it)
            result = std::max(it->second, result);
        return result;
    }

    Float min() const
    {
        Float result = std::numeric_limits<Float>::infinity();
        for (const_iterator it = this->begin();  it != this->end();  ++it)
            result = std::min(it->second, result);
        return result;
    }

    template<class Archive>
    void serialize(Archive & archive, unsigned version)
    {
        archive & (base_type *)(this);
    }

    template<class Archive>
    void serialize(Archive & archive, unsigned version) const
    {
        archive & (base_type *)(this);
    }
};

#define DIST_DIST_OP(op) \
template<class I, class F> \
sparse_distribution<I, F> \
operator op (const sparse_distribution<I, F> & d1, \
             const sparse_distribution<I, F> & d2);

DIST_DIST_OP(+);
DIST_DIST_OP(-);
DIST_DIST_OP(*);
DIST_DIST_OP(/);
DIST_DIST_OP(&);
DIST_DIST_OP(|);
DIST_DIST_OP(&&);
DIST_DIST_OP(||);
#undef DIST_DIST_OP

#define SCALAR_DIST_OP(op) \
template<class I, class F> \
sparse_distribution<I, F> \
operator op (F val, const sparse_distribution<I, F> & d2);

SCALAR_DIST_OP(+);
SCALAR_DIST_OP(-);
SCALAR_DIST_OP(*);
SCALAR_DIST_OP(/);
SCALAR_DIST_OP(&);
SCALAR_DIST_OP(|);
SCALAR_DIST_OP(&&);
SCALAR_DIST_OP(||);
#undef SCALAR_DIST_OP

template<typename I, typename F>
std::ostream &
operator << (std::ostream & stream, const sparse_distribution<I, F> & dist);

} // namespace ML

#endif /* __stats__sparse_distribution_h__ */

/* fixed_array.h                                                   -*- C++ -*-
   Jeremy Barnes, 5 February 2005
   Copyright(c) 2005 Jeremy Barnes.  All rights reserved.
      
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

   Fixed array.  Just a boost::multi_array, but with a few extra convenience
   methods.
*/

#ifndef __utils__fixed_array_h__
#define __utils__fixed_array_h__


#include <boost/multi_array.hpp>
#include <boost/type_traits.hpp>
#include "jml/utils/multi_array_utils.h"
#include <boost/function.hpp>
#include <numeric>


namespace ML {


template<class T>
struct array_deleter {
    array_deleter(T * p) : p(p) {}

    void operator () () const
    {
        delete[] (p);
    }

    T * p;
};

template<size_t Dim, class Array>
size_t product(const Array & array)
{
    size_t result = 1;
    for (unsigned i = 0;  i < Dim;  ++i) result *= array[i];
    return result;
}

template<size_t Dim>
size_t array_size(const boost::detail::multi_array::extent_gen<Dim> & ex)
{
    size_t result = 1;
    for (unsigned i = 0;  i < Dim;  ++i)
        result *= ex.ranges_[i].size();
    return result;
}

template<typename T, size_t Dim,
         class Allocator = std::allocator<typename boost::remove_const<T>::type > >
class fixed_array_base
    : public boost::multi_array<T, Dim, Allocator> {

public:
    typedef boost::multi_array<T, Dim, Allocator> base_type;

    fixed_array_base()
    {
    }

    template <typename ExtentList>
    explicit
    fixed_array_base(const ExtentList & sizes,
                     const typename base_type::storage_order_type & store
                         = boost::c_storage_order(),
                     const Allocator & alloc = Allocator())
        : base_type(sizes, store, alloc)
    {
    }

    explicit
    fixed_array_base(const boost::detail::multi_array::extent_gen<Dim> & dims)
        : base_type(dims)
    {
    }

    fixed_array_base & operator = (const fixed_array_base & other)
    {
        fixed_array_base new_me(other);
        swap(new_me);
    }

    size_t dim(int dimension) const { return this->shape()[dimension]; }

    void fill(T val)
    {
        std::fill(data_begin(), data_end(), val);
    }

    T * data_begin()
    {
        return this->data();
    }

    T * data_end()
    {
        return this->data() + this->num_elements();
    }

    const T * data_begin() const
    {
        return this->data();
    }

    const T * data_end() const
    {
        return this->data() + this->num_elements();
    }

    void swap(fixed_array_base & other)
    {
        boost::swap(*this, other);
    }
};

template<typename T, size_t Dim>
class fixed_array {
};

template<typename T>
class fixed_array<T, 1> : public fixed_array_base<T, 1> {
    typedef fixed_array_base<T, 1> base_type;

public:
    fixed_array(int d0 = 0)
        : base_type(boost::extents[d0])
    {
    }

#if 0
    fixed_array(int d0, T * data, bool delete_data = false)
        : base_type(data, boost::extents[d0])
    {
        if (delete_data) deleter_ = array_deleter<T>(data);
    }
#endif

    fixed_array(const base_type & base)
        : base_type(base)
    {
    }

    fixed_array deep_copy() const
    {
        return fixed_array(*(const base_type *)this);
    }
};

template<typename T>
class fixed_array<T, 2> : public fixed_array_base<T, 2> {
    typedef fixed_array_base<T, 2> base_type;
public:
    fixed_array(int d0 = 0, int d1 = 0)
        : base_type(boost::extents[d0][d1])
    {
    }

#if 0
    fixed_array(int d0, int d1, T * data, bool delete_data = false)
        : base_type(data, boost::extents[d0][d1])
    {
        if (delete_data) deleter_ = array_deleter<T>(data);
    }
#endif

    fixed_array(const base_type & base)
        : base_type(base)
    {
    }

    fixed_array deep_copy() const
    {
        return fixed_array(*(const base_type *)this);
    }
};

template<typename T>
class fixed_array<T, 3> : public fixed_array_base<T, 3> {
    typedef fixed_array_base<T, 3> base_type;

public:
    fixed_array(int d0 = 0, int d1 = 0, int d2 = 0)
        : base_type(boost::extents[d0][d1][d2])
    {
    }

#if 0
    fixed_array(int d0, int d1, int d2, T * data, bool delete_data = false)
        : base_type(data, boost::extents[d0][d1][d2])
    {
        if (delete_data) deleter_ = array_deleter<T>(data);
    }
#endif

    fixed_array(const base_type & base)
        : base_type(base)
    {
    }

    fixed_array deep_copy() const
    {
        return fixed_array(*(const base_type *)this);
    }
};


} // namespace ML


#endif /* __utils__fixed_array_h__ */


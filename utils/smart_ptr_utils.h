/* smart_ptr_utils.h                                               -*- C++ -*-
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

   Utilities to help with smart pointers.
*/

#ifndef __utils__smart_ptr_utils_h__
#define __utils__smart_ptr_utils_h__


#include <boost/shared_ptr.hpp>
#include "jml/compiler/compiler.h"
#include <memory>

namespace ML {

template<class T>
std::shared_ptr<T> make_sp(T * val)
{
    return std::shared_ptr<T>(val);
}

template<class T>
std::shared_ptr<T> make_std_sp(T * val)
{
    return std::shared_ptr<T>(val);
}

struct Dont_Delete {
    template<class X> void operator () (const X & x) const
    {
    }
};

template<class T>
std::shared_ptr<T> make_unowned_sp(T & val)
{
    return std::shared_ptr<T>(&val, Dont_Delete());
}

template<class T>
std::shared_ptr<const T> make_unowned_sp(const T & val)
{
    return std::shared_ptr<const T>(&val, Dont_Delete());
}

template<class T>
std::shared_ptr<T> make_unowned_std_sp(T & val)
{
    return std::shared_ptr<T>(&val, Dont_Delete());
}

template<class T>
std::shared_ptr<const T> make_unowned_std_sp(const T & val)
{
    return std::shared_ptr<const T>(&val, Dont_Delete());
}

extern const struct Null_SP {
    template<typename T>
    JML_ALWAYS_INLINE operator std::shared_ptr<T>() const
    {
        return std::shared_ptr<T>();
    }

} NULL_SP;

extern const struct Null_STD_SP {
    template<typename T>
    JML_ALWAYS_INLINE operator std::shared_ptr<T>() const
    {
        return std::shared_ptr<T>();
    }

} NULL_STD_SP;

template<typename T>
struct Keep_Ref {
    Keep_Ref(const std::shared_ptr<T> & p)
        : ref(p)
    {
    }

    std::shared_ptr<T> ref;

    template<class X> void operator () (const X & x) const
    {
    }
};

template<class T>
std::shared_ptr<T> make_sp(const std::shared_ptr<T> & val)
{
    return std::shared_ptr<T>(val.get(), Keep_Ref<T>(val));
}

template<typename T>
struct Keep_RefB {
    Keep_RefB(const std::shared_ptr<T> & p)
        : ref(p)
    {
    }

    std::shared_ptr<T> ref;

    template<class X> void operator () (const X & x) const
    {
    }
};

template<class T>
std::shared_ptr<T> make_std_sp(const std::shared_ptr<T> & val)
{
    return std::shared_ptr<T>(val.get(), Keep_RefB<T>(val));
}

} // namespace ML

#endif /* __utils__smart_ptr_utils_h__ */

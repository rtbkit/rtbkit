/* guard.h                                                         -*- C++ -*-
   Jeremy Barnes, 13 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

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
*/

#ifndef __utils__guard_h__
#define __utils__guard_h__

#include "jml/compiler/compiler.h"
#include <boost/function.hpp>

namespace ML {

struct Call_Guard {
    
    typedef boost::function<void ()> Fn;

    Call_Guard(const Fn & fn, bool condition = true)
        : fn(condition ? fn : Fn())
    {
    }

    Call_Guard()
    {
    }
    
#if JML_HAS_RVALUE_REFERENCES
    Call_Guard(Call_Guard && other)
        : fn(other.fn)
    {
        other.clear();
    }

    Call_Guard & operator = (Call_Guard && other)
    {
        if (fn) fn();
        fn = other.fn;
        other.clear();
        return *this;
    }
#endif

    ~Call_Guard()
    {
        if (fn) fn();
    }

    void clear() { fn = Fn(); }

    void set(const Fn & fn) { this->fn = fn; }

    boost::function<void ()> fn;

private:
    Call_Guard(const Call_Guard & other);
    void operator = (const Call_Guard & other);
};

#if JML_HAS_RVALUE_REFERENCES
template<typename Fn>
Call_Guard call_guard(Fn fn, bool condition = true)
{
    return Call_Guard(fn, condition);
}
#endif


} // namespace ML


#endif /* __utils__guard_h__ */

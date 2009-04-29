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

#include <boost/function.hpp>

namespace ML {

struct Call_Guard {
    
    typedef boost::function<void ()> Fn;


    Call_Guard(const Fn & fn)
        : fn(fn)
    {
    }

    Call_Guard()
    {
    }
    
    ~Call_Guard()
    {
        if (fn) fn();
    }

    void clear() { fn = Fn(); }

    void set(const Fn & fn) { this->fn = fn; }

    boost::function<void ()> fn;
};


} // namespace ML


#endif /* __utils__guard_h__ */

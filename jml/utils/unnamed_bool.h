/* unnamed_bool.h                                                  -*- C++ -*-
   Jeremy Barnes, 30 January 2005
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

   An unnamed bool type.  Used to enable operator bool without having the
   object be convertable to int.
*/

#ifndef __utils__unnamed_bool_h__
#define __utils__unnamed_bool_h__

#include "jml/compiler/compiler.h"

namespace ML {

struct unnamed_bool_t {
    unnamed_bool_t(bool val)
        : val(val) {}

    int fn() const { return 0; }
    
    typedef int (unnamed_bool_t::* type)() const;

    operator type () const
    {
        static const type true_val = &unnamed_bool_t::fn;
        static const type false_val = 0;
        return val ? true_val : false_val;
    }

    bool val;
};

typedef unnamed_bool_t::type unnamed_bool;
static const unnamed_bool unnamed_true = &unnamed_bool_t::fn;
static const unnamed_bool unnamed_false = 0;

JML_ALWAYS_INLINE unnamed_bool make_unnamed_bool(bool val)
{
    return (val ? unnamed_true : unnamed_false);
}

#define JML_IMPLEMENT_OPERATOR_BOOL(expr) \
    operator ML::unnamed_bool () const { return ML::make_unnamed_bool(expr); }

} // namespace ML

#endif /* __utils__unnamed_bool_h__ */


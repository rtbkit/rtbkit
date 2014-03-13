/* xdiv.h                                                          -*- C++ -*-
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

   Our old friend the xdiv function.
*/

#ifndef __math__xdiv_h__
#define __math__xdiv_h__

#include "jml/utils/float_traits.h"
#include "jml/compiler/compiler.h"

namespace ML {

template<typename F1, typename F2>
typename float_traits<F1, F2>::fraction_type
xdiv(F1 x, F2 y)
{
    return (y == 0 ? 0 : x / y);
}

/* Divide, but round up */
template<class X, class Y>
JML_COMPUTE_METHOD
X rudiv(X val, Y by)
{
    X result = (val / by);
    X missing = val - (result * by);
    result += (missing > 0);
    return result;
}


} // namespace ML


#endif /* __math__xdiv_h__ */

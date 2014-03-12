/* bound.h                                                         -*- C++ -*-
   Jeremy Barnes, 15 March 2005
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

   Bound function.  Returns a value bounded by a minimum and maximum.
*/

#ifndef __math__bound_h__
#define __math__bound_h__

#include "jml/utils/float_traits.h"

namespace ML {

template<class X, class Y, class Z>
typename float_traits3<X, Y, Z>::return_type
bound(X x, Y min, Z max)
{
    typename float_traits3<X, Y, Z>::return_type result = x;
    if (result < min) result = min;
    if (result > max) result = max;
    return result;
}

template<class X>
X bound(X x, X min, X max)
{
    if (x < min) x = min;
    if (x > max) x = max;
    return x;
}

} // namespace ML

#endif /* __math__bound_h__ */

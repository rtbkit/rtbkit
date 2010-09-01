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

#ifndef __arch__math_builtins_h__
#define __arch__math_builtins_h__

#include "jml/arch/arch.h"
#include <math.h>

namespace ML {

// 4 June 2009, Jeremy, Ubuntu 9.04
// For some reason, the x86_64 implementation of expf sets and resets the FPU
// control word, which makes it extremely slow.  The expm1f implementation
// doesn't have this problem, and so we use it to emulate an exp.

inline float exp(float val)
{
    return expm1f(val) + 1.0f;
}

inline double exp(double val)
{
    return std::exp(val);
}

} // namespace ML

#endif /* __arch__math_builtins_h__ */

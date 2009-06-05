/* distribution_simd.h                                             -*- C++ -*-
   Jeremy Barnes, 12 March 2005
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

   Vectorizes some distribution operations.
*/

#ifndef __stats__distribution_simd_h__
#define __stats__distribution_simd_h__

#include "distribution.h"
#include "arch/simd_vector.h"

namespace ML {
namespace Stats {

template<>
inline float
distribution<float>::
total() const
{
    return SIMD::vec_sum_dp(&(*this)[0], this->size());
}

} // namespace Stats
} // namespace ML

#endif /* __stats__distribution_simd_h__ */

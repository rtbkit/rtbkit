/* distribution.cc
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

   Implementation of static distribution functions.
*/

#include "distribution.h"
#include "jml/arch/exception.h"

namespace ML {
namespace Stats {

#if 0
void wrong_sizes_exception()
{
    throw Exception("distribution: operation between different sizes");
}
#endif

} // namespace Stats
} // namespace ML

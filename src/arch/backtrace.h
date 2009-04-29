/* backtrace.h                                                      -*- C++ -*-
   Jeremy Barnes, 26 February 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   This file is part of "Jeremy's Machine Learning Library", copyright (c)
   1999-2009 Jeremy Barnes.
   
   This program is available under the GNU General Public License, the terms
   of which are given by the file "license.txt" in the top level directory of
   the source code distribution.  If this file is missing, you have no right
   to use the program; please contact the author.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
   for more details.

   ---

   Interface to a bactrace function.
*/

#include <iostream>

namespace ML {

void backtrace(std::ostream & stream = std::cerr, int num_to_skip = 1);

} // namespace ML

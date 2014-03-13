/* hash_map.h                                                      -*- C++ -*-
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

   Include file to find the hash map class and make it available as if it was
   in the standard library.
*/

#ifndef __utils__hash_map_h__
#define __utils__hash_map_h__


#define _BACKWARD_BACKWARD_WARNING_H 1
#include <ext/hash_map>
#include "hash_specializations.h"

namespace std {

using __gnu_cxx::hash_map;
using __gnu_cxx::hash_multimap;

} // namespace std

#endif /* __utils__hash_map_h__ */



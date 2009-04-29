/* info.h                                                          -*- C++ -*-
   Jeremy Barnes, 3 April 2006
   Copyright (c) 2006 Jeremy Barnes.  All rights reserved.

   This file is part of "Jeremy's Machine Learning Library", copyright (c)
   1999-2006 Jeremy Barnes.
   
   This program is available under the GNU General Public License, the terms
   of which are given by the file "license.txt" in the top level directory of
   the source code distribution.  If this file is missing, you have no right
   to use the program; please contact the author.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
   for more details.

   ---

   Generic information about the current machine.
*/

#ifndef __utils__info_h__
#define __utils__info_h__

#include <string>
#include "compiler/compiler.h"

namespace ML {

/** A compact string giving context about the current program. */

std::string all_info();

/** Returns the number of CPU cores installed in the system. */

extern int num_cpus_result;

void init_num_cpus();

JML_ALWAYS_INLINE int num_cpus()
{
    if (JML_UNLIKELY(!num_cpus_result)) init_num_cpus();
    return num_cpus_result;
}

/** Return the username of the current user. */

std::string username();

int userid();

std::string userid_to_username(int userid);

} // namespace ML

#endif /* __utils__info_h__ */

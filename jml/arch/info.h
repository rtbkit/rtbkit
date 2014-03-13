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
#include "jml/compiler/compiler.h"
#include "jml/arch/cpu_info.h"

namespace ML {

/** A compact string giving context about the current program. */

std::string all_info();

/** Return the username of the current user. */

std::string username();

std::string hostname();

std::string fqdn_hostname(std::string const & port);

int userid();

std::string userid_to_username(int userid);

/** Returns the number of file descriptors that the process has open. */
size_t num_open_files();

/** Turn an fd into a filename */
std::string fd_to_filename(int fd);

} // namespace ML

#endif /* __utils__info_h__ */

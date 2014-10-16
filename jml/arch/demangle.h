/* demangle.h                                                      -*- C++ -*-
   Jeremy Barnes, 27 January 2005
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

   Interface to a demangler.
*/

#ifndef __arch__demangle_h__
#define __arch__demangle_h__


#include <string>
#include <typeinfo>

namespace ML {

/* returns a null-terminated string allocated on the heap */
char * char_demangle(const char * name);

std::string demangle(const std::string & name);
std::string demangle(const std::type_info & type);

template<typename T>
std::string type_name(const T & val)
{
    return demangle(typeid(val));
}

template<typename T>
std::string type_name()
{
    return demangle(typeid(T));
}

} // namespace ML

#endif /* __arch__demangle_h__ */


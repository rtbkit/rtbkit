/* string_functions.h                                              -*- C++ -*-
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

   Functions for the manipulation of strings.
*/

#ifndef __utils__string_functions_h__
#define __utils__string_functions_h__

#include <sstream>
#include <string>
#include <vector>
#include "jml/arch/format.h"

namespace ML {


template<typename T>
std::string ostream_format(const T & val)
{
    std::ostringstream str;
    str << val;
    return str.str();
}

std::vector<std::string> split(const std::string & str, char c = ' ');

std::string lowercase(const std::string & str);

std::string remove_trailing_whitespace(const std::string & str);

/** If the given string ends with the ending, then remove that ending from the
    string and return true.  Otherwise return false.
*/
bool removeIfEndsWith(std::string & str, const std::string & ending);

bool endsWith(const std::string & haystack, const std::string & needle);

/* replace unprintable characters with a hex representation thereof */
std::string hexify_string(const std::string & str);

/* Parse an integer stored in the chars between "start" and "end",
   where all characters are expected to be strict digits. The name is inspired
   from "atoi" with the "n" indicating that it is reading only from a numbered
   set of bytes. Base 10 can be negative. */
int antoi(const char * start, const char * end, int base = 10);

} // namespace ML


#endif /* __utils__string_functions_h__ */

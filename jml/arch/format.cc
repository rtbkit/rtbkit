/* format.cc                                                       -*- C++ -*-
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

   Functions for the manipulation of strings.
*/

#include "format.h"
#include "exception.h"
#include <stdlib.h>
#include <errno.h>
#include <stdio.h>

using namespace std;

namespace ML {

struct va_ender {
    va_ender(va_list & ap)
        : ap(ap)
    {
    }

    ~va_ender()
    {
        va_end(ap);
    }

    va_list & ap;
};

std::string formatImpl(const char * fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    try {
        string result = vformat(fmt, ap);
        va_end(ap);
        return result;
    }
    catch (...) {
        va_end(ap);
        throw;
    }
}

std::string vformat(const char * fmt, va_list ap)
{
    char * mem;
    string result;
    int res = vasprintf(&mem, fmt, ap);
    if (res < 0)
        throw Exception("format(): vasprintf error on %s", fmt);

    try {
        result = mem;
        free(mem);
        return result;
    }
    catch (...) {
        free(mem);
        throw;
    }
}

} // namespace ML

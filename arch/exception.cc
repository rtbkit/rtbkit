/* exception.cc
   Jeremy Barnes, 7 February 2005
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

   Exception class.
*/

#include "exception.h"
#include "format.h"
#include <string.h>
#include <cxxabi.h>
#include "demangle.h"

using namespace std;


namespace ML {

Exception::Exception(const std::string & msg)
    : message(msg)
{
    message.c_str();  // make sure we have a null terminator
}

Exception::Exception(const char * msg, ...)
{
    va_list ap;
    va_start(ap, msg);
    try {
        message = vformat(msg, ap);
        message.c_str();
        va_end(ap);
    }
    catch (...) {
        va_end(ap);
        throw;
    }
}

Exception::Exception(const char * msg, va_list ap)
{
    message = vformat(msg, ap);
    message.c_str();
}

Exception::
Exception(int errnum, const std::string & msg, const char * function)
{
    string error = strerror(errnum);

    if (function) {
        message = function;
        message += ": ";
    }

    message += msg;
    message += ": ";

    message += error;

    message.c_str();
}

Exception::~Exception() throw()
{
}

const char * Exception::what() const throw()
{
    return message.c_str();
}

std::string getExceptionString()
{
    const std::type_info* t = __cxxabiv1::__cxa_current_exception_type();
    return demangle(t->name());
}


} // namespace ML

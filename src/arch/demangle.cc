/* demangle.cc
   Jeremy Barnes, 17 March 2005
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

   Demangler.  Just calls the ABI one, but does the memory management for
   us.
*/

#include "demangle.h"

#include <string>
#include <cxxabi.h>
#include <stdlib.h>

using namespace std;


namespace ML {

std::string demangle(const std::string & name)
{
    int status;
    string result;
    char * ptr = abi::__cxa_demangle(name.c_str(), 0, 0, &status);

    if (status != 0)
        return name;
        //return "<<DEMANGLE ERROR: " + name + ">>";
    
    try {
        result = ptr;
    }
    catch (...) {
        free(ptr);
        throw;
    }

    free(ptr);
    return result;
}

} // namespace ML


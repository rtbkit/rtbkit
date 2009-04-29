/* backtrace.cc                                                    -*- C++ -*-
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
#include <execinfo.h>
#include <stdlib.h>
#include "demangle.h"
#include "format.h"

// Include the GNU extentions necessary for this functionality
//#define _GNU_SOURCE
//#define __USE_GNU 1
#include <dlfcn.h>
//#undef _GNU_SOURCE
//#undef __USE_GNU


using namespace std;

namespace ML {

void backtrace(std::ostream & stream, int num_to_skip)
{
    /* Obtain a backtrace and print it to stdout. */
    void *array[200];
    
    size_t size = ::backtrace (array, 200);
 
    for (unsigned i = num_to_skip;  i < size;  ++i) {
        stream << format("%4d: %016p ", i - num_to_skip, array[i]);

        Dl_info info;
        int ret = dladdr( array[i], &info);
        if (ret == 0) {
            stream << " (unknown)" << endl;
            continue;
        }

        if (info.dli_sname) {
            stream << demangle(info.dli_sname)
                   << format(" + 0x%x", (const char *)array[i]
                                   - (const char *)info.dli_saddr);
        }
        if (info.dli_fname) {
            stream << format(" in %s + 0x%x", info.dli_fname,
                             (const char *)array[i]
                             - (const char *)info.dli_fbase)
                   << endl;
        }
        else stream << "unknown" << endl;
    }
}

} // namespace ML

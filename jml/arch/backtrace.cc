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

#include "backtrace.h"
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

size_t backtrace(char * buffer, size_t bufferSize, int num_to_skip)
{
    size_t remaining(bufferSize);
    size_t written, totalWritten(0);

    vector<BacktraceFrame> result = backtrace(num_to_skip);

    for (unsigned i = 0;  i < result.size();  ++i) {
        string line = result[i].print();
        written = snprintf(buffer + totalWritten, remaining,
                           "%02u: %s\n", i, line.c_str());
        totalWritten += written;
        if (written >= remaining) {
            break;
        }
        remaining -= written;
    }

    return totalWritten;
}

void backtrace(std::ostream & stream, int num_to_skip)
{
    vector<BacktraceFrame> result = backtrace(num_to_skip);

    for (unsigned i = 0;  i < result.size();  ++i)
        stream << format("%02d: ", i) << result[i].print() << endl;
}

/** The information in a backtrace frame. */
BacktraceFrame::
BacktraceFrame(int num, const void * frame, const std::string & symbol)
{
    init(num, frame, symbol);
}

void
BacktraceFrame::
init(int num, const void * frame, const std::string & symbol)
{
    address = frame;
    number = num;

    Dl_info info;
    int ret = 0;
    if (frame) ret = dladdr( frame, &info);

    if (ret == 0) {
        function = object = "";
        function_start = object_start = 0;
        return;
    }

    if (info.dli_sname) {
        function = demangle(info.dli_sname);
        function_start = info.dli_saddr;
    }
    else {
        function = "";
        function_start = 0;
    }
    if (info.dli_fname) {
        object = info.dli_fname;
        object_start = info.dli_fbase;
    }
    else {
        object = "";
        object_start = 0;
    }
    this->symbol = symbol;
}

static ssize_t ptr_offset(const void * from, const void * to)
{
    return (const char *)to - (const char *)from;
}

std::string
BacktraceFrame::
print() const
{
    string result = format("0x%8p", address);

    if (function_start)
        result += format(" at %s + 0x%zx", function.c_str(),
                         ptr_offset(function_start, address));
    if (object_start)
        result += format(" in %s + 0x%zx", object.c_str(),
                         ptr_offset(object_start, address));
    if (symbol != "")
        result += symbol;
    
    return result;
}

std::string
BacktraceFrame::
print_for_trace() const
{
    if (!address)
        return "(uninitialized)";
    else if (function != "")
        return function;
    else if (object != "")
        return "in " + object;
    else return format("0x%8p", address);
}

std::vector<BacktraceFrame> backtrace(int num_to_skip)
{
    /* Obtain a backtrace and print it to stdout. */
    void *array[200];
    
    size_t size = ::backtrace (array, 200);

    char ** symbols = ::backtrace_symbols (array, size);

    vector<BacktraceFrame> result;

    if (symbols) {
        for (unsigned i = num_to_skip;  i < size;  ++i)
            result.push_back(BacktraceFrame(i, array[i], symbols[i]));
    }
    else {
        for (unsigned i = num_to_skip;  i < size;  ++i)
            result.push_back(BacktraceFrame(i, array[i]));
    }

    free(symbols);
    
    return result;
}

std::vector<BacktraceFrame>
backtrace(const BacktraceInfo & info,
          int num_to_skip)
{
    /* Obtain a backtrace and print it to stdout. */
    vector<BacktraceFrame> result;

    for (unsigned i = num_to_skip;  i < info.size;  ++i)
        result.push_back(BacktraceFrame(i, info.frames[i]));
    
    return result;
}

} // namespace ML

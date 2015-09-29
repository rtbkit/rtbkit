/* string_functions.cc
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
   
   String manipulation functions.
*/

#include "string_functions.h"
#include <stdarg.h>
#include <stdio.h>
#include "jml/arch/exception.h"
#include <sys/errno.h>
#include <stdlib.h>

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

std::string format(const char * fmt, ...)
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
        throw Exception(errno, "vasprintf", "format()");

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

std::vector<std::string> split(const std::string & str, char c)
{
    vector<string> result;
    size_t start = 0;
    size_t pos = 0;
    while (pos < str.size()) {
        if (str[pos] == c) {
            result.push_back(string(str, start, pos - start));
            start = pos + 1;
        }
        ++pos;
    }

    if (start < str.size())
        result.push_back(string(str, start, pos - start));

    return result;
}

std::string lowercase(const std::string & str)
{
    string result = str;
    for (unsigned i = 0;  i < str.size();  ++i)
        result[i] = tolower(result[i]);
    return result;
}

std::string remove_trailing_whitespace(const std::string & str)
{
    int startOfSpace = -1;
    for (unsigned i = 0;  i < str.length();  ++i) {
        if (isspace(str[i])) {
            if (startOfSpace == -1) startOfSpace = i;
        }
        else startOfSpace = -1;
    }

    if (startOfSpace == -1) return str;
    return string(str, 0, startOfSpace);
}

bool removeIfEndsWith(std::string & str, const std::string & ending)
{
    if (str.rfind(ending) == str.size() - ending.length()) {
        str = string(str, 0, str.size() - ending.length());
        return true;
    }
    
    return false;
}

bool endsWith(const std::string & haystack, const std::string & needle)
{
    string::size_type result = haystack.rfind(needle);
    return result != string::npos
        && result == haystack.size() - needle.size();
}

std::string hexify_string(const std::string & str)
{
    size_t i, len(str.size());
    std::string newString;
    newString.reserve(len * 3);

    for (i = 0; i < len; i++) {
        if (str[i] < 32 || str[i] > 127) {
            newString += format("\\x%.2x", int(str[i] & 0xff));
        }
        else {
            newString += str[i];
        }
    }

    return newString;
}

int
antoi(const char * start, const char * end, int base)
{
    int result(0);
    bool neg = false;
    if (*start == '-') {
        if (base == 10) {
            neg = true;
        }
        else {
            throw ML::Exception("Cannot negate non base 10");
        }
        start++;
    }
    else if (*start == '+') {
        start++;
    }

    for (const char * ptr = start; ptr < end; ptr++) {
        int digit;
        if (*ptr >= '0' and *ptr <= '9') {
            digit = *ptr - '0';
        }
        else if (*ptr >= 'A' and *ptr <= 'F') {
            digit = *ptr - 'A' + 10;
        }
        else if (*ptr >= 'a' and *ptr <= 'f') {
            digit = *ptr - 'a' + 10;
        }
        else {
            throw ML::Exception("expected digit");
        }
        if (digit > base) {
            intptr_t offset = ptr - start;
            throw ML::Exception("digit '%c' (%d) exceeds base '%d'"
                                " at offset '%d'",
                                *ptr, digit, base, offset);
        }
        result = result * base + digit;
    }

    if (neg) {
        return result * -1;
    }

    return result;
}

} // namespace ML

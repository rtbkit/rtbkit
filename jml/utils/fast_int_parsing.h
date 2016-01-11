/* fast_int_parsing.h                                              -*- C++ -*-
   Jeremy Barnes, 24 March 2005
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

   Routines to quickly parse a base 10 integer.
*/

#ifndef __utils__fast_int_parsing_h__
#define __utils__fast_int_parsing_h__

#include "jml/utils/parse_context.h"
#include <iostream>

using namespace std;


namespace ML {

inline bool match_unsigned(unsigned long & val, Parse_Context & c)
{
    Parse_Context::Revert_Token tok(c);

    val = 0;
    unsigned digits = 0;
    
    while (c) {
        if (isdigit(*c)) {
            int digit = *c - '0';
            val = val * 10 + digit;
            ++digits;
        }
        else break;
        
        ++c;
    }
    
    if (!digits) return false;
    
    tok.ignore();  // we are returning true; ignore the token
    
    return true;
}

inline bool match_int(long int & result, Parse_Context & c)
{
    Parse_Context::Revert_Token tok(c);

    int sign = 1;
    if (c.match_literal('+')) ;
    else if (c.match_literal('-')) sign = -1;

    long unsigned mag;
    if (!match_unsigned(mag, c)) return false;

    result = mag * sign;

    tok.ignore();
    return true;
}

inline bool match_hex4(long int & result, Parse_Context & c)
{
    Parse_Context::Revert_Token tok(c);

    int code = 0;
    unsigned digits = 0;
    
    for(unsigned i=0 ; i<4; ++i) {
        int digit;
        if(c && isalnum(*c)) {
            int car = *c;
            if (car >= '0' && car <= '9')
                digit = car - '0';
            else if (car >= 'a' && car <= 'f')
                digit = car - 'a' + 10 ;
            else if (car >= 'A' && car <= 'F')
                digit = car - 'A' + 10;
            else {
                return false;
            }
            code = (code << 4) | digit;
            ++digits;
        }
        else {
            break;
        }
        c++;
    }

    if(digits!=4)
        return false;

    result = code;

    tok.ignore();
    return true;
}

inline bool match_unsigned_long(unsigned long & val,
                                Parse_Context & c)
{
    Parse_Context::Revert_Token tok(c);

    val = 0;
    int digits = 0;
    
    while (c) {
        if (isdigit(*c)) {
            int digit = *c - '0';
            val = val * 10 + digit;
            ++digits;
        }
        else break;
        
        ++c;
    }
    
    if (!digits) return false;
    
    tok.ignore();  // we are returning true; ignore the token
    
    return true;
}

inline bool match_long(long int & result, Parse_Context & c)
{
    Parse_Context::Revert_Token tok(c);

    long sign = 1;
    if (c.match_literal('+')) ;
    else if (c.match_literal('-')) sign = -1;

    long unsigned mag;
    if (!match_unsigned_long(mag, c)) return false;
    
    result = (long)mag;
    result *= sign;
    
    tok.ignore();
    return true;
}


inline bool match_unsigned_long_long(unsigned long long & val,
                                     Parse_Context & c)
{
    Parse_Context::Revert_Token tok(c);

    val = 0;
    int digits = 0;

    bool dot = false;
    int coeffs = 0;
    
    while (c) {
        if (isdigit(*c)) {
            int digit = *c - '0';
            val = val * 10 + digit;
            ++digits;
            if (dot) ++coeffs;
        }
        else if (*c == '.') { dot = true; }
        else {
            break;
        }
        
        ++c;
    }
    
    if (!digits) return false;
    if (dot) {
        if (!c.match_literal('e')) return false;
        if (!c.match_literal('+')) return false;

        long int exp;
        if (!match_int(exp, c)) return false;
        if (exp != coeffs) {
            if (exp < coeffs) return false;
            else val *= static_cast<unsigned>(std::pow(10, exp - coeffs));
        }
    }
    
    tok.ignore();  // we are returning true; ignore the token
    
    return true;
}

inline bool match_long_long(long long int & result, Parse_Context & c)
{
    Parse_Context::Revert_Token tok(c);

    long long sign = 1;
    if (c.match_literal('+')) ;
    else if (c.match_literal('-')) sign = -1;

    long long unsigned mag;
    if (!match_unsigned_long_long(mag, c)) return false;

    result = (long long)mag;
    result *= sign;

    tok.ignore();
    return true;
}

} // namespace ML

#endif /* __utils__fast_int_parsing_h__ */

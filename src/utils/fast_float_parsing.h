/* fast_float_parsing.h                                            -*- C++ -*-
   Jeremy Barnes, 25 February 2005
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

   Fast inline float parsing routines.
*/

#ifndef __utils__fast_float_parsing_h__
#define __utils__fast_float_parsing_h__


#include "utils/parse_context.h"
#include <limits>


namespace ML {

inline bool match_float(float & result, Parse_Context & c)
{
    Parse_Context::Revert_Token tok(c);

    unsigned long num = 0;
    double den = 0.0;
    double sign = 1;
    int digits = 0;

    if (c.match_literal('+')) ;
    else if (c.match_literal('-')) sign = -1.0;

    if (*c == 'n') {
        ++c;
        if (!c.match_literal("an"))
            return false;
        tok.ignore();
        result = sign * std::numeric_limits<float>::quiet_NaN();
        return true;
    }
    else if (*c == 'i') {
        ++c;
        if (!c.match_literal("nf"))
            return false;
        tok.ignore();
        result = sign * INFINITY;
        return true;
    }

    while (c) {
        if (isdigit(*c)) {
            int digit = *c - '0';
            num = 10*num + digit;
            den *= 0.1;
            ++digits;
        }
        else if (*c == '.') {
            if (den != 0.0) break;
            else den = 1.0;
        }
        else if ((*c == 'e' || *c == 'E')) {
            ++c;
            if (c.match_literal('+'));
            int expi = c.expect_int();
            sign *= exp10(expi);
            break;
        }
        else break;
        ++c;
    }

    if (!digits) return false;

    tok.ignore();  // we are returning true; ignore the token
    
    if (den == 0.0) result = sign * num;
    else result = sign * (double)num * den;

    return true;
}

inline float expect_float(Parse_Context & c,
                          const char * error = "expected float")
{
    float result;
    if (!match_float(result, c))
        c.exception(error);
    return result;
}


} // namespace ML


#endif /* __utils__fast_float_parsing_h__ */

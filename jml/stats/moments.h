/* moments.h                                                       -*- C++ -*-
   Jeremy Barnes, 2 February 2005
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

   Functions to do with calculating moments (mean, std dev, etc).
*/

#ifndef __stats__moments_h__
#define __stats__moments_h__

#include <limits>
#include <cmath>

namespace ML {

template<class Iterator>
double mean(Iterator first, Iterator last)
{
    double result = 0.0;
    int divisor = 0;

    while (first != last) {
        result += *first++;
        divisor += 1;
    }

    return result / divisor;
}

inline double sqr(double val)
{
    return val * val;
}

/** Unbiased estimate of standard deviation. */

template<class Iterator>
double std_dev(Iterator first, Iterator last, double mean)
{
    double total = 0.0;
    size_t count = std::distance(first, last);

    while (first != last)
        total += sqr(*first++ - mean);
    
    if (count == 0 || count == 1)
        return std::numeric_limits<double>::quiet_NaN();

    return std::sqrt(total / (double)(count - 1));
}

} // namespace ML


#endif /* __stats__moments_h__ */



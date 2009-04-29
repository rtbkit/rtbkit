/* less.h                                                          -*- C++ -*-
   Jeremy Barnes, 5 March 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Functions to implement an operator <.
*/

#ifndef __utils__less_h__
#define __utils__less_h__

namespace ML {

template<typename T1>
bool less_all(const T1 & x1, const T1 & y1)
{
    return x1 < y1;
}

template<typename T1, typename T2>
bool less_all(const T1 & x1, const T1 & y1,
              const T2 & x2, const T2 & y2)
{
    return x1 < y1
        || (x1 == y1
            && (x2 < y2));
}

template<typename T1, typename T2, typename T3>
bool less_all(const T1 & x1, const T1 & y1,
              const T2 & x2, const T2 & y2,
              const T3 & x3, const T3 & y3)
{
    return x1 < y1
        || (x1 == y1
            && (x2 < y2
                || (x2 == y2
                    && (x3 < y3))));
}

template<typename T1, typename T2, typename T3, typename T4>
bool less_all(const T1 & x1, const T1 & y1,
              const T2 & x2, const T2 & y2,
              const T3 & x3, const T3 & y3,
              const T4 & x4, const T4 & y4)
{
    return x1 < y1
        || (x1 == y1
            && (x2 < y2
                || (x2 == y2
                    && (x3 < y3
                        || (x3 == y3
                            && (x4 < y4))))));
}

} // namespace ML

#endif /* __utils__less_h__ */



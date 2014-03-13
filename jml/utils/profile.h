/* profile.h                                                       -*- C++ -*-
   Jeremy Barnes, 15 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.

   Profiling code.
*/

#ifndef __utils__profile_h__
#define __utils__profile_h__

#include <boost/timer.hpp>

namespace ML {

class Function_Profiler {
public:
    boost::timer * t;
    double & var;
    Function_Profiler(double & var, bool profile)
        : t(0), var(var)
    {
        if (profile) t = new boost::timer();
    }
    ~Function_Profiler()
    {
        if (t) var += t->elapsed();
        delete t;
    }
};

#define PROFILE_FUNCTION(var) \
Function_Profiler __profiler(var, profile);

} // namespace ML

#endif /* __utils__profile_h__ */

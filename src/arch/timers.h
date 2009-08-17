/* timers.h                                                        -*- C++ -*-
   Jeremy Barnes, 1 April 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Different types of timers.
*/

#ifndef __arch__timers_h__
#define __arch__timers_h__

#include <sys/time.h>
#include "tick_counter.h"
#include "format.h"
#include <string>
#include <cerrno>

namespace ML {

inline double cpu_time()
{
    clock_t clk = clock();
    return clk / 1000000.0;  // clocks per second
}

inline double wall_time()
{
    struct timeval tv;
    struct timezone tz;
    int res = gettimeofday(&tv, &tz);
    if (res != 0)
        throw Exception("gettimeofday() returned "
                        + std::string(strerror(errno)));

    return tv.tv_sec + (tv.tv_usec / 1000000.0);
}
 
struct Timer {
    double wall_, cpu_;
    unsigned long long ticks_;
    
    Timer()
    {
        restart();
    }
    
    void restart()
    {
        wall_ = wall_time();
        cpu_ = cpu_time();
        ticks_ = ticks();
    }

    std::string elapsed() const
    {
        return format("elapsed: [%.2fs cpu, %.4f mticks, %.2fs wall]",
                      cpu_time() - cpu_,
                      (ticks() - ticks_) / 1000000.0,
                      wall_time() - wall_);
    }
};

} // namespace ML

#endif

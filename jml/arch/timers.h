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
#include <string.h>
#include "exception.h"
#include <sys/select.h>
#include "atomic_ops.h"
#include "jml/math/xdiv.h"


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
    bool enabled;

    explicit Timer(bool enable = true)
        : enabled(enable)
    {
        restart();
    }
    
    void restart()
    {
        if (!enabled)
            return;
        wall_ = wall_time();
        cpu_ = cpu_time();
        ticks_ = ticks();
    }

    std::string elapsed() const
    {
        if (!enabled)
            return "disabled";
        return format("elapsed: [%.2fs cpu, %.4f mticks, %.2fs wall]",
                      cpu_time() - cpu_,
                      (ticks() - ticks_) / 1000000.0,
                      wall_time() - wall_);
    }

    double elapsed_cpu() const { return cpu_time() - cpu_; }
    double elapsed_ticks() const { return ticks() - ticks_; }
    double elapsed_wall() const { return wall_time() - wall_; }
};

inline int64_t timeDiff(const timeval & tv1, const timeval & tv2)
{
    return 1000000 * ((int64_t)tv2.tv_sec - (int64_t)tv1.tv_sec)
        + (int64_t)tv2.tv_usec - (int64_t)tv1.tv_usec;
}

inline void sleep(double sleepTime)
{
    long secs = sleepTime;
    long usec = (sleepTime - secs) * 1000000;
    struct timeval timeout = { secs, usec };
    for (;;) {
        if (timeout.tv_sec < 0 || timeout.tv_usec < 0) break;
        int res = select(0, 0, 0, 0, &timeout);
        if (res == -1 && errno == EINTR) continue;
        else if (res == -1)
            throw Exception("error sleeping: %s, secs %ld, usec %ld",
                            strerror(errno), timeout.tv_sec, timeout.tv_usec);
        else break;
    }
}


struct Duty_Cycle_Timer {

    struct Stats {
        uint64_t usAsleep, usAwake;
        uint64_t numWakeups;

        double duty_cycle() const
        {
            return xdiv<double>(usAwake, usAsleep + usAwake);
        }
    };

    enum Timer_Source {
        TS_TSC,   ///< Get from the timestamp counter (fast)
        TS_RTC    ///< Get from the real time clock (accurate)
    };
    
    Duty_Cycle_Timer(Timer_Source source = TS_TSC)
        : source(source)
    {
        clear();
        afterSleep = getTime();
    }

    void clear()
    {
        afterSleep = getTime();
        current.usAsleep = current.usAwake = current.numWakeups = 0;
    }

    void notifyBeforeSleep()
    {
        beforeSleep = getTime();
        uint64_t useconds = (beforeSleep - afterSleep) * 1000000;
#if 0
        using namespace std;
        cerr << "sleeping at " << beforeSleep << " after "
             << (beforeSleep - afterSleep)
             << " (" << useconds << "us)" << endl;
#endif
        atomic_add(current.usAwake, useconds);
    }

    void notifyAfterSleep()
    {
        afterSleep = getTime();
        uint64_t useconds = (afterSleep - beforeSleep) * 1000000;
#if 0
        using namespace std;
        cerr << "awake at " << beforeSleep << " after "
             << (afterSleep - beforeSleep)
             << " (" << useconds << "us)" << endl;
        cerr << "seconds_per_tick = " << seconds_per_tick << endl;
#endif
        atomic_add(current.usAsleep, useconds);
        atomic_add(current.numWakeups, 1);
    }

    double getTime()
    {
        if (source == TS_TSC)
            return ticks() * seconds_per_tick;
        else return wall_time();
    }

    Stats stats() const
    {
        return current;
    }
    
    Stats current;
    Timer_Source source;

    double beforeSleep, afterSleep;
};

} // namespace ML

#endif

/* profiler.h                                                      -*- C++ -*-
   Jeremy Barnes, 30 May 2012
   Profiler class for router.
*/

#ifndef __router__profiler_h__
#define __router__profiler_h__

#include "jml/arch/timers.h"
#include "jml/arch/exception.h"
#include "jml/arch/atomic_ops.h"

namespace RTBKIT {


/*****************************************************************************/
/* ROUTER PROFILER                                                           */
/*****************************************************************************/

inline double getProfilingTime()
{
    return ML::ticks() * ML::seconds_per_tick;
}

/** Microseconds between the two times, which are expressed in seconds. */
inline int64_t microsecondsBetween(double after, double before)
{
    if (after < before) {
        // This can happen on a CPU that doesn't have a monotonic
        // synchronous TSC over all CPUs
        using namespace std;
        cerr << "warning: microseconds between problem: after "
             << after << " before " << before << " distance "
             << after - before << endl;
        after = before;
#if 0
        throw ML::Exception("microseconds between problem: "
                            "after %f vs before %f",
                            after, before);
#endif
    }
    return (after - before) * 1000000;
}


struct RouterProfiler {

    RouterProfiler(uint64_t & counter)
        : counter(counter)
    {
        startTime = getProfilingTime();
    }

    ~RouterProfiler()
    {
        ML::atomic_add(counter,
                       microsecondsBetween(getProfilingTime(), startTime));
    }

    uint64_t & counter;
    double startTime;
};

} // namespace RTBKIT


#endif /* __router__profiler_h__ */

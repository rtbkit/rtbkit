/* tick_counter.cc
   Jeremy Barnes, 16 February 2007
   Copyright (c) 2007 Jeremy Barnes.  All rights reserved.
   
   Access to the processor's tick counter functions.
*/

#include "tick_counter.h"
#include "jml/math/xdiv.h"
#include <iostream>
#include <sys/time.h>

using namespace std;


namespace ML {

/** The average number of ticks of overhead for the tick counter. */
double ticks_overhead = -1.0;

/** The average number of ticks per second. */
double ticks_per_second = -1.0;

/** Number of seconds per tick */
double seconds_per_tick = -1.0;

namespace {

JML_ALWAYS_INLINE uint64_t fake_ticks()
{
    uint64_t result;
    asm volatile ("" : "=A" (result));
    return result;
}

} // file scope

double calc_ticks_overhead()
{
    static const unsigned ITERATIONS = 100;

    uint64_t before1 = ticks();

    for (unsigned i = 0;  i < ITERATIONS;  ++i)
        (void)ticks();

    uint64_t after1 = ticks();

    uint64_t before2 = ticks();

    for (unsigned i = 0;  i < ITERATIONS;  ++i)
        (void)fake_ticks();
    
    uint64_t after2 = ticks();

    uint64_t nt1 = after1 - before1, nt2 = after2 - before2;

    uint64_t overhead = nt1 - nt2;

    double result = xdiv<double>(overhead, ITERATIONS);

    //cerr << "nt1 = " << nt1 << " nt2 = " << nt2 << " overhead = "
    //     << overhead << " result = " << result << endl;

    return result;
}

namespace {

// TODO: use clock_gettime
double elapsed_since(const timeval & tv_start)
{
    struct timeval tv_end;
    gettimeofday(&tv_end, 0);

    double start_sec = tv_start.tv_sec + (tv_start.tv_usec / 1000000.0);
    double end_sec = tv_end.tv_sec + (tv_end.tv_usec / 1000000.0);

    return (end_sec - start_sec);
}

} // file scope

double calc_ticks_per_second(double to_elapse)
{
    sched_yield();

    struct timeval tv;
    gettimeofday(&tv, 0);
    
    size_t before = ticks();

    double elapsed = 0.0;

    while ((elapsed = elapsed_since(tv)) < to_elapse) ;

    size_t after = ticks();

    return (after - before) / elapsed;
}

namespace {

struct Init {
    Init()
    {
        ticks_overhead = calc_ticks_overhead();
        ticks_per_second = calc_ticks_per_second();
        seconds_per_tick = 1.0 / ticks_per_second;
    }

} init;

} // file scope

} // namespace ML


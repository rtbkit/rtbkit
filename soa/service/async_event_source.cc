/* async_event_source.cc
   Jeremy Barnes, 9 November 2012
*/

#include "async_event_source.h"
#include <sys/timerfd.h>
#include "jml/arch/exception.h"
#include "jml/arch/futex.h"
#include <iostream>
#include "message_loop.h"


using namespace std;

namespace Datacratic {

/*****************************************************************************/
/* ASYNC EVENT SOURCE                                                        */
/*****************************************************************************/

void
AsyncEventSource::
disconnect()
{
    if (!parent_)
        return;
    parent_->removeSource(this);
    parent_ = nullptr;
}

void
AsyncEventSource::
waitConnectionState(int state)
    const
{
    while (connectionState_ != state) {
        int oldVal = connectionState_;
        ML::futex_wait(connectionState_, oldVal);
    }
}

/*****************************************************************************/
/* PERIODIC EVENT SOURCE                                                     */
/*****************************************************************************/

PeriodicEventSource::
PeriodicEventSource()
    : timerFd(-1),
      timePeriodSeconds(0),
      singleThreaded_(true)
      
{
}

PeriodicEventSource::
PeriodicEventSource(double timePeriodSeconds,
                    std::function<void (uint64_t)> onTimeout,
                    bool singleThreaded)
    : timerFd(-1),
      timePeriodSeconds(timePeriodSeconds),
      onTimeout(onTimeout),
      singleThreaded_(singleThreaded)
{
    init(timePeriodSeconds, onTimeout, singleThreaded);
}

void
PeriodicEventSource::
init(double timePeriodSeconds,
     std::function<void (uint64_t)> onTimeout,
     bool singleThreaded)
{
    if (timerFd != -1)
        throw ML::Exception("double initialization of periodic event source");

    this->timePeriodSeconds = timePeriodSeconds;
    this->onTimeout = onTimeout;
    this->singleThreaded_ = singleThreaded;

    timerFd = timerfd_create(CLOCK_MONOTONIC, TFD_NONBLOCK);
    if (timerFd == -1)
        throw ML::Exception(errno, "timerfd_create");

    itimerspec spec;
    
    int res = clock_gettime(CLOCK_MONOTONIC, &spec.it_value);
    if (res == -1)
        throw ML::Exception(errno, "clock_gettime");
    uint64_t seconds, nanoseconds;
    seconds = timePeriodSeconds;
    nanoseconds = (timePeriodSeconds - seconds) * 1000000000;

    spec.it_interval.tv_sec = spec.it_value.tv_sec = seconds;
    spec.it_interval.tv_nsec = spec.it_value.tv_nsec = nanoseconds;

#if 0    
    // Relative to current time, so zero
    spec.it_value.tv_sec = spec.it_value.tv_nsec = 0;

    
    spec.it_value.tv_nsec += nanoseconds;
    if (spec.it_value.tv_nsec >= 1000000000) {
        ++spec.it_value.tv_sec;
        spec.it_value.tv_nsec -= 1000000000;
    }
    spec.it_value.tv_sec += seconds;
#endif

    res = timerfd_settime(timerFd, 0, &spec, 0);
    if (res == -1)
        throw ML::Exception(errno, "timerfd_settime");
}

PeriodicEventSource::
~PeriodicEventSource()
{
    int res = close(timerFd);
    if (res == -1)
        cerr << "warning: close on timerfd: " << strerror(errno) << endl;
}

int
PeriodicEventSource::
selectFd() const
{
    return timerFd;
}

bool
PeriodicEventSource::
processOne()
{
    uint64_t numWakeups = 0;
    for (;;) {
        int res = read(timerFd, &numWakeups, 8);
        if (res == -1 && errno == EINTR) continue;
        if (res == -1 && (errno == EAGAIN || errno == EWOULDBLOCK))
            break;
        if (res == -1)
            throw ML::Exception(errno, "timerfd read");
        else if (res != 8)
            throw ML::Exception("timerfd read: wrong number of bytes: %d",
                                res);
        onTimeout(numWakeups);
        break;
    }
    return false;
}


} // namespace Datacratic

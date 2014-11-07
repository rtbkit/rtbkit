/* epoller.cc
   Jeremy Barnes, 26 September 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

*/

#include "soa/service/epoller.h"

#include <sys/epoll.h>
#include <poll.h>
#include "jml/arch/exception.h"
#include "jml/arch/backtrace.h"
#include <string.h>
#include <iostream>
#include "soa/types/date.h"

using namespace std;
using namespace ML;

namespace Datacratic {

// Maximum number of events that we can handle
static constexpr int MaxEvents = 1024;


/*****************************************************************************/
/* EPOLLER                                                                   */
/*****************************************************************************/

Epoller::
Epoller()
    : epoll_fd(-1), timeout_(0), numFds_(0)
{
}

Epoller::
~Epoller()
{
    close();
}

void
Epoller::
init(int maxFds, int timeout)
{
    //cerr << "initializing epoller at " << this << endl;
    //backtrace();
    close();

    epoll_fd = epoll_create(maxFds);
    if (epoll_fd == -1)
        throw ML::Exception(errno, "EndpointBase epoll_create()");

    timeout_ = timeout;
}

void
Epoller::
close()
{
    if (epoll_fd < 0)
        return;
    //cerr << "closing epoller at " << this << endl;
    //backtrace();
    ::close(epoll_fd);
    epoll_fd = -2;
}

void
Epoller::
removeFd(int fd)
{
    //cerr << Date::now().print(4) << "removed " << fd << endl;

    int res = epoll_ctl(epoll_fd, EPOLL_CTL_DEL, fd, 0);
    
    if (res == -1) {
        if (errno != EBADF)
            throw ML::Exception("epoll_ctl DEL fd %d: %s", fd,
                                strerror(errno));
    }

    if (numFds_ > 0) {
        numFds_--;
    }
    else {
        throw ML::Exception("too many file descriptors removed");
    }
}

int
Epoller::
handleEvents(int usToWait, int nEvents,
             const HandleEvent & handleEvent_,
             const OnEvent & beforeSleep_,
             const OnEvent & afterSleep_)
{
    if (nEvents == -1) {
        nEvents = std::max<int>(numFds_, 1);
    }

    const HandleEvent & handleEvent
        = handleEvent_ ? handleEvent_ : this->handleEvent;
    const OnEvent & beforeSleep
        = beforeSleep_ ? beforeSleep_ : this->beforeSleep;
    const OnEvent & afterSleep
        = afterSleep_ ? afterSleep_ : this->afterSleep;

    if (nEvents <= 0)
        throw ML::Exception("can't wait for no events");

    if (nEvents > MaxEvents)
        nEvents = MaxEvents;

    for (;;) {
        epoll_event events[nEvents];
                
        if (beforeSleep)
            beforeSleep();

        // Do the sleep with nanosecond resolution
        // Let's hope it doesn't busy-wait
        if (usToWait != 0) {
            pollfd fd[1] = { { epoll_fd, POLLIN, 0 } };
            timespec timeout = { 0, usToWait * 1000 };
            int res = ppoll(fd, 1, &timeout, 0);
            if (res == -1 && errno == EBADF) {
                cerr << "got bad FD on sleep" << endl;
                return -1;
            }
            if (res == -1 && errno == EINTR)
                continue;
            //if (debug) cerr << "handleEvents: res = " << res << endl;
            if (res == 0) return 0;
        }

        int res = epoll_wait(epoll_fd, events, nEvents, timeout_);

        if (afterSleep)
            afterSleep();

        // sys call interrupt
        if (res == -1 && errno == EINTR) continue;
        if (res == -1 && errno == EBADF) {
            cerr << "got bad FD" << endl;
            return -1;
        }
        if (res == 0) return 0;
        
        if (res == -1) {
            //cerr << "epoll_fd = " << epoll_fd << endl;
            //cerr << "timeout_ = " << timeout_ << endl;
            //cerr << "nEvents = " << nEvents << endl;
            throw Exception(errno, "epoll_wait");
        }
        nEvents = res;
        
        for (unsigned i = 0;  i < nEvents;  ++i) {
            if (handleEvent(events[i]) == SHUTDOWN) return -1;
        }
                
        return nEvents;
    }
}

bool
Epoller::
poll() const
{
    for (;;) {
        pollfd fds[1] = { { epoll_fd, POLLIN, 0 } };
        int res = ::poll(fds, 1, 0);

        //cerr << "poll res = " << res << endl;

        if (res == -1 && errno == EBADF)
            return false;
        if (res == -1 && errno == EINTR)
            continue;
        if (res == -1)
            throw ML::Exception("ppoll in Epoller::poll");

        return res > 0;
    }
}

void
Epoller::
performAddFd(int fd, void * data, bool oneshot, bool restart)
{
    // cerr << (Date::now().print(4)
    //          + " performAddFd: epoll_fd=" + to_string(epoll_fd)
    //          + " fd=" + to_string(fd)
    //          + " one-shot=" + to_string(oneshot)
    //          + " restart=" + to_string(restart)
    //          + "\n");

    struct epoll_event event;
    event.events = EPOLLIN;
    if (oneshot) {
        event.events |= EPOLLONESHOT;
    }
    event.data.ptr = data;

    if (!restart) {
        numFds_++;
    }

    int action = restart ? EPOLL_CTL_MOD : EPOLL_CTL_ADD;
    int res = epoll_ctl(epoll_fd, action, fd, &event);

    if (res == -1)
        throw ML::Exception("epoll_ctl: %s (fd=%d, epollfd=%d, oneshot=%d,"
                            " restart=%d)",
                            strerror(errno), fd, epoll_fd, oneshot, restart);
}

bool
Epoller::
processOne()
{
    int res = handleEvents();
    //cerr << "processOne res = " << res << endl;
    if (res == -1) return false;  // wakeup for shutdown
    return poll();
}

} // namespace Datacratic

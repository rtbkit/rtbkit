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


/*****************************************************************************/
/* EPOLLER                                                                   */
/*****************************************************************************/

Epoller::
Epoller()
    : epoll_fd(-1)
{
}

Epoller::
~Epoller()
{
    close();
}

void
Epoller::
init(int maxFds)
{
    //cerr << "initializing epoller at " << this << endl;
    //backtrace();
    close();

    epoll_fd = epoll_create(maxFds);
    if (epoll_fd == -1)
        throw ML::Exception(errno, "EndpointBase epoll_create()");
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
addFd(int fd, void * data)
{
    //cerr << Date::now().print(4) << "added " << fd << " multiple shot" << endl;

    struct epoll_event event;
    event.events = EPOLLIN;
    event.data.ptr = data;
    
    int res = epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &event);
    
    if (res == -1)
        throw ML::Exception(errno, "epoll_ctl ADD");
}
    
void
Epoller::
addFdOneShot(int fd, void * data)
{
    //cerr << Date::now().print(4) << "added " << fd << " one-shot" << endl;

    struct epoll_event event;
    event.events = EPOLLIN | EPOLLONESHOT;
    event.data.ptr = data;
    
    int res = epoll_ctl(epoll_fd, EPOLL_CTL_ADD, fd, &event);
        
    if (res == -1)
        throw ML::Exception("epoll_ctl ADD: %s (fd = %d, epollfd = %d)",
                            strerror(errno), fd, epoll_fd);
}

void
Epoller::
restartFdOneShot(int fd, void * data)
{
    //cerr << Date::now().print(4) << "restarted " << fd << " one-shot" << endl;

    struct epoll_event event;
    event.events = EPOLLIN | EPOLLONESHOT;
    event.data.ptr = data;
    
    int res = epoll_ctl(epoll_fd, EPOLL_CTL_MOD, fd, &event);
    
    if (res == -1)
        throw ML::Exception(errno, "epoll_ctl MOD");
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
}

int
Epoller::
handleEvents(int usToWait, int nEvents,
             const HandleEvent & handleEvent_,
             const OnEvent & beforeSleep_,
             const OnEvent & afterSleep_)
{
    const HandleEvent & handleEvent
        = handleEvent_ ? handleEvent_ : this->handleEvent;
    const OnEvent & beforeSleep
        = beforeSleep_ ? beforeSleep_ : this->beforeSleep;
    const OnEvent & afterSleep
        = afterSleep_ ? afterSleep_ : this->afterSleep;

    if (nEvents > 1024)
        throw ML::Exception("waiting for too many events will overflow the stack");

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

        int res = epoll_wait(epoll_fd, events, nEvents, 0);

        if (afterSleep)
            afterSleep();

        // sys call interrupt
        if (res == -1 && errno == EINTR) continue;
        if (res == -1 && errno == EBADF) {
            cerr << "got bad FD" << endl;
            return -1;
        }
        if (res == 0) return 0;

        if (res == -1)
            throw Exception(errno, "epoll_wait");
        nEvents = res;

        for (unsigned i = 0;  i < nEvents;  ++i) {
            if (handleEvent(events[i])) return -1;
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

/* epoll_loop.cc
   Wolfgang Sourdeau, 25 February 2015
   Copyright (c) 2015 Datacratic.  All rights reserved.

   An alternative event loop to Epoller.
*/

#include <string>

#include "jml/utils/exc_assert.h"
#include "epoll_loop.h"

using namespace std;
using namespace Datacratic;


EpollLoop::
EpollLoop(const OnException & onException)
    : AsyncEventSource(),
      epollFd_(-1),
      numFds_(0),
      onException_(onException)
{
    epollFd_ = ::epoll_create(666);
    if (epollFd_ == -1)
        throw ML::Exception(errno, "epoll_create");
}

EpollLoop:: 
~EpollLoop()
{
    closeEpollFd();
}
     
bool
EpollLoop::
processOne()
{
    loop(-1, 0);

    return false;
}

void
EpollLoop::
loop(int maxEvents, int timeout)
{
    ExcAssert(maxEvents != 0);

    if (numFds_ > 0) {
        if (maxEvents == -1) {
            maxEvents = numFds_;
        }
        struct epoll_event events[maxEvents];

        try {
            int res;
            while (true) {
                res = epoll_wait(epollFd_, events, maxEvents, timeout);
                if (res == -1) {
                    if (errno == EINTR) {
                        continue;
                    }
                    throw ML::Exception(errno, "epoll_wait");
                }
                break;
            }

            for (int i = 0; i < res; i++) {
                auto * fn = static_cast<EpollCallback *>(events[i].data.ptr);
                ExcAssert(fn != nullptr);
                (*fn)(events[i]);
            }

            map<int, OnUnregistered> delayedUnregistrations;
            {
                std::unique_lock<mutex> guard(callbackLock_);
                delayedUnregistrations = move(delayedUnregistrations_);
                delayedUnregistrations_.clear();
            }
            for (const auto & unreg: delayedUnregistrations) {
                unregisterFdCallback(unreg.first, false, unreg.second);
            }
        }
        catch (const std::exception & exc) {
            handleException();
        }
    }
}

void
EpollLoop::
closeEpollFd()
{
    if (epollFd_ != -1) {
        ::close(epollFd_);
        epollFd_ = -1;
    }
}

void
EpollLoop::
performAddFd(int fd, bool readerFd, bool writerFd, bool modify, bool oneshot)
{
    if (epollFd_ == -1)
        return;
    ExcAssert(fd > -1);

    struct epoll_event event;
    if (oneshot) {
        event.events = EPOLLONESHOT;
    }
    else {
        event.events = 0;
    }
    if (readerFd) {
        event.events |= EPOLLIN;
    }
    if (writerFd) {
        event.events |= EPOLLOUT;
    }

    EpollCallback & cb = fdCallbacks_.at(fd);
    event.data.ptr = &cb;

    int operation = modify ? EPOLL_CTL_MOD : EPOLL_CTL_ADD;

    int res = epoll_ctl(epollFd_, operation, fd, &event);
    if (res == -1) {
        string message = (string("epoll_ctl:")
                          + " modify=" + to_string(modify)
                          + " fd=" + to_string(fd)
                          + " readerFd=" + to_string(readerFd)
                          + " writerFd=" + to_string(writerFd));
        throw ML::Exception(errno, message);
    }

    if (!modify) {
        numFds_++;
    }
}

void
EpollLoop::
removeFd(int fd, bool unregisterCallback)
{
    if (epollFd_ == -1)
        return;
    ExcAssert(fd > -1);

    int res = epoll_ctl(epollFd_, EPOLL_CTL_DEL, fd, 0);
    if (res == -1) {
        throw ML::Exception(errno, "epoll_ctl DEL " + to_string(fd));
    }
    if (numFds_ == 0) {
        throw ML::Exception("inconsistent number of fds registered");
    }
    numFds_--;

    if (unregisterCallback) {
        unregisterFdCallback(fd, true);
    }
}

void
EpollLoop::
registerFdCallback(int fd, const EpollCallback & cb)
{
    std::unique_lock<mutex> guard(callbackLock_);
    if (delayedUnregistrations_.count(fd) == 0) {
        if (fdCallbacks_.find(fd) != fdCallbacks_.end()) {
            throw ML::Exception("callback already registered for fd");
        }
    }
    else {
        delayedUnregistrations_.erase(fd);
    }
    fdCallbacks_[fd] = cb;
}

void
EpollLoop::
unregisterFdCallback(int fd, bool delayed,
                     const OnUnregistered & onUnregistered)
{
    std::unique_lock<mutex> guard(callbackLock_);
    if (fdCallbacks_.find(fd) == fdCallbacks_.end()) {
        throw ML::Exception("callback not registered for fd");
    }
    if (delayed) {
        ExcAssert(delayedUnregistrations_.count(fd) == 0);
        delayedUnregistrations_[fd] = onUnregistered;
    }
    else {
        delayedUnregistrations_.erase(fd);
        fdCallbacks_.erase(fd);
        if (onUnregistered) {
            onUnregistered();
        }
    }
}

void
EpollLoop::
handleException()
{
    onException(current_exception());
}

void
EpollLoop::
onException(const exception_ptr & excPtr)
{
    if (onException_) {
        onException_(excPtr);
    }
    else {
        rethrow_exception(excPtr);
    }
}

/* epoll_loop.h                                                    -*- C++ -*-
   Wolfgang Sourdeau, 25 February 2015
   Copyright (c) 2015 Datacratic.  All rights reserved.

   An alternative event loop to Epoller.

   Although not a complete replacement, this class provides most of the
   functionality of Epoller, whilst providing a simpler event handling
   mechanism and enabling write events on registered file descriptors. Both
   should ultimately be merged.
*/

#pragma once

#include <sys/epoll.h>

#include <exception>
#include <map>
#include <functional>

#include "soa/service/async_event_source.h"


namespace Datacratic {

/****************************************************************************/
/* EPOLL LOOP                                                               */
/****************************************************************************/

/* This class provides an alternative to Epoller, where file file descriptors
 * can be registered for reading, writing and where callbacks are associated to file
 * descriptors. It is mostly useful for compound classes making use of
 * multiple file descriptors.
*/

struct EpollLoop : public AsyncEventSource
{
    /* Type of callback invoked whenever an uncaught exception occurs. */
    typedef std::function<void(const std::exception_ptr &)> OnException;

    /* Type of callback invoked when a callback has been unregistered. */
    typedef std::function<void()> OnUnregistered;

    /* Type of callback invoked whenever an epoll event is reported for a file
     * descriptor. */
    typedef std::function<void (const ::epoll_event &)> EpollCallback;

    EpollLoop(const OnException & onException);
    ~EpollLoop();

    /* AsyncEventSource interface */
    virtual int selectFd() const
    { return epollFd_; }

    virtual bool processOne();

    /* Perform a single loop, where "maxEvents" is the maximum numbers of
     * events to handle (-1 for unlimited), and "timeout" the value passed as
     * timeout to epoll_wait */
    void loop(int maxEvents, int timeout);

    /* Register a file descriptor into the internal epoll queue for reading
       and/or writing. If "callback" is specified and not null, it will be
       registered for the given file descriptor. In any case the correponding
       callback *must* be registered beforehand. */
    void addFd(int fd, bool readerFd, bool writerFd,
               const EpollCallback & callback = nullptr)
    {
        if (callback) {
            registerFdCallback(fd, callback);
        }
        performAddFd(fd, readerFd, writerFd, false, false);
    }

    /* Same as addFd, with the EPOLLONESHOT flag. */
    void addFdOneShot(int fd, bool readerFd, bool writerFd,
                      const EpollCallback & callback = nullptr)
    {
        if (callback) {
            registerFdCallback(fd, callback);
        }
        performAddFd(fd, readerFd, writerFd, false, true);
    }

    /* Modify a file descriptor in the epoll queue. */
    void modifyFd(int fd, bool readerFd, bool writerFd)
    { performAddFd(fd, readerFd, writerFd, true, false); }

    /* Same as modifyFd, with the EPOLLONESHOT flag. */
    void modifyFdOneShot(int fd, bool readerFd, bool writerFd)
    { performAddFd(fd, readerFd, writerFd, true, true); }

    /* Remove a file descriptor from the internal epoll queue. If
     * "unregisterCallback" is specified, "unregisterFdCallback" will be
     * specified on the given fd, in delayed mode. */
    void removeFd(int fd, bool unregisterCallback = false);

    /* Associate a callback with a file descriptor for future epoll
       operations. */
    void registerFdCallback(int fd, const EpollCallback & cb);

    /* Dissociate a callback and a file descriptor from the callback registry,
       with the "delayed" parameter indicating whether the operation must
       occur immediately or at the end of the epoll loop. */
    void unregisterFdCallback(int fd, bool delayed,
                              const OnUnregistered & onUnregistered
                              = nullptr);

    /* Function invoked when an exception occurs during the handling of
     * events, rethrowing the exception by default. */
    virtual void onException(const std::exception_ptr & excPtr);

private:
    void performAddFd(int fd, bool readerFd, bool writerFd,
                      bool modify, bool oneshot);

    /* epoll operations */
    void closeEpollFd();

    void handleException();

    int epollFd_;
    size_t numFds_;

    std::map<int, EpollCallback> fdCallbacks_;
    std::map<int, OnUnregistered> delayedUnregistrations_;

    OnException onException_;
};

} // namespace Datacratic

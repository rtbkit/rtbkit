/* epoller.h                                                       -*- C++ -*-
   Jeremy Barnes, 26 September 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Structure do allow multiplexing of FDs based upon epoll.
*/

#ifndef __endpoint__epoller_h__
#define __endpoint__epoller_h__

#include <functional>
#include "soa/service/async_event_source.h"

struct epoll_event;

namespace Datacratic {

/*****************************************************************************/
/* EPOLLER                                                                   */
/*****************************************************************************/

/** Basic wrapper around the epoll interface to turn it into an async event
    source.
*/

struct Epoller: public AsyncEventSource {

    Epoller();

    ~Epoller();

    void init(int maxFds, int timeout = 0);

    void close();

    /** Set the timeout value used when calling epoll_wait. */
    void setPollTimeout(int newTimeout)
    {
        timeout_ = newTimeout;
    }

    /** Add the given fd to multiplex fd.  It will repeatedly wake up the
        loop without being restarted.
    */
    void addFd(int fd, void * data = 0)
    {
        performAddFd(fd, data, false, false);
    }
    
    /** Add the given fd to wake up one a one-shot basis.  It will need to
        be restarted once the event is handled.
    */
    void addFdOneShot(int fd, void * data = 0)
    {
        performAddFd(fd, data, true, false);
    }

    /** Restart a woken up one-shot fd. */
    void restartFdOneShot(int fd, void * data = 0)
    {
        performAddFd(fd, data, true, true);
    }

    /** Remove the given fd from the multiplexer set. */
    void removeFd(int fd);
    
    enum HandleEventResult {
        DONE,
        SHUTDOWN
    };

    typedef std::function<HandleEventResult (epoll_event & event)> HandleEvent;
    typedef std::function<void ()> OnEvent;

    /** Default event handler function to use. */
    HandleEvent handleEvent;

    OnEvent beforeSleep;
    OnEvent afterSleep;

    /** Wait up to the given number of microseconds and handle up to
        the given number of events.
        
        The handleEvent function should return true if the loop should exit
        immediately, or false if there are other events to continue.
        
        Returns the number of events handled or -1 if a handler forced the
        event handler to exit.

    */

    int handleEvents(int usToWait = 0, int nEvents = -1,
                     const HandleEvent & handleEvent = HandleEvent(),
                     const OnEvent & beforeSleep = OnEvent(),
                     const OnEvent & afterSleep = OnEvent());

    virtual int selectFd() const
    {
        return epoll_fd;
    }

    virtual bool poll() const;

    virtual bool processOne();
    
private:
    /* Perform the fd addition and modification */
    void performAddFd(int fd, void * data, bool oneShot, bool restart);

    /* Fd for the epoll mechanism. */
    int epoll_fd;

    /* Timeout value to use for epoll_wait */
    int timeout_;

    /* Number of registered file descriptors */
    size_t numFds_;
};

} // namespace Datacratic

#endif /* __endpoint__epoller_h__ */

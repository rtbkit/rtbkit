/* async_event_source.h                                            -*- C++ -*-
   Jeremy Barnes, 9 November 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Source of asynchronous events; a bit like a reactor.
*/

#ifndef __service__async_event_source_h__
#define __service__async_event_source_h__


#include <boost/thread/thread.hpp>
#include <functional>
#include "jml/arch/exception.h"
#include <thread>

namespace Datacratic {

struct MessageLoop;

/*****************************************************************************/
/* ASYNC EVENT SOURCE                                                        */
/*****************************************************************************/

struct AsyncEventSource {
    constexpr static int DISCONNECTED = 0;
    constexpr static int CONNECTED = 1;

    AsyncEventSource()
        : needsPoll(false), debug_(false), parent_(0), connectionState_(DISCONNECTED)
    {
    }

    AsyncEventSource(AsyncEventSource && other)
        : needsPoll(other.needsPoll), debug_(other.debug_), parent_(nullptr), connectionState_(other.connectionState_)
    {
        if (other.parent_ != nullptr) {
            fprintf(stderr,
                    "AsyncEventSource(&&): moved instance is attached to a MessageLoop\n");
            abort();
        }
    }

    AsyncEventSource & operator = (const AsyncEventSource & other)
        noexcept
    {
        if (other.parent_ != nullptr) {
            fprintf(stderr, "AsyncEventSource::=(const&): "
                    "copied instance is attached to a MessageLoop\n");
            abort();
        }
        needsPoll = other.needsPoll;
        debug_ = other.debug_;

        return *this;
    }

    virtual ~AsyncEventSource()
    {
        // disconnect(); calling this is evil because it better be already removed from the message loop
    }

    /** Return the file descriptor on which one should select() for messages
        from this source.  The source should organize itself such that if
        the fd indicates ready for a read, there is something to do.

        Should return -1 if it requires polling.

        Should never block.
    */
    virtual int selectFd() const
    {
        return -1;
    }

    /** Returns true if there is work to be done.  May be called from more
        than one thread.  Should never block.
    */
    virtual bool poll() const
    {
        return false;
    }

    /** Process a single message and return true if there are more to be
        processed.

        This may be called from more than one thread if singleThreaded()
        is false.
    */
    virtual bool processOne() = 0;

    /** Return whether the callbacks need to be called from a single thread
        or not.
    */
    virtual bool singleThreaded() const
    {
        return true;
    }

    /** Sets whether or not it is in debug mode. */
    virtual void debug(bool debugOn)
    {
        debug_ = debugOn;
    }

    /** Notify that the given message loop is responsible. */
    virtual void connected(MessageLoop * parent)
    {
        if (parent_)
            throw ML::Exception("attempt to connect AsyncEventSource "
                                "to two parents");
        parent_ = parent;
    }

    /** Disconnect from the parent message loop. */
    void disconnect();

    /** Blocks until the connection state changes to the specified value. */
    void waitConnectionState(int state) const;

    /** Sets whether or not this source requires polling periodically (as
        the selectFd may not include all events).
    */
    bool needsPoll;

    /** Sets whether this event handler is being debugged. */
    bool debug_;

    /** The parent message loop. */
    MessageLoop * parent_;

    /** The connection state to the message loop. */
    int connectionState_;
};


/*****************************************************************************/
/* MULTIPLE EVENT SOURCE                                                     */
/*****************************************************************************/

/** An async event source that can deal with multiple events. */

// ...

/*****************************************************************************/
/* PERIODIC EVENT SOURCE                                                     */
/*****************************************************************************/

struct PeriodicEventSource : public AsyncEventSource {
    PeriodicEventSource();

    PeriodicEventSource(double timePeriodSeconds,
                        std::function<void (uint64_t)> onTimeout,
                        bool singleThreaded = true);

    ~PeriodicEventSource();

    void init(double timePeriodSeconds,
              std::function<void (uint64_t)> onTimeout,
              bool singleThreaded = true);

    virtual int selectFd() const;

    virtual bool processOne();

    virtual bool singleThreaded() const
    {
        return singleThreaded_;
    }

private:    
    int timerFd;
    double timePeriodSeconds;
    std::function<void (uint64_t)> onTimeout;
    bool singleThreaded_;
};


#if 0
/*****************************************************************************/
/* WAKEUP EVENT SOURCE                                                       */
/*****************************************************************************/

struct WakeupEventSource : public AsyncEventSource {
    WakeupEventSource();

    virtual int selectFd() const;

    /** Process a single message and return true if there are more to be
        processed.

        This may be called from more than one thread.
    */
    virtual bool processOne();

private:    
    ML::Wakeup_Fd wakeup;
};
#endif

} // namespace Datacratic


#endif /* __service__async_event_source_h__ */

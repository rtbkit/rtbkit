/* connection_handler.h                                            -*- C++ -*-
   Jeremy Barnes, 27 February 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Connection handler logic.
*/

#ifndef __rtb__connection_handler_h__
#define __rtb__connection_handler_h__

#include "transport.h"
#include <iostream>
#include <boost/function.hpp>
#include <list>
#include "jml/arch/format.h"
#include "jml/arch/demangle.h"
#include "jml/arch/atomic_ops.h"

namespace Datacratic {


/*****************************************************************************/
/* CONNECTION HANDLER                                                        */
/*****************************************************************************/

struct ConnectionHandler {

    static uint32_t created, destroyed;

    ConnectionHandler()
        : transport_(0), magic(0x1234)
    {
        ML::atomic_add(created, 1);
    }

    virtual ~ConnectionHandler()
    {
        ML::atomic_add(destroyed, 1);

        if (magic != 0x1234)
            throw ML::Exception("Attempt to double free connection handler");
        magic = 0;
    }

    /** Function called when we ge the transport. */
    virtual void onGotTransport()
    {
    }

    /** Function called when we're dissociating the connection. */
    virtual void onDisassociate()
    {
    }

    /** Function called when a handler throws an exception. */
    virtual void onHandlerException(const std::string & handler,
                                    const std::exception & exc)
    {
        //using namespace std;
        //cerr << "handler " << handler << " threw exception "
        //     << exc.what() << endl;
        doError("handler " + handler + " had exception " + exc.what());
    }

    /** Function called out to to clean up before we finish for whatever
        reason.
    */
    virtual void onCleanup()
    {
    }

    /** Function called out to when we got an error from the socket. */
    virtual void handleError(const std::string & message)
    {
        doError(message);
        closeWhenHandlerFinished();
    }

    /** Function called when there is a disconnection when reading. */
    virtual void handleDisconnect()
    {
        closeWhenHandlerFinished();
    }

    virtual std::string status() const
    {
        return ML::format("%p of type %s", this, ML::type_name(*this).c_str());
    }

    /** Thing to call when we have an error */
    virtual void doError(const std::string & error) = 0;

    /** Close the connection. */
    void closeConnection();

    /** Pass on a send request to the transport. */
    ssize_t send(const char * buf, size_t len, int flags)
    {
        return transport().send(buf, len, flags);
    }

    /** Pass on a recv request to the transport. */
    ssize_t recv(char * buf, size_t buf_size, int flags)
    {
        return transport().recv(buf, buf_size, flags);
    }

    int getHandle() const
    {
        return transport().getHandle();
    }

    /** React to read events */
    void startReading();

    /** Stop reacting to read events */
    void stopReading();

    /** React to read events */
    void startWriting();

    /** Stop reacting to read events */
    void stopWriting();

    /** Schedule a timeout at the given absolute time.  Only one timer is
        available per connection. */
    void scheduleTimerAbsolute(Date timeout,
                               size_t cookie = 0,
                               void (*freecookie) (size_t) = 0)
    {
        transport().scheduleTimerAbsolute(timeout, cookie);
    }

    /** Schedule a timeout at the given number of seconds from now.  Again,
        only one timer is available per connection.
    */
    void scheduleTimerRelative(double secondsFromNow,
                               size_t cookie = 0,
                               void (*freecookie) (size_t) = 0)
    {
        transport().scheduleTimerRelative(secondsFromNow, cookie);
    }
    
    /** Cancel the timer for this connection if it exists. */
    void cancelTimer()
    {
        transport().cancelTimer();
    }

    void closeWhenHandlerFinished()
    {
        transport().closeWhenHandlerFinished();
    }

    void recycleWhenHandlerFinished()
    {
        transport().recycleWhenHandlerFinished();
    }

    /* Event callbacks.  The default implementations throw an exception.
    */
    virtual void handleInput();
    virtual void handleOutput();
    virtual void handlePeerShutdown();
    virtual void handleTimeout(Date time, size_t cookie);

    bool hasTransport() const
    {
        return transport_;
    }

    TransportBase & transport()
    {
        if (!transport_)
            throw ML::Exception("connection asked for transport with none "
                                "set");
        return *transport_;
    }

    const TransportBase & transport() const
    {
        if (!transport_)
            throw ML::Exception("connection asked for transport with none "
                                "set");
        return *transport_;
    }

    EndpointBase * get_endpoint() { return transport().get_endpoint(); }

    /** What should the return code be for the handler?  Should be zero if
        there isn't a fundamental protocol error, -1 if the connection
        should be closed (normally due to an error) and 1 if more events
        should be handled.
    */
    //virtual int handlerReturnCode() const = 0;

    /** Add an activity to the stream of activities for debugging. */
    void addActivity(const std::string & activity);

    /** Add an activity to the stream of activities for debugging. */
    void addActivityS(const char * activity);

    void addActivity(const char * fmt, ...);

    void checkMagic() const;

    /** Run the given function from a worker thread in the context of this
        handler.
    */
    void doAsync(const boost::function<void ()> & callback,
                 const char * name)
    {
        transport().doAsync(callback, name);
    }

private:
    void setTransport(TransportBase * transport);
    TransportBase * transport_;
    friend class TransportBase;

    /** Pass on a close peer request. */
    int closePeer()
    {
        return transport().closePeer();
    }

    int magic;
};


/*****************************************************************************/
/* PASSIVE CONNECTION HANDLER                                                */
/*****************************************************************************/

struct PassiveConnectionHandler: public ConnectionHandler {

    PassiveConnectionHandler()
        : inSend(false)
    {
    }

    std::string error;

    int done;
    bool inSend;

    /** Action to perform once we've finished sending. */
    enum NextAction {
        NEXT_CLOSE,
        NEXT_RECYCLE,
        NEXT_CONTINUE
    };

    typedef boost::function<void ()> OnWriteFinished;

    struct WriteEntry {
        Date date;
        std::string data;
        OnWriteFinished onWriteFinished;
        NextAction next;
    };

    std::list<WriteEntry> toWrite;
    
    /** Send some data, with the given set of actions to be done once it's
        finished.
    */
    void send(const std::string & str,
              NextAction action = NEXT_CONTINUE,
              OnWriteFinished onWriteFinished = OnWriteFinished());
    
    /** Function called out to when we got some data */
    virtual void handleData(const std::string & data) = 0;
    
    /** Function called out to when we got an error from the socket. */
    virtual void handleError(const std::string & message) = 0;

    /** Thing to call when we have an error internally.  This may be
        called from within another handler.
    */
    virtual void doError(const std::string & error);

    /* ACE_Event_Handler callbacks */
    virtual void handleInput();
    virtual void handleOutput();
    virtual void handleTimeout(Date time, size_t cookie);

    friend class TransportBase;
};

} // namespace Datacratic

#endif /* __rtb__connection_handler_h__ */

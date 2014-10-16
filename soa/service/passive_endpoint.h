/* passive_endpoint.h                                              -*- C++ -*-
   Jeremy Barnes, 29 April 2011
   Copyright (C) 2011 Datacratic.  All rights reserved.

   Base class for a passive endpoint.
*/

#pragma once

#include "soa/service/endpoint.h"
#include "soa/service/port_range_service.h"
#include "jml/arch/wakeup_fd.h"

namespace Datacratic {

enum {
    DEF_BACKLOG = 128
};

/*****************************************************************************/
/* ACCEPTOR                                                                  */
/*****************************************************************************/

struct Acceptor {
    virtual ~Acceptor()
    {
    }

    virtual int
    listen(PortRange const & portRange, const std::string & hostname,
           PassiveEndpoint * endpoint, bool nameLookup, int backlog) = 0;

    virtual void closePeer() = 0;

    /** What host are we connected to? */
    virtual std::string hostname() const = 0;

    /** What port are we listening on? */
    virtual int port() const = 0;

    /** Wait until we are ready to accept connections */
    virtual void waitListening() const = 0;
};


/*****************************************************************************/
/* PASSIVE ENDPOINT                                                          */
/*****************************************************************************/

/** An endpoint that listens for a connection that comes in on a port, before
    passing it off to all the standard stuff.
*/

struct PassiveEndpoint: public EndpointBase {

    PassiveEndpoint(const std::string & name);

    virtual ~PassiveEndpoint();

    /** Initialize the endpoint.  If port is -1, then it will scan for ports
        until it finds one that's free.  Returns the port number that it's
        listening on.  Returns -1 on an error.

        Will start up the given number of threads to help spread the work.

        If synchronous is true, it will only return once there is at least
        one thread waiting for connections to come in.  If synchronous is
        false, then it will return before there is anything listening and
        may eventually fail to make the connection.  In this case, the
        notifyStartup() function will be called once everything has started
        up.

        If threads is zero, then nothing will actually be done until a
        thread calls useThisThread() to do work.
    */
    int init(PortRange const & portRange = PortRange(), const std::string & hostname = "localhost",
             int threads = 1, bool synchronous = true, bool nameLookup=true,
             int backlog = DEF_BACKLOG);

    /** Listen on the given port.  If port is -1, then it should scan
        for a port and return that.  Returns the port number.
    */
    virtual int listen(PortRange const & portRange, const std::string & host,bool nameLookup=true,
                       int backlog = DEF_BACKLOG)
    {
        if (!acceptor)
            throw ML::Exception("can't listen without acceptor");

        return acceptor->listen(portRange, host, this, nameLookup, backlog);
    }

    /** Wait until we are ready to accept connections */
    void waitListening()
        const
    {
        if (!acceptor)
            throw ML::Exception("can't listen without acceptor");

        acceptor->waitListening();
    }

    /** Closing the peer in the context of a passive endpoint means
        simply not accepting connections any more.
    */
    virtual void closePeer()
    {
        return acceptor->closePeer();
    }

    /** What host are we connected to? */
    virtual std::string hostname() const
    {
        return acceptor->hostname();
    }

    /** What port are we listening on? */
    virtual int port() const
    {
        return acceptor->port();
    }

    /** Object that can be overridden to create the connection handler to
        be associated with the transport.
    */
    boost::function<std::shared_ptr<ConnectionHandler> ()> onMakeNewHandler;

    /** Object that can be overridden to deal with an accept error. */
    boost::function<void (std::string)> onAcceptError;
    
protected:

    /** Acceptor object to actually do the accepting. */
    std::shared_ptr<Acceptor> acceptor;

    virtual void
    associateHandler(const std::shared_ptr<TransportBase> & transport)
    {
        if (!transport)
            throw ML::Exception("no transport");
        transport->hasConnection();
        notifyNewTransport(transport);

        auto finishAccept = [=] ()
            {
                std::shared_ptr<ConnectionHandler> handler
                    = this->makeNewHandler();
                transport->associate(handler);
            };

        transport->doAsync(finishAccept, "finishAccept");
    }

    virtual std::shared_ptr<ConnectionHandler>
    makeNewHandler()
    {
        return onMakeNewHandler();
    }

    virtual void acceptError(const std::string & error)
    {
        if (onAcceptError) onAcceptError(error);
        else {
            using namespace std;
            cerr << "error accepting connection: " << error << endl;
        }
    }

    template<typename Transport> friend struct AcceptorT;
    // whether or not to perform a host name look up
    bool nameLookup_;// whether or not to perform a host name look up
};


/*****************************************************************************/
/* ACCEPTOR TEMPLATE                                                         */
/*****************************************************************************/

template<typename Transport>
struct AcceptorT: public Acceptor {
};

/*****************************************************************************/
/* ACCEPTOR TEMPLATE FOR SOCKET TRANSPORT                                    */
/*****************************************************************************/

template<>
struct AcceptorT<SocketTransport> : public Acceptor {

    AcceptorT();
    virtual ~AcceptorT();

    /** Listen on the given address for connections. */
    virtual int listen(PortRange const & portRange,
                       const std::string & hostname,
                       PassiveEndpoint * endpoint,
                       bool nameLookup,
                       int backlog);

    /** Close down the acceptor. */
    virtual void closePeer();

    /** What host are we connected to? */
    virtual std::string hostname() const;

    /** What port are we listening on? */
    virtual int port() const;

    /** Special thread to deal with accepting connections all by itself to
        avoid multiplexing them on the router.
    */
    void runAcceptThread();

    /** Wait until we are ready to accept connections */
    void waitListening() const;

protected:
    std::shared_ptr<boost::thread> acceptThread;
    ML::Wakeup_Fd wakeup;
    ACE_INET_Addr addr;
    int fd;
    PassiveEndpoint * endpoint;
    int listening_; // whether the socket is listening
    bool nameLookup;
    bool shutdown;
};


/*****************************************************************************/
/* PASSIVE ENDPOINT TEMPLATE                                                 */
/*****************************************************************************/

template<typename Transport>
struct PassiveEndpointT: public PassiveEndpoint {

    PassiveEndpointT(const std::string & name)
        : PassiveEndpoint(name)
    {
        acceptor.reset(new AcceptorT<Transport>());
    }
    
    virtual ~PassiveEndpointT()
    {
    }
};

} // namespace Datacratic

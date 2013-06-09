/* active_endpoint.h                                               -*- C++ -*-
   Jeremy Barnes, 29 April 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.
   
   Active endpoint class.
*/

#pragma once

#include "soa/service//endpoint.h"
#include "connection_handler.h"
#include "ace/SOCK_Connector.h"

namespace Datacratic {

/*****************************************************************************/
/* CONNECTOR                                                                 */
/*****************************************************************************/

struct Connector {
    virtual ~Connector()
    {
    }

    virtual std::shared_ptr<TransportBase>
    makeNewTransport(EndpointBase * owner) = 0;

    virtual int doConnect(const std::shared_ptr<TransportBase> & transport,
                          const ACE_INET_Addr & addr,
                          double timeout,
                          bool block) = 0;
    virtual void closePeer() = 0;
    
};


/*****************************************************************************/
/* ACTIVE ENDPOINT                                                           */
/*****************************************************************************/

/** An endpoint that actively creates a given type of connection.
*/

struct ActiveEndpoint: public EndpointBase {

    ActiveEndpoint(const std::string & name)
        : EndpointBase(name)
    {
        modifyIdle = false;
    }

    ~ActiveEndpoint()
    {
        closePeer();
        shutdown();
    }

    /** Default function that will throw an exception on a connection
        error.
    */
    static void throwExceptionOnConnectionError(const std::string & error);

    /** Do nothing on a connection error. */
    static void doNothingOnConnectionError(const std::string & error);

    typedef boost::function<void (const std::shared_ptr<TransportBase> &)>
        OnNewConnection;
    typedef boost::function<void (std::string)> OnConnectionError;

    /** Initialize the endpoint.  This will pre-create the given number of
        connections to the given hostname:port and the given number of
        threads to service them.

        Returns the number of connections that were created.
    */
    int init(int port, const std::string & hostname,
             int connections = 0,
             int threads = 1, bool synchronous = true,
             bool throwIfAnyConnectionError = true,
             OnConnectionError onEachConnectionError
                 = doNothingOnConnectionError,
             double timeout = 1.0);

    /** Pre-seed with the given number of connections in the connection
        pool.
    */
    int createConnections(int num_connections, bool synchronous,
                          bool throwIfAnyConnectionError,
                          OnConnectionError onEachConnectionError,
                          double timeout = 1.0);

    /** Create a new connection. */
    virtual void newConnection(OnNewConnection onNewConnection,
                               OnConnectionError onConnectionError,
                               double timeout = 1.0)
    {
        std::shared_ptr<TransportBase> transport = makeNewTransport();
        transport->associate
            (ML::make_std_sp(new ConnectManager(onNewConnection,
                                            onConnectionError,
                                            this)));

        doConnect(transport, addr, timeout, false /* block */);
    }

    /** Return a connection, either from the pool or by creating a new
        connection.
    */
    virtual void getConnection(OnNewConnection onNewConnection,
                               OnConnectionError onConnectionError,
                               double timeout);

    /** Method to cause the connection to actually happen. */
    virtual int doConnect(const std::shared_ptr<TransportBase> & transport,
                           const ACE_INET_Addr & addr,
                           double timeout,
                           bool block)
    {
        return connector->doConnect(transport, addr, timeout, block);
    }

    virtual std::shared_ptr<TransportBase> makeNewTransport()
    {
        return connector->makeNewTransport(this);
    }

    void closePeer()
    {
        return connector->closePeer();
    }

    /** What host are we connected to? */
    virtual std::string hostname() const
    {
        return hostname_;
    }

    /** What port are we listening on? */
    virtual int port() const
    {
        return port_;
    }

    int numActiveConnections() const
    {
        Guard guard(lock);
        return active.size();
    }
    
    int numInactiveConnections() const
    {
        Guard guard(lock);
        return inactive.size();
    }

    /** Dump the state of the endpoint for debugging. */
    virtual void dumpState() const;

    void shutdown();

protected:
    Connections active, inactive;
    int port_;
    std::string hostname_;
    ACE_INET_Addr addr;

    std::shared_ptr<Connector> connector;

    /** Tell the endpoint that a new transport has been created. */
    virtual void
    notifyNewTransport(const std::shared_ptr<TransportBase> & transport);

    /** Tell the endpoint that a connection has been opened. */
    virtual void
    notifyTransportOpen(const std::shared_ptr<TransportBase> & transport);

    /** Tell the endpoint that a connection has been closed.  Removes it
        from the active/inactive connections.
    */
    virtual void
    notifyCloseTransport(const std::shared_ptr<TransportBase> & transport);

    /** Tell the endpoint that a connection has been recycled.  Moves it
        from active to inactive.
    */
    virtual void
    notifyRecycleTransport(const std::shared_ptr<TransportBase> & transport);

    struct ConnectManager : public ConnectionHandler {

        ConnectManager(OnNewConnection onNewConnection,
                       OnConnectionError onConnectionError,
                       ActiveEndpoint * owner)
            : onNewConnection(onNewConnection),
              onConnectionError(onConnectionError),
              success(false), doneError(false),
              owner(owner)
        {
        }

        ActiveEndpoint::OnNewConnection onNewConnection;
        ActiveEndpoint::OnConnectionError onConnectionError;
        bool success;
        bool doneError;
        ActiveEndpoint * owner;

        virtual void doError(const std::string & error)
        {
            onConnectionError(error);
            success = false;
            doneError = true;
            //transport().asyncClose();
        }

        virtual void handleError(const std::string & error)
        {
            onConnectionError(error);
            success = false;
            doneError = true;
            closeWhenHandlerFinished();
        }
        
        virtual void onCleanup()
        {
        }

        virtual void handleOutput()
        {
            if (doneError || success)
                throw ML::Exception("too many events for connector");
            using namespace std;
            //cerr << "connect on " << fd << " finished" << endl;

            transport().cancelTimer();
            stopWriting();

            // Connection finished or has an error; check which one
            int error = 0;
            socklen_t error_len = sizeof(int);
            int res = getsockopt(getHandle(), SOL_SOCKET, SO_ERROR,
                                 &error, &error_len);
            if (res == -1 || error_len != sizeof(int))
                std::cerr << "error getting connect message: "
                          << strerror(errno)
                          << std::endl;
            
            if (error != 0)
                throw ML::Exception("connect success but error");

            transport().hasConnection();

            Guard guard(owner->lock);
            
            // We how have a connection
            
            owner->notifyTransportOpen(transport().shared_from_this());
            
            //cerr << "doing onNewConnection" << endl;
            
            onNewConnection(transport().shared_from_this());
            
            //cerr << "finished onNewConnection" << endl;            
        }
    
        virtual void handleTimeout(Date date, size_t)
        {
            onConnectionError("connect: connection timed out");
            closeWhenHandlerFinished();
        }
    };
};


/*****************************************************************************/
/* CONNECTOR TEMPLATE                                                        */
/*****************************************************************************/

template<typename Transport>
struct ConnectorT : public Connector {
    ACE_SOCK_Connector connector;

    ConnectorT()
    {
    }

    virtual std::shared_ptr<TransportBase>
    makeNewTransport(EndpointBase * owner)
    {
        return ML::make_std_sp(new Transport(owner));
    }

    virtual int doConnect(const std::shared_ptr<TransportBase> & transport,
                          const ACE_INET_Addr & addr,
                          double timeout,
                          bool block)
    {
        using namespace std;
        //cerr << "doConnect: block = " << block << endl;

        Transport * t = static_cast<Transport *>(transport.get());

        ACE_Time_Value to(0, 0);
        if (block)
            to = ACE_Time_Value((int)timeout,
                                (timeout - (int)timeout) * 1000000);
        int res;
        do {
            res = connector.connect(t->peer(), addr, &to);
#if 0
            using namespace std;
            cerr << "connect block=" << block << " on fd "
                 << t->getHandle() << " returned "
                 << res << " errno "
                 << strerror(errno) << endl;
#endif
        } while (res == -1 && errno == EINTR);
        
        if (res == -1 && (block
                          || (errno != EINPROGRESS && errno != EAGAIN))) {
            using namespace std;
            cerr << "connect returned " << res << " with errno "
                 << strerror(errno) << endl;
            abort();
            t->doError("connect: " + std::string(strerror(errno)));
            //t->close();
            return -1;
        }

        if (res == -1) {
            // Asynchronous setup

            //cerr << "transport->getHandle() = " << transport->getHandle()
            //     << endl;

            transport->get_endpoint()->notifyNewTransport(transport);

            auto finishSetup = [=] ()
                {
                    // Set up a timeout
                    t->scheduleTimerRelative(timeout);
                    
                    // Ready to write
                    t->startWriting();
                };
            
            // Call the rest in a handler context
            transport->doAsync(finishSetup, "connect");
        }
        else {
            t->handleOutput();
            return 0;
        }

        return res;
    }

    void closePeer()
    {
    }
};


/*****************************************************************************/
/* ACTIVE ENDPOINT TEMPLATE                                                  */
/*****************************************************************************/

template<typename Transport>
struct ActiveEndpointT : public ActiveEndpoint {
    ActiveEndpointT(const std::string & name)
        : ActiveEndpoint(name)
    {
        connector.reset(new ConnectorT<Transport>());
    }
};


} // namespace Datacratic

/* zmq_endpoint.h                                                  -*- C++ -*-
   Jeremy Barnes, 25 September 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Endpoints for zeromq.
*/

#ifndef __service__zmq_endpoint_h__
#define __service__zmq_endpoint_h__

#include "named_endpoint.h"
#include "message_loop.h"
#include "logs.h"
#include <set>
#include <type_traits>
#include "jml/utils/smart_ptr_utils.h"
#include "jml/utils/vector_utils.h"
#include <boost/make_shared.hpp>
#include "jml/arch/backtrace.h"
#include "jml/arch/timers.h"
#include "jml/arch/cmp_xchg.h"
#include "zmq_utils.h"

namespace Datacratic {


/******************************************************************************/
/* ZMQ LOGS                                                                   */
/******************************************************************************/

struct ZmqLogs
{
    static Logging::Category print;
    static Logging::Category error;
    static Logging::Category trace;
};


/*****************************************************************************/
/* ZMQ EVENT SOURCE                                                          */
/*****************************************************************************/

/** Adaptor that allows any zeromq socket to hook into an event loop. */

struct ZmqEventSource : public AsyncEventSource {

    typedef std::function<void (std::vector<std::string>)>
        AsyncMessageHandler;
    AsyncMessageHandler asyncMessageHandler;

    typedef std::function<std::vector<std::string> (std::vector<std::string>)>
        SyncMessageHandler;
    SyncMessageHandler syncMessageHandler;

    typedef std::mutex SocketLock;

    ZmqEventSource();

    ZmqEventSource(zmq::socket_t & socket, SocketLock * lock = nullptr);

    /** Construct the event source from a function object that returns
        something that is not convertible to a std::vector<std::string>.
        This will cause the asynchronous message handler to be replaced
        by the passed function.
    */
    template<typename T>
    ZmqEventSource(zmq::socket_t & socket,
                   const T & handler,
                   SocketLock * lock = nullptr,
                   typename std::enable_if<!std::is_convertible<decltype(std::declval<T>()(std::declval<std::vector<std::string> >())),
                                                                std::vector<std::string> >::value, void>::type * = 0)
        : asyncMessageHandler(handler)
    {
        init(socket, lock);
    }

    /** Construct the event source from a function object that returns a
        std::vector<std::string>.  This will cause the synchronous message
        handler to be replaced by the passed function.
    */
    template<typename T>
    ZmqEventSource(zmq::socket_t & socket,
                   const T & handler,
                   SocketLock * lock = nullptr,
                   typename std::enable_if<std::is_convertible<decltype(std::declval<T>()(std::declval<std::vector<std::string> >())),
                                                                std::vector<std::string> >::value, void>::type * = 0)
        : syncMessageHandler(handler)
    {
        init(socket, lock);
    }

    void init(zmq::socket_t & socket, SocketLock * lock = nullptr);

    virtual int selectFd() const;

    virtual bool poll() const;

    virtual bool processOne();

    /** Handle a message.  The default implementation will call
        syncMessageHandler if it is defined; otherwise it calls
        handleSyncMessage and writes back the response to the socket.
    */
    virtual void handleMessage(const std::vector<std::string> & message);

    /** Handle a message and write a synchronous response.  This will forward
        to asyncMessageHandler if defined, or otherwise throw an exception.
    */
    virtual std::vector<std::string>
    handleSyncMessage(const std::vector<std::string> & message);

    zmq::socket_t & socket() const
    {
        ExcAssert(socket_);
        return *socket_;
    }

    SocketLock * socketLock() const
    {
        return socketLock_;
    }

protected:
    zmq::socket_t * socket_;

    SocketLock * socketLock_;

    /** Update the current cached event mask.  Note that this requires that the
        socketLock be taken if non-null.
    */
    void updateEvents() const;

    /// Mask of current events that are pending on the socket.
    mutable int currentEvents;
};


/*****************************************************************************/
/* ZMQ BINARY EVENT SOURCE                                                   */
/*****************************************************************************/

/** Adaptor that allows any zeromq socket to hook into an event loop. */

struct ZmqBinaryEventSource : public AsyncEventSource {

    typedef std::function<void (std::vector<zmq::message_t> &&)>
        MessageHandler;
    MessageHandler messageHandler;

    ZmqBinaryEventSource()
        : socket_(0)
    {
        needsPoll = true;
    }

    ZmqBinaryEventSource(zmq::socket_t & socket,
                         MessageHandler messageHandler = MessageHandler())
        : messageHandler(std::move(messageHandler)),
          socket_(&socket)
    {
        needsPoll = true;
    }

    void init(zmq::socket_t & socket)
    {
        socket_ = &socket;
        needsPoll = true;
    }

    virtual int selectFd() const
    {
        int res = -1;
        size_t resSize = sizeof(int);
        socket().getsockopt(ZMQ_FD, &res, &resSize);
        if (res == -1)
            THROW(ZmqLogs::error) << "no fd for zeromq socket" << std::endl;
        return res;
    }

    virtual bool poll() const
    {
        return getEvents(socket()).first;
    }

    virtual bool processOne()
    {
        ExcAssert(socket_);

        std::vector<zmq::message_t> messages;

        int64_t more = 1;
        size_t more_size = sizeof (more);

        while (more) {
            zmq::message_t message;
            bool got = socket_->recv(&message, messages.empty() ? ZMQ_NOBLOCK: 0);
            if (!got) return false;  // no first part available
            messages.emplace_back(std::move(message));
            socket_->getsockopt(ZMQ_RCVMORE, &more, &more_size);
        }

        handleMessage(std::move(messages));

        return poll();
    }

    /** Handle a message.  The default implementation will call
        syncMessageHandler if it is defined; otherwise it calls
        handleSyncMessage and writes back the response to the socket.
    */
    virtual void handleMessage(std::vector<zmq::message_t> && message)
    {
        if (messageHandler)
            messageHandler(std::move(message));
        else {
            THROW(ZmqLogs::error) << "no message handler set" << std::endl;
        }
    }

    zmq::socket_t & socket() const
    {
        ExcAssert(socket_);
        return *socket_;
    }

    zmq::socket_t * socket_;

};


/*****************************************************************************/
/* ZMQ BINARY TYPED EVENT SOURCE                                             */
/*****************************************************************************/

/** An adaptor that is used to deal with a zeromq connection that sends
    length one messages of a binary data structure.
*/

template<typename Arg>
struct ZmqBinaryTypedEventSource: public AsyncEventSource {

    typedef std::function<void (Arg)> MessageHandler;
    MessageHandler messageHandler;

    ZmqBinaryTypedEventSource()
        : socket_(0)
    {
        needsPoll = true;
    }

    ZmqBinaryTypedEventSource(zmq::socket_t & socket,
                              MessageHandler messageHandler = MessageHandler())
        : messageHandler(std::move(messageHandler)),
          socket_(&socket)
    {
        needsPoll = true;
    }

    void init(zmq::socket_t & socket)
    {
        socket_ = &socket;
        needsPoll = true;
    }

    virtual int selectFd() const
    {
        int res = -1;
        size_t resSize = sizeof(int);
        socket().getsockopt(ZMQ_FD, &res, &resSize);
        if (res == -1)
            THROW(ZmqLogs::error) << "no fd for zeromq socket" << std::endl;
        return res;
    }

    virtual bool poll() const
    {
        return getEvents(socket()).first;
    }

    virtual bool processOne()
    {
        zmq::message_t message;
        ExcAssert(socket_);
        bool got = socket_->recv(&message, ZMQ_NOBLOCK);
        if (!got) return false;

        handleMessage(* reinterpret_cast<const Arg *>(message.data()));

        return poll();
    }

    virtual void handleMessage(const Arg & arg)
    {
        if (messageHandler)
            messageHandler(arg);
        else {
            THROW(ZmqLogs::error) << "no message handler set" << std::endl;
        }
    }

    zmq::socket_t & socket() const
    {
        ExcAssert(socket_);
        return *socket_;
    }

    zmq::socket_t * socket_;
};


/*****************************************************************************/
/* ZMQ TYPED EVENT SOURCE                                                    */
/*****************************************************************************/

/** A sink that listens for zeromq messages of the format:

    1.  address      (optional)
    2.  message kind
    3.  message type
    4.  binary payload

    and calls a callback on the decoded message.
*/

template<typename T>
struct ZmqTypedEventSource: public ZmqEventSource {
    typedef std::function<void (T &&, std::string address)>
      OnMessage;

    OnMessage onMessage;
    bool routable;
    std::string messageTopic;

    ZmqTypedEventSource()
    {
    }

    ZmqTypedEventSource(zmq::socket_t & socket,
                        bool routable,
                        const std::string & messageTopic)
    {
        init(routable, messageTopic);
    }

    void init(zmq::socket_t & socket,
              bool routable,
              const std::string messageTopic)
    {
        this->routable = routable;
        this->messageTopic = messageTopic;
    }

    virtual void handleMessage(const std::vector<std::string> & message)
    {
        int expectedSize = routable + 2;
        if (message.size() != expectedSize) {
            THROW(ZmqLogs::error) << "unexpected message size: "
                << "expected=" << expectedSize << ", got=" << message.size()
                << std::endl;
        }

        int i = routable;
        if (message[i + 1] != messageTopic) {
            THROW(ZmqLogs::error) << "unexpected message kind: "
                << "expected=" << message[i + 1] << ", got=" << messageTopic
                << std::endl;
        }

        std::istringstream stream(message[i + 2]);
        ML::DB::Store_Reader store(stream);
        T result;
        store >> result;

        handleTypedMessage(std::move(result), routable ? message[0] : "");
    }

    virtual void handleTypedMessage(T && message, const std::string & address)
    {
        if (onMessage)
            onMessage(message, address);
        else {
            THROW(ZmqLogs::error) << "no message handler set" << std::endl;
        }
    }
};


/*****************************************************************************/
/* ZMQ SOCKET MONITOR                                                        */
/*****************************************************************************/

/** This class allows a zeromq socket to be monitored and events received
    on various events.

    It is an AsyncEventHandler which must be integrated into a message
    queue.  The events will be delivered when processed by the message
    queue.
*/

struct ZmqSocketMonitor : public ZmqBinaryTypedEventSource<zmq_event_t> {

    ZmqSocketMonitor(zmq::context_t & context);

    ~ZmqSocketMonitor()
    {
        shutdown();
    }

    void shutdown();

    /** Initialize the socket monitor to monitor the given socket.

        The events parameter is an event mask; see
        http://api.zeromq.org/3-2:zmq-socket-monitor for details of
        what it means.
    */
    void init(zmq::socket_t & socketToMonitor, int events = ZMQ_EVENT_ALL);

    /** Disconnect so that no more events will be received. */
    void disconnect();

    /** Event handler for events:
        - The first argument is the address which the event concerns;
        - The second argument is
          - In the case of a success message or an event message,
            the fd which was created to deal with the socket or the
            fd on which the event occurred
          - In the case of a failure message, the errno for the failed
            operation
          - In the case of a retry message, the time (in milliseconds) when
            the operation will be retried.
    */
    typedef std::function<void (std::string, int, zmq_event_t)> EventHandler;

    // Success handlers
    EventHandler connectHandler, bindHandler, acceptHandler;

    // Socket event handlers
    EventHandler closeHandler, disconnectHandler;

    // Failure handlers
    EventHandler connectFailureHandler, acceptFailureHandler,
        bindFailureHandler, closeFailureHandler;

    // Retry handlers
    EventHandler connectRetryHandler;

    // Catch all handler, for when other handlers aren't registered
    EventHandler defaultHandler;

    /** Function called to handle an event.  Default implementation looks
        at the event type and dispatches to one of the xxxHandler variables
        above, or defaultHandler if it's empty.

        This can be overriden to modify the behaviour.
    */
    virtual int handleEvent(const zmq_event_t & event);

private:
    typedef std::mutex Lock;
    mutable Lock lock;

    /// Uri of socket connected to
    std::string connectedUri;

    /// Zeromq pull endpoint that listens to state change messages
    std::unique_ptr<zmq::socket_t> monitorEndpoint;

    /// Address of connected socket
    zmq::socket_t * monitoredSocket;
};


/*****************************************************************************/
/* ZEROMQ NAMED ENDPOINT                                                     */
/*****************************************************************************/

/** An endpoint that exposes a zeromq interface that is passive (bound to
    one or more listening ports) and is published in a configuration service.

    Note that the endpoint may be connected to more than one thing.
*/

struct ZmqNamedEndpoint : public NamedEndpoint, public MessageLoop {

    ZmqNamedEndpoint(std::shared_ptr<zmq::context_t> context);

    ~ZmqNamedEndpoint()
    {
        shutdown();
    }

    void init(std::shared_ptr<ConfigurationService> config,
              int socketType,
              const std::string & endpointName);

    void shutdown()
    {
        MessageLoop::shutdown();

        if (socket_) {
            unbindAll();
            socket_.reset();
        }
    }

    /** Bind into a tcp port.  If the preferred port is not available, it will
        scan until it finds one that is.

        Returns the uri to connect to.
    */
    std::string bindTcp(PortRange const & portRange = PortRange(), std::string host = "");

    /** Bind to the given zeromq uri, but don't publish it. */
    void bind(const std::string & address)
    {
        if (!socket_)
            THROW(ZmqLogs::error) << "bind called before init" << std::endl;

        std::unique_lock<Lock> guard(lock);
        socket_->bind(address);
        boundAddresses[address];
    }

    /** Unbind to all addresses. */
    void unbindAll()
    {
        std::unique_lock<Lock> guard(lock);
        ExcAssert(socket_);
        for (auto addr: boundAddresses)
            socket_->tryUnbind(addr.first);
    }

    template<typename... Args>
    void sendMessage(Args&&... args)
    {
        using namespace std;
        std::unique_lock<Lock> guard(lock);
        ExcAssert(socket_);
        Datacratic::sendMessage(*socket_, std::forward<Args>(args)...);
    }

    void sendMessage(const std::vector<std::string> & message)
    {
        using namespace std;
        std::unique_lock<Lock> guard(lock);
        ExcAssert(socket_);
        Datacratic::sendAll(*socket_, message);
    }

    /** Send a raw message on. */
    void sendMessage(std::vector<zmq::message_t> && message)
    {
        using namespace std;
        std::unique_lock<Lock> guard(lock);
        ExcAssert(socket_);
        for (unsigned i = 0;  i < message.size();  ++i) {
            socket_->send(message[i],
                          i == message.size() - 1
                          ? 0 : ZMQ_SNDMORE);
        }
    }

    /** Very unsafe method as it bypasses all thread safety. */
    zmq::socket_t & getSocketUnsafe() const
    {
        ExcAssert(socket_);
        return *socket_;
    }

    typedef std::function<void (std::vector<zmq::message_t> &&)>
        RawMessageHandler;
    RawMessageHandler rawMessageHandler;

    typedef std::function<void (std::vector<std::string> &&)> MessageHandler;
    MessageHandler messageHandler;

    /** Handle a message.  The default implementation will call
        syncMessageHandler if it is defined; otherwise it converts the message
        to strings and calls handleMessage.
    */
    virtual void handleRawMessage(std::vector<zmq::message_t> && message)
    {
        if (rawMessageHandler)
            rawMessageHandler(std::move(message));
        else {
            std::vector<std::string> msg2;
            for (unsigned i = 0;  i < message.size();  ++i) {
                msg2.push_back(message[i].toString());
            }
            handleMessage(std::move(msg2));
        }
    }

    virtual void handleMessage(std::vector<std::string> && message)
    {
        if (messageHandler)
            messageHandler(std::move(message));
        else
            THROW(ZmqLogs::error) << "no message handler set" << std::endl;
    }

    typedef std::function<void (std::string bindAddress)>
        ConnectionEventHandler;

    /** Callback for when we accept an incoming connection. */
    ConnectionEventHandler acceptEventHandler;

    /** Callback for when an incoming connection is closed. */
    ConnectionEventHandler disconnectEventHandler;

    /** Callback for when we are no longer bound to an address. */
    ConnectionEventHandler closeEventHandler;

    /** Callback for when we accept an incoming connection. */
    virtual void handleAcceptEvent(std::string boundAddress)
    {
        if (acceptEventHandler)
            acceptEventHandler(boundAddress);
    }

    /** Callback for when an incoming connection is closed. */
    virtual void handleDisconnectEvent(std::string boundAddress)
    {
        if (disconnectEventHandler)
            disconnectEventHandler(boundAddress);
    }

    /** Callback for when we are no longer bound to an address. */
    virtual void handleCloseEvent(std::string boundAddress)
    {
        if (closeEventHandler)
            closeEventHandler(boundAddress);
    }

    /** Interrogates the number of addresses we're bound to. */
    size_t numBoundAddresses() const
    {
        std::unique_lock<Lock> guard(lock);
        return boundAddresses.size();
    }

    /** Interrogate the number of connections.  If addr is the empty string
        then it is over all bound addresses.
    */
    size_t numActiveConnections(std::string addr = "") const
    {
        std::unique_lock<Lock> guard(lock);
        if (addr == "") {
            size_t result = 0;
            for (auto & addr: boundAddresses)
                result += addr.second.connectedFds.size();
            return result;
        }
        else {
            auto it = boundAddresses.find(addr);
            if (it == boundAddresses.end())
                return 0;
            return it->second.connectedFds.size();
        }
    }

private:
    typedef std::mutex Lock;
    mutable Lock lock;

    std::shared_ptr<zmq::context_t> context_;
    std::shared_ptr<zmq::socket_t> socket_;

    struct AddressInfo {
        AddressInfo()
            : listeningFd(-1)
        {
        }

        /// File descriptor we're listening on
        int listeningFd;

        /// Set of file descriptors we're connected to
        std::set<int> connectedFds;
    };

    /// Information for each bound address
    std::map<std::string, AddressInfo> boundAddresses;

    /// Zeromq socket type
    int socketType;
};


/*****************************************************************************/
/* ZMQ NAMED CLIENT BUS                                                      */
/*****************************************************************************/

/** A named service endpoint that keeps track of the clients that are
    connected and will notify on connection and disconnection.
*/
struct ZmqNamedClientBus: public ZmqNamedEndpoint {

    ZmqNamedClientBus(std::shared_ptr<zmq::context_t> context,
                      double deadClientDelay = 5.0)
        : ZmqNamedEndpoint(context), deadClientDelay(deadClientDelay)
    {
    }

    void init(std::shared_ptr<ConfigurationService> config,
              const std::string & endpointName)
    {
        ZmqNamedEndpoint::init(config, ZMQ_XREP, endpointName);
        addPeriodic("ZmqNamedClientBus::checkClient", 1.0,
                    [=] (uint64_t v) { this->onCheckClient(v); });
    }

    virtual ~ZmqNamedClientBus()
    {
        shutdown();
    }

    void shutdown()
    {
        MessageLoop::shutdown();
        ZmqNamedEndpoint::shutdown();
    }

    /** How long until we decide a client that's not sending a heartbeat is
        dead.
    */
    double deadClientDelay;

    /** Function called when something connects to the bus */
    std::function<void (std::string)> onConnection;

    /** Function called when something disconnects from the bus (we can tell
        due to timeouts).
    */
    std::function<void (std::string)> onDisconnection;



    template<typename... Args>
    void sendMessage(const std::string & address,
                     const std::string & topic,
                     Args&&... args)
    {
        ZmqNamedEndpoint::sendMessage(address, topic,
                                      std::forward<Args>(args)...);
    }

    virtual void handleMessage(std::vector<std::string> && message)
    {
        using namespace std;

        const std::string & agent = message.at(0);
        const std::string & topic = message.at(1);

        if (topic == "HEARTBEAT") {
            // Not the first message from the client
            auto it = clientInfo.find(agent);
            if (it == clientInfo.end()) {
                // Disconnection then reconnection
                if (onConnection)
                    onConnection(agent);
                it = clientInfo.insert(make_pair(agent, ClientInfo())).first;
            }
            it->second.lastHeartbeat = Date::now();
            sendMessage(agent, "HEARTBEAT");
        }

        else if (topic == "HELLO") {
            // First message from client
            auto it = clientInfo.find(agent);
            if (it == clientInfo.end()) {
                // New connection
                if (onConnection)
                    onConnection(agent);
                it = clientInfo.insert(make_pair(agent, ClientInfo())).first;
            }
            else {
                // Client must have disappeared then reappared without us
                // noticing.
                // Do this disconnection, then the reconnection
                if (onDisconnection)
                    onDisconnection(agent);
                if (onConnection)
                    onConnection(agent);
            }
            it->second.lastHeartbeat = Date::now();
            sendMessage(agent, "HEARTBEAT");
        }
        else {
            handleClientMessage(message);
        }

    }

    typedef std::function<void (std::vector<std::string>)>
    ClientMessageHandler;
    ClientMessageHandler clientMessageHandler;

    virtual void handleClientMessage(const std::vector<std::string> & message)
    {
        if (clientMessageHandler)
            clientMessageHandler(message);
        else {
            THROW(ZmqLogs::error)
                << "no message handler set for client " << message.at(1)
                << std::endl;
        }
    }

private:
    void onCheckClient(uint64_t numEvents)
    {
        Date now = Date::now();
        Date expiry = now.plusSeconds(-deadClientDelay);

        std::vector<std::string> deadClients;

        for (auto & c: clientInfo)
            if (c.second.lastHeartbeat < expiry)
                deadClients.push_back(c.first);

        for (auto d: deadClients) {
            if (onDisconnection)
                onDisconnection(d);
            clientInfo.erase(d);
        }
    }

    struct ClientInfo {
        ClientInfo()
            : lastHeartbeat(Date::now())
        {
        }

        Date lastHeartbeat;
    };

    std::map<std::string, ClientInfo> clientInfo;
};


/** Flags to modify how we create a connection. */
enum ConnectionStyle {
    CS_ASYNCHRONOUS,  ///< Asynchronous; returns true
    CS_SYNCHRONOUS,   ///< Synchronous; returns state of connection
    CS_MUST_SUCCEED   ///< Synchronous; returns true or throws
};




/*****************************************************************************/
/* ZEROMQ NAMED PROXY                                                        */
/*****************************************************************************/

/** Proxy to connect to a named zeromq-based service. */

// THIS SHOULD BE REPLACED BY ZmqNamedSocket

struct ZmqNamedProxy: public MessageLoop {

    ZmqNamedProxy();

    ZmqNamedProxy(std::shared_ptr<zmq::context_t> context, int shardIndex = -1);

    ~ZmqNamedProxy()
    {
        shutdown();
    }

    void shutdown()
    {
        MessageLoop::shutdown();
        if(socket_) {
            std::lock_guard<ZmqEventSource::SocketLock> guard(socketLock_);
            socket_.reset();
        }
    }

    bool isConnected() const { return connectionState == CONNECTED; }

    /** Type of callback for a new connection. */
    typedef std::function<void (std::string)> ConnectionHandler;

    /** Callback that will be called when we get a new connection if
        onConnect() is not overridden. */
    ConnectionHandler connectHandler;

    /** Function that will be called when we make a new connection to a
        remote service provider.
    */
    virtual void onConnect(const std::string & source)
    {
        if (connectHandler)
            connectHandler(source);
    }

    /** Callback that will be called when we get a disconnection if
        onDisconnect() is not overridden. */
    ConnectionHandler disconnectHandler;

    /** Function that will be called when we lose a connection to a
        remote service provider.
    */
    virtual void onDisconnect(const std::string & source)
    {
        if (disconnectHandler)
            disconnectHandler(source);
    }

    void init(std::shared_ptr<ConfigurationService> config,
              int socketType,
              const std::string & identity = "");

    /** Connect to the given endpoint via zeromq.  Returns whether the
        connection could be established.

        If synchronous is true, then the method will not return until
        the connection is really established.

        If mustSucceed is true, then an exception will be thrown if the
        connection cannot be established.
    */
    bool connect(const std::string & endpointName,
                 ConnectionStyle style = CS_ASYNCHRONOUS);

    /** Connect to one of the endpoints that provides the given service.
        The endpointName tells which endpoint of the service class that
        should be connected to.

        Looks up the service providers in the configuration service.
    */
    bool connectToServiceClass(const std::string & serviceClass,
                               const std::string & endpointName,
                               bool local = true,
                               ConnectionStyle style = CS_ASYNCHRONOUS);

    /** Called back when one of our endpoints either changes or disappears. */
    bool onConfigChange(ConfigurationService::ChangeType change,
                        const std::string & key,
                        const Json::Value & newValue);

    /** Get the zeromq socket to listen to. */
    zmq::socket_t & socket() const
    {
        ExcAssert(socket_);
        return *socket_;
    }

    ZmqEventSource::SocketLock * socketLock() const
    {
        return &socketLock_;
    }

    template<typename... Args>
    void sendMessage(Args&&... args)
    {
        std::lock_guard<ZmqEventSource::SocketLock> guard(socketLock_);

        ExcCheckNotEqual(connectionState, NOT_CONNECTED,
                "sending on an unconnected socket: " + endpointName);

        if (connectionState == CONNECTION_PENDING) {
            LOG(ZmqLogs::error)
                << "dropping message for " << endpointName << std::endl;
            return;
        }

        Datacratic::sendMessage(socket(), std::forward<Args>(args)...);
    }

    void disconnect()
    {
        if (connectionState == NOT_CONNECTED) return;

        {
            std::lock_guard<ZmqEventSource::SocketLock> guard(socketLock_);

            if (connectionState == CONNECTED)
                socket_->disconnect(connectedUri);

            connectionState = NOT_CONNECTED;
        }

        onDisconnect(connectedUri);
    }

    size_t getShardIndex() const
    {
        return shardIndex;
    }

protected:
    ConfigurationService::Watch serviceWatch, endpointWatch;
    std::shared_ptr<ConfigurationService> config;
    std::shared_ptr<zmq::context_t> context_;
    std::shared_ptr<zmq::socket_t> socket_;

    mutable ZmqEventSource::SocketLock socketLock_;

    enum ConnectionType {
        NO_CONNECTION,        ///< No connection type yet
        CONNECT_DIRECT,       ///< Connect directly to a named service
        CONNECT_TO_CLASS,     ///< Connect to a service class
    } connectionType;

    enum ConnectionState {
        NOT_CONNECTED,      // connect() was not called
        CONNECTION_PENDING, // connect() was called but service is not available
        CONNECTED           // connect() was called and the socket was connected
    } connectionState;

    void onServiceNodeChange(const std::string & path,
                             ConfigurationService::ChangeType change);
    void onEndpointNodeChange(const std::string & path,
                              ConfigurationService::ChangeType change);

    std::string serviceClass;      ///< Service class we're connecting to
    std::string endpointName;      ///< Name of endpoint to connect to
    std::string connectedService;  ///< Name of service we're connected to
    std::string connectedUri;      ///< URI we're connected to
    bool local;
    int shardIndex;
};


/*****************************************************************************/
/* ZEROMQ NAMED CLIENT BUS PROXY                                             */
/*****************************************************************************/

/** Class designed to go on the other end of a zeromq named client bus.  This
    takes care of sending the keepalives and will allow the other end to
    detect when the connection is broken.

*/

struct ZmqNamedClientBusProxy : public ZmqNamedProxy {

    ZmqNamedClientBusProxy()
        : timeout(2.0)
    {
    }

    ZmqNamedClientBusProxy(std::shared_ptr<zmq::context_t> context, int shardIndex = -1)
        : ZmqNamedProxy(context, shardIndex), timeout(2.0)
    {
    }

    ~ZmqNamedClientBusProxy()
    {
        shutdown();
    }

    void init(std::shared_ptr<ConfigurationService> config,
              const std::string & identity = "")
    {
        ZmqNamedProxy::init(config, ZMQ_XREQ, identity);

        auto doMessage = [=] (const std::vector<std::string> & message)
            {
                const std::string & topic = message.at(0);
                if (topic == "HEARTBEAT")
                    this->lastHeartbeat = Date::now();
                else handleMessage(message);
            };

        addSource("ZmqNamedClientBusProxy::doMessage",
                  std::make_shared<ZmqEventSource>(socket(), doMessage, socketLock()));
 
        auto doHeartbeat = [=] (int64_t skipped)
            {
                if (connectionState != CONNECTED) return;

                sendMessage("HEARTBEAT");
            };

        addPeriodic("ZmqNamedClientBusProxy::doHeartbeat", 1.0, doHeartbeat);
   }

    void shutdown()
    {
        MessageLoop::shutdown();
        ZmqNamedProxy::shutdown();
    }

    virtual void onConnect(const std::string & where)
    {
        lastHeartbeat = Date::now();

        sendMessage("HELLO");

        if (connectHandler)
            connectHandler(where);
    }

    virtual void onDisconnect(const std::string & where)
    {
        if (disconnectHandler)
            disconnectHandler(where);
    }

    ZmqEventSource::AsyncMessageHandler messageHandler;

    virtual void handleMessage(const std::vector<std::string> & message)
    {
        if (messageHandler)
            messageHandler(message);
        else
            THROW(ZmqLogs::error) << "no message handler set" << std::endl;
    }

    Date lastHeartbeat;
    double timeout;
};


/*****************************************************************************/
/* ZEROMQ MULTIPLE NAMED CLIENT BUS PROXY                                    */
/*****************************************************************************/

/** Class designed to go on the other end of a zeromq named client bus.  This
    takes care of sending the keepalives and will allow the other end to
    detect when the connection is broken.

*/

struct ZmqMultipleNamedClientBusProxy: public MessageLoop {
    friend class ServiceDiscoveryScenario;
    friend class ServiceDiscoveryScenarioTest;

#define CHANGES_MAP_INITIALIZER \
    { \
        { ConfigurationService::VALUE_CHANGED, 0 }, \
        { ConfigurationService::DELETED, 0 }, \
        { ConfigurationService::CREATED, 0 }, \
        { ConfigurationService::NEW_CHILD, 0 } \
    }

    ZmqMultipleNamedClientBusProxy()
       : zmqContext(new zmq::context_t(1)),
         changesCount( CHANGES_MAP_INITIALIZER )
    {
        connected = false;
        inProvidersChanged = false;
    }

    ZmqMultipleNamedClientBusProxy(std::shared_ptr<zmq::context_t> context)
        : zmqContext(context),
          changesCount( CHANGES_MAP_INITIALIZER )
    {
        connected = false;
        inProvidersChanged = false;
    }

#undef CHANGES_MAP_INITIALIZER

    ~ZmqMultipleNamedClientBusProxy()
    {
        shutdown();
    }

    void init(std::shared_ptr<ConfigurationService> config,
              const std::string & identity = "")
    {
        this->config = config;
        this->identity = identity;
    }

    void shutdown()
    {
        MessageLoop::shutdown();
        for (auto & c: connections)
            if (c.second)
                c.second->shutdown();
    }

    template<typename... Args>
    void sendMessage(const std::string & recipient,
                     const std::string & topic,
                     Args&&... args) const
    {
        std::unique_lock<Lock> guard(connectionsLock);
        auto it = connections.find(recipient);
        if (it == connections.end()) {
            THROW(ZmqLogs::error)
                << "unable to deliver " << topic
                << " to unknown client " << recipient
                << std::endl;
        }
        it->second->sendMessage(topic, std::forward<Args>(args)...);
    }


    template<typename... Args>
    bool sendMessageToShard(size_t shard,
                            const std::string & topic,
                            Args&&... args) const
    {
        std::unique_lock<Lock> guard(connectionsLock);
        for (const auto& conn : connections) {
            if (conn.second->getShardIndex() != shard) continue;

            conn.second->sendMessage(topic, std::forward<Args>(args)...);
            return true;
        }
        return false;
    }

    bool isConnectedToShard(size_t shard) const
    {
        std::unique_lock<Lock> guard(connectionsLock);
        for (const auto& conn : connections) {
            if (conn.second->getShardIndex() != shard) continue;

            return conn.second->isConnected();
        }
        return false;
    }

    /** Connect to all instances of the given service type, and make sure
        that we listen and connect to any further ones that appear.
    */
    void connectAllServiceProviders(const std::string & serviceClass,
                                    const std::string & endpointName,
                                    bool local = true);

    /** Connect to a single named service. */
    void connectSingleServiceProvider(const std::string & service);

    /** Connect directly to a zeromq uri. */
    void connectUri(const std::string & zmqUri);

    size_t connectionCount() const
    {
        std::unique_lock<Lock> guard(connectionsLock);
        return connections.size();
    }

    /** Type of callback for a new connection. */
    typedef std::function<void (std::string)> ConnectionHandler;

    /** Callback that will be called when we get a new connection if
        onConnect() is not overridden. */
    ConnectionHandler connectHandler;

    /** Function that will be called when we make a new connection to a
        remote service provider.
    */
    virtual void onConnect(const std::string & source)
    {
        if (connectHandler)
            connectHandler(source);
    }

    /** Callback that will be called when we get a disconnection if
        onDisconnect() is not overridden. */
    ConnectionHandler disconnectHandler;

    /** Function that will be called when we lose a connection to a
        remote service provider.
    */
    virtual void onDisconnect(const std::string & source)
    {
        if (disconnectHandler)
            disconnectHandler(source);
    }

    /** Type of handler to override what we do when we get a message. */
    typedef std::function<void (std::string, std::vector<std::string>)>
    MessageHandler;

    /** Function to call when we get a message from a remote connection
        if handleMessage() is not overridden.
    */
    MessageHandler messageHandler;

    /** Handle a message from a remote service.  Source is the name of the
        service (this can be used to send something back).  Message is the
        content of the message.
    */
    virtual void handleMessage(const std::string & source,
                               const std::vector<std::string> & message)
    {
        if (messageHandler)
            messageHandler(source, message);
        else
            THROW(ZmqLogs::error) << "no message handler set" << std::endl;
    }


private:
    /** Are we connected? */
    bool connected;

    /** Common zeromq context for all connections. */
    std::shared_ptr<zmq::context_t> zmqContext;

    /** Number of times a particular change has occured. This is only meant for tests
     *  purposes.
     */
    std::map<int, uint32_t> changesCount;

    /** Configuration service from where we learn where to connect. */
    std::shared_ptr<ConfigurationService> config;

    /** Class of service we're connecting to. */
    std::string serviceClass;

    /** Name of the endpoint on the service we're connecting to. */
    std::string endpointName;

    /** Identity for our zeromq socket. */
    std::string identity;

    typedef ML::Spinlock Lock;

    /** Lock to be used when modifying the list of connections. */
    mutable Lock connectionsLock;

    /** List of currently connected connections. */
    typedef std::map<std::string, std::shared_ptr<ZmqNamedClientBusProxy> > ConnectionMap;
    ConnectionMap connections;

    /** Current watch on the list of service providers. */
    ConfigurationService::Watch serviceProvidersWatch;

    /** Are we currently in onServiceProvidersChanged? **/
    bool inProvidersChanged;
    ML::Spinlock providersChangedLock;

    /** Queue for defered onServiceProvidersChanged calls */
    typedef std::pair<std::string, bool> DeferedProvidersChanges;
    std::vector<DeferedProvidersChanges> deferedProvidersChanges;

    bool enterProvidersChanged(const std::string& path, bool local);
    void exitProvidersChanged();

    /** Callback that will be called when the list of service providers has changed. */
    void onServiceProvidersChanged(const std::string & path, bool local);

    struct OnConnectCallback;

    /** Call this to watch for a given service provider. */
    void watchServiceProvider(const std::string & name, const std::string & path, int shardIndex);

};



} // namespace Datacratic

#endif /* __service__zmq_endpoint_h__ */

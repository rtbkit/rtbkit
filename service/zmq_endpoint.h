/* zmq_endpoint.h                                                  -*- C++ -*-
   Jeremy Barnes, 25 September 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Endpoints for zeromq.
*/

#ifndef __service__zmq_endpoint_h__
#define __service__zmq_endpoint_h__

#include "named_endpoint.h"
#include "message_loop.h"
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
            throw ML::Exception("no fd for zeromq socket");
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
        else throw ML::Exception("need to override handleMessage");
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
            throw ML::Exception("no fd for zeromq socket");
        return res;
    }

    virtual bool poll() const
    {
        return getEvents(socket()).first;
    }

#if 0
    template<typename Arg, int Index>
    const Arg &
    getArg(const std::vector<zmq::message_t> & messages,
           const ML::InPosition<Arg, Index> * arg)
    {
        auto & m = messages.at(Index);
        ExcAssertEqual(m.size(), sizeof(Arg));
        return * reinterpret_cast<const Arg *>(msg.data());
    }
#endif

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
        else throw ML::Exception("handleMessage not done");
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
        if (message.size() != expectedSize)
            throw ML::Exception("unexpected message size in ZmqTypedMessageSink");

        int i = routable;
        if (message[i + 1] != messageTopic)
            throw ML::Exception("unexpected messake kind in ZmqTypedMessageSink");

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
        else
            throw ML::Exception("need to override handleTypedMessage or assign "
                                "to onMessage");
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
            throw ML::Exception("need to call ZmqNamedEndpoint::init() before "
                                "bind");

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
        else throw ML::Exception("need to override handleRawMessage or "
                                 "handleMessage");
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
        //cerr << "ZmqNamedClientBus got message " << message << endl;

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

#if 0
        cerr << "poll() returned " << poll() << endl;

        std::vector<std::string> msg
            = recvAllNonBlocking(socket());

        while (!msg.empty()) {
            cerr << "*** GOT FURTHER MESSAGE " << msg << endl;
            msg = recvAllNonBlocking(socket());
        }

        cerr << "poll() returned " << poll() << endl;
#endif
    }

    typedef std::function<void (std::vector<std::string>)>
    ClientMessageHandler;
    ClientMessageHandler clientMessageHandler;

    virtual void handleClientMessage(const std::vector<std::string> & message)
    {
        if (clientMessageHandler)
            clientMessageHandler(message);
        else {
            throw ML::Exception("need to assign to onClientMessage "
                                "or override handleClientMessage for message "
                                + message.at(1));
        }
#if 0
        using namespace std;
        cerr << "ZmqNamedClientBus handleClientMessage " << message << endl;
        throw ML::Exception("handleClientMessage");
#endif
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

    ZmqNamedProxy(std::shared_ptr<zmq::context_t> context);

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
            std::cerr << ("dropping message for " + endpointName + "\n");
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

    ZmqNamedClientBusProxy(std::shared_ptr<zmq::context_t> context)
        : ZmqNamedProxy(context), timeout(2.0)
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

                auto now = Date::now();
                auto end = now.plusSeconds(-timeout);
                //if(lastHeartbeat < end) {
                    //std::cerr << "no heartbeat for " << timeout << "s... should be disconnecting from " << connectedUri << std::endl;
                    //disconnect();
                //}
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
            throw ML::Exception("need to override on messageHandler or handleMessage");
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
            throw ML::Exception("attempt to deliver " + topic + " message to unknown client "
                                + recipient);
        }
        it->second->sendMessage(topic, std::forward<Args>(args)...);
    }

    /** Connect to all instances of the given service type, and make sure
        that we listen and connect to any further ones that appear.
    */
    void connectAllServiceProviders(const std::string & serviceClass,
                                    const std::string & endpointName,
                                    bool local = true)
    {
        if (connected)
            throw ML::Exception("already connected to service providers");

        this->serviceClass = serviceClass;
        this->endpointName = endpointName;

        serviceProvidersWatch.init([=] (const std::string & path,
                                        ConfigurationService::ChangeType change)
                                   {
                                       ++changesCount[change];
                                       onServiceProvidersChanged("serviceClass/" + serviceClass, local);
                                   });

        onServiceProvidersChanged("serviceClass/" + serviceClass, local);
        connected = true;
    }


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
            throw ML::Exception("need to override on messageHandler or handleMessage");
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
    bool inProvidersChanged ;
    /** Queue of operations to perform asynchronously from our own thread. */

    /** Callback that will be called when the list of service providers has changed. */
    void onServiceProvidersChanged(const std::string & path, bool local)
    {
        using namespace std;
        // this function is invoked upon a disconnect
        if( inProvidersChanged)
        {
            std::cerr << "!!!Already in service providers changed - bailing out "
                << std::endl;
            return ;
        }
        inProvidersChanged = true;
        //cerr << "onServiceProvidersChanged(" << path << ")" << endl;

        // The list of service providers has changed

        vector<string> children
            = config->getChildren(path, serviceProvidersWatch);
        for (auto c: children) {
            Json::Value value = config->getJson(path + "/" + c);
            std::string name = value["serviceName"].asString();
            std::string path = value["servicePath"].asString();

            std::string location = value["serviceLocation"].asString();
            if (local && location != config->currentLocation) {
                std::cerr << "dropping " << location << " != " << config->currentLocation << std::endl;
                continue;
            }

            watchServiceProvider(name, path);
        }

        // deleting the connection could trigger a callback which is a bad idea
        // while we're holding the connections lock. So instead we move all the
        // connections to be deleted to a temp map which we'll wipe once the
        // lock is released.
        ConnectionMap pendingDisconnects;
        {
            std::unique_lock<Lock> guard(connectionsLock);

            // Services that are no longer in zookeeper are considered to be
            // disconnected so remove them from our connection map.
            for (auto& conn : connections) {
                auto it = find(children.begin(), children.end(), conn.first);
                if (it != children.end()) continue;

                // Erasing from connections in this loop would invalidate our
                // iterator so defer until we're done with the connections map.
                removeSource(conn.second.get());
                pendingDisconnects[conn.first] = std::move(conn.second);
            }

            for (const auto& conn : pendingDisconnects)
                connections.erase(conn.first);
        }
        // We're no longer holding the lock so any delayed. Time to really
        // disconnect and trigger the callbacks.
        pendingDisconnects.clear();
        inProvidersChanged = false;
    }

    /** Encapsulates a lock-free state machine that manages the logic of the on
        config callback. The problem being solved is that the onConnect callback
        should not call the user's callback while we're holding the connection
        lock but should do it when the lock is released.

        The lock-free state machine guarantees that no callbacks are lost and
        that no callbacks will be triggered before a call to release is made.
        The overhead of this class amounts to at most 2 CAS when the callback is
        triggered; one if there's no contention.

        Note that this isn't an ideal solution to this problem because we really
        should get rid of the locks when manipulating these events.

        \todo could be generalized if we need this pattern elsewhere.
    */
    struct OnConnectCallback
    {
        OnConnectCallback(const ConnectionHandler& fn, std::string name) :
            fn(fn), name(name), state(DEFER)
        {}

        /** Should ONLY be called AFTER the lock is released. */
        void release()
        {
            State old = state;

            ExcAssertNotEqual(old, CALL);

            // If the callback wasn't triggered while we were holding the lock
            // then trigger it the next time we see it.
            if (old == DEFER && ML::cmp_xchg(state, old, CALL)) return;

            ExcAssertEqual(old, DEFERRED);
            fn(name);
        }

        void operator() (std::string)
        {
            State old = state;
            ExcAssertNotEqual(old, DEFERRED);

            // If we're still in the locked section then trigger the callback
            // when release is called.
            if (old == DEFER && ML::cmp_xchg(state, old, DEFERRED)) return;

            // We're out of the locked section so just trigger the callback.
            ExcAssertEqual(old, CALL);
            fn(name);
        }

    private:

        ConnectionHandler fn;
        std::string name;

        enum State {
            DEFER,    // We're holding the lock so defer an incoming callback.
            DEFERRED, // We were called while holding the lock.
            CALL      // We were not called while holding the lock.
        } state;
    };

    /** Call this to watch for a given service provider. */
    void watchServiceProvider(const std::string & name, const std::string & path)
    {
        // Protects the connections map... I think.
        std::unique_lock<Lock> guard(connectionsLock);

        auto & c = connections[name];
        //ML::backtrace();

        // already connected
        if (c) 
        {
            std::cerr << "watchServiceProvider: name " << name << " already connected " << std::endl;
            return;
        }
        std::cerr << "watchServiceProvider: name " << name << " not already connected " << std::endl;
        
        try {
            auto newClient = std::make_shared<ZmqNamedClientBusProxy>(zmqContext);
            newClient->init(config, identity);

            // The connect call below could trigger this callback while we're
            // holding the connectionsLock which is a big no-no. This fancy
            // wrapper ensures that it's only called after we call its release
            // function.
            if (connectHandler)
                newClient->connectHandler = OnConnectCallback(connectHandler, name);

            newClient->disconnectHandler = [=] (std::string s)
                {
                    // TODO: chain in so that we know it's not around any more
                    this->onDisconnect(s);
                };

            newClient->connect(path + "/" + endpointName);
            newClient->messageHandler = [=] (const std::vector<std::string> & msg)
                {
                    this->handleMessage(name, msg);
                };
            //newClient->debug(true);

            c = std::move(newClient);

            // Add it to our message loop so that it can process messages
            addSource("ZmqMultipleNamedClientBusProxy child " + name, c);

            guard.unlock();
            if (connectHandler)
                c->connectHandler.target<OnConnectCallback>()->release();

        } catch (...) {
            // Avoid triggering the disconnect callbacks while holding the
            // connectionsLock by defering the delete of the connection until
            // we've manually released the lock.
            ConnectionMap::mapped_type conn(std::move(connections[name]));
            connections.erase(name);

            guard.unlock();
            // conn is a unique_ptr so it gets destroyed here.
            throw;
        }
    }

};



} // namespace Datacratic

#endif /* __service__zmq_endpoint_h__ */

/* zmq_named_pub_sub.h                                             -*- C++ -*-
   Jeremy Barnes, 8 January 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.
   
   Named publish/subscribe endpoint.
*/

#pragma once

#include "zmq_endpoint.h"
#include "typed_message_channel.h"
#include <sys/utsname.h>
#include "jml/arch/backtrace.h"

namespace Datacratic {



/*****************************************************************************/
/* ZMQ NAMED PUBLISHER                                                       */
/*****************************************************************************/

/** Class that publishes messages.  It also knows what is connected to it.
 */
struct ZmqNamedPublisher: public MessageLoop {

    ZmqNamedPublisher(std::shared_ptr<zmq::context_t> context,
                      int messageBufferSize = 65536)
        : publishEndpoint(context),
          publishQueue(messageBufferSize)
    {
    }

    virtual ~ZmqNamedPublisher()
    {
        shutdown();
    }

    void init(std::shared_ptr<ConfigurationService> config,
              const std::string & endpointName,
              const std::string & identity = "")
    {
        using namespace std;

        // Initialize the publisher endpoint
        publishEndpoint.init(config, ZMQ_XPUB, endpointName);

        // Called when we queue up a message to be published
        publishQueue.onEvent = [=] (std::vector<zmq::message_t> && message)
            {
                using namespace std;
                //cerr << "popped message to publish" << endl;
                publishEndpoint.sendMessage(std::move(message));
            };

        // Called when there is a new subscription.
        // The first byte is 1 (subscription) or 0 (unsubscription).
        // The rest of the message is the filter for the subscription
        auto doPublishMessage = [=] (const std::vector<std::string> & msg)
            {
#if 0
                cerr << "got publish subscription" << msg << endl;
                cerr << "msg.size() = " << msg.size() << endl;
                cerr << "msg[0].size() = " << msg[0].size() << endl;
                cerr << "msg[0][0] = " << (int)msg[0][0] << endl;
#endif
            };

        publishEndpoint.messageHandler = doPublishMessage;

        // Hook everything into the message loop
        addSource("ZmqNamedPublisher::publishEndpoint", publishEndpoint);
        addSource("ZmqNamedPublisher::publishQueue", publishQueue);

    }

    std::string bindTcp(PortRange const & portRange = PortRange(), std::string host = "")
    {
        return publishEndpoint.bindTcp(portRange, host);
    }

    void shutdown()
    {
        MessageLoop::shutdown();
        publishEndpoint.shutdown();
        //publishEndpointMonitor.shutdown();
    }

    //std::vector<std::string> getSubscribers()
    //{
    //}

    template<typename Head, typename... Tail>
    void encodeAll(std::vector<zmq::message_t> & messages,
                   Head head,
                   Tail&&... tail)
    {
        messages.emplace_back(std::move(encodeMessage(head)));
        encodeAll(messages, std::forward<Tail>(tail)...);
    }

    // Vectors treated specially... they are copied
    template<typename... Tail>
    void encodeAll(std::vector<zmq::message_t> & messages,
                   const std::vector<std::string> & head,
                   Tail&&... tail)
    {
        for (auto & m: head)
            messages.emplace_back(std::move(encodeMessage(m)));
        encodeAll(messages, std::forward<Tail>(tail)...);
    }

    void encodeAll(std::vector<zmq::message_t> & messages)
    {
    }

    template<typename... Args>
    void publish(const std::string & channel, Args&&... args)
    {
        std::vector<zmq::message_t> messages;
        messages.reserve(sizeof...(Args) + 1);
        
        encodeAll(messages, channel,
                  std::forward<Args>(args)...);
        publishQueue.push(messages);
    }

private:
    /// Zeromq endpoint on which messages are published
    ZmqNamedEndpoint publishEndpoint;

    /// Queue of things to be published
    TypedMessageSink<std::vector<zmq::message_t> > publishQueue;
};


/*****************************************************************************/
/* ENDPOINT CONNECTOR                                                        */
/*****************************************************************************/

/** This class keeps track of watching and connecting to a particular
    endpoint.

    Specifically, an endpoint is in the configuration service under the
    path

    /serviceName/endpoint

    It can have one or more entries under that path for different versions
    of the endpoint:

    /serviceName/endpoint/tcp
    /serviceName/endpoint/ipc
    /serviceName/endpoint/shm

    etc.

    This class will:
    1.  If there are no entries under the endpoint, then it will wait until
        one or more appear in the object.
    2.  Once there is one or more endpoints, it will attempt to connect to
        each of them until it finds one that works.
    3.  If there is a disconnection from an endpoint, it will attempt to
        re-establish a connection.
    4.  If the configuration string of an endpoint changes, it will attempt
        a re-connection.

    It is designed to integrate into an existing message loop so that the
    notifications will be processed in the same thread as the other messages
    and locking is not required.
*/

struct EndpointConnector : public MessageLoop {

    EndpointConnector()
        : changes(32)
    {
    }

    void init(std::shared_ptr<ConfigurationService> config)
    {
        this->config = config;

        changes.onEvent
            = std::bind(&EndpointConnector::handleEndpointChange,
                        this,
                        std::placeholders::_1);

        addSource("EndpointConnector::changes", changes);
    }

    void watchEndpoint(const std::string & endpointPath)
    {
        std::unique_lock<Lock> guard(lock);

        using namespace std;
        //cerr << "watching endpoint " << endpointPath << endl;

        auto & entry = endpoints[endpointPath];

        if (!entry.watch) {
            // First time that we watch this service; set it up

            //cerr << "=============== initializing watch for " << endpointPath << endl;

            entry.watch.init([=] (const std::string & path,
                                  ConfigurationService::ChangeType change)
                             {
                                 using namespace std;
                                 //cerr << "endpoint changed path " << path << " " << this << endl;

                                 {
                                     std::unique_lock<Lock> guard(lock);
                                     if (endpoints.count(endpointPath)) {
                                         auto & entry = endpoints[endpointPath];
                                         entry.watchIsSet = false;
                                     }
                                 }
                                 
                                 changes.push(endpointPath);
                             });

            changes.push(endpointPath);
        }
    }
    
    /** Stop watching the given endpoint, which should have already been
        watched via watchEndpoint.
    */
    void unwatchEndpoint(const std::string & endpoint,
                         bool forceDisconnect)
    {
        throw ML::Exception("unwatchEndpoint not done");
    }

    std::function<bool (const std::string & endpointPath,
                        const std::string & entry,
                        const Json::Value & params)> connectHandler;
    
    /** This method will be called when a connection is needed to a service.
        It should call the onDone function with a boolean indicating whether
        or not a connection was established.

        Return code:
        false = connection failed
        true = connection succeeded

        TODO: make this work with asynchronous connections
    */
    virtual bool connect(const std::string & endpointPath,
                         const std::string & entryName,
                         const Json::Value & params)
    {
        if (connectHandler)
            return connectHandler(endpointPath, entryName, params);
        throw ML::Exception("no connect override");
    }

    /** Method to be called back when there is a disconnection. */
    void handleDisconnection(const std::string & endpointPath,
                             const std::string & entryName)
    {
        std::unique_lock<Lock> guard(lock);

        if (!endpoints.count(endpointPath))
            return;

        auto & entry = endpoints[endpointPath];

        ExcAssertEqual(entry.connectedTo, entryName);
        entry.connectedTo = "";

        // And we attempt to reconnect
        // Note that we can't push endpointPath on changes
        changes.push(endpointPath);
    }

private:
    typedef std::mutex Lock;
    mutable Lock lock;

    /** We put zookeeper messages on this queue so that they get handled in
        the message loop thread and don't cause locking problems.
    */
    TypedMessageSink<std::string> changes;

    struct Entry {
        Entry()
            : watchIsSet(false)
        {
        }

        ConfigurationService::Watch watch;
        std::string connectedTo;
        bool watchIsSet;
    };

    std::map<std::string, Entry> endpoints;

    std::shared_ptr<ConfigurationService> config;

    void handleEndpointChange(const std::string & endpointPath)
    {
        using namespace std;

        //cerr << "handleEndpointChange " << endpointPath << endl;

        std::unique_lock<Lock> guard(lock);

        if (!endpoints.count(endpointPath))
            return;

        auto & entry = endpoints[endpointPath];

        //cerr << "handling service class change for " << endpointPath << endl;

        vector<string> children;
        if (entry.watchIsSet)
            children = config->getChildren(endpointPath);
        else children = config->getChildren(endpointPath, entry.watch);
        entry.watchIsSet = true;

        //cerr << "children = " << children << endl;

        // If we're connected, look for a change in the endpoints
        if (!entry.connectedTo.empty()) {

            // Does our connected node still exist?
            if (std::find(children.begin(), children.end(), entry.connectedTo)
                == children.end()) {
                // Node disappeared; we need to disconnect
                guard.unlock();
                handleDisconnection(endpointPath, entry.connectedTo);
                return;  // handleDisconnection will call into us recursively
            }
            else {
                // Node is still there
                // TODO: check for change in value
                return;
            }
        }

        guard.unlock();

        // If we got here, we're not connected
        for (auto & c: children) {
            Json::Value cfg = config->getJson(endpointPath + "/" + c);
            if (connect(endpointPath, c, cfg)) {
                notifyConnectionStatus(endpointPath, c, true);
                return;
            }
        }

        cerr << "warning: could not connect to " << endpointPath << " immediately"
             << endl;
    }

    void notifyConnectionStatus(const std::string & endpointPath,
                                const std::string & entryName,
                                bool status)
    {
        std::unique_lock<Lock> guard(lock);

        if (!endpoints.count(endpointPath))
            return;

        auto & entry = endpoints[endpointPath];

        if (status) {
            if (!entry.connectedTo.empty()) {
                // TODO
                throw ML::Exception("TODO: handle connection with one already "
                                    "active");
            }
            entry.connectedTo = entryName;
            // TODO: watch config
        }
        else {
            if (entry.connectedTo.empty())
                return;
            if (entry.connectedTo != entryName) {
                throw ML::Exception("TODO: handle disconnection from non-active");
            }
            entry.connectedTo = "";
        }
    }
};


/*****************************************************************************/
/* ZMQ NAMED SOCKET                                                      */
/*****************************************************************************/

/** Active socket that attempts to connect into an endpoint. */

struct ZmqNamedSocket: public MessageLoop {

    enum ConnectionState {
        NO_CONNECTION,   ///< No connection was requested
        CONNECTING,      ///< Connection is attempting to connect
        CONNECTED,       ///< Connection is connected
        DISCONNECTED     ///< Connection was connected but has disconnected
    };

    ZmqNamedSocket(zmq::context_t & context, int type)
        : context(&context),
          socketType(type),
          connectionState(NO_CONNECTION)
    {
        //using namespace std;
        //cerr << "created zmqNamedSocket at " << this << endl;
    }

    virtual ~ZmqNamedSocket()
    {
        shutdown();
    }

    /** Initialization.  Can only be called once and is not protected from
        multiple threads.
    */
    void init(std::shared_ptr<ConfigurationService> config)
    {
        if (socket)
            throw ML::Exception("socket already initialized");
        socket.reset(new zmq::socket_t(*context, socketType));
        
        using namespace std;

        connector.connectHandler
            = std::bind(&ZmqNamedSocket::doConnect,
                        this,
                        std::placeholders::_1,
                        std::placeholders::_2,
                        std::placeholders::_3);

        connector.init(config);

        addSource("ZmqNamedSocket::connector", connector);
        addSource("ZmqNamedSocket::socket",
                  std::make_shared<ZmqBinaryEventSource>
                  (*socket, [=] (std::vector<zmq::message_t> && message)
                   {
                       this->handleMessage(std::move(message));
                   }));
    }

    void shutdown()
    {
        MessageLoop::shutdown();

        if (!socket)
            return;

        disconnect();
        connector.shutdown();

        socket->tryDisconnect(this->connectedAddress);
        socket.reset();
    }

    void connectToEndpoint(const std::string & endpointPath)
    {
        if (connectedEndpoint != "")
            throw ML::Exception("attempt to connect a ZmqNamedSocket "
                                "to an enpoint that is already connected");
     
        this->connectedEndpoint = connectedEndpoint;
        this->connectionState = CONNECTING;

        // No lock needed as the connector has its own lock

        // Tell the connector to watch the endpoint.  When an entry pops
        // up, it will connect to it.
        connector.watchEndpoint(endpointPath);
    }

    /// Disconnect from the connected endpoint
    void disconnect()
    {
        std::unique_lock<Lock> guard(lock);
        ExcAssert(socket);
        socket->tryDisconnect(connectedAddress);
        this->connectedEndpoint = "";
        this->connectedAddress = "";
        this->connectionState = DISCONNECTED;
        //connector.disconnect();
    }

    /// Endpoint we're configured to connect to
    std::string getConnectedEndpoint() const
    {
        std::unique_lock<Lock> guard(lock);
        return connectedEndpoint;
    }

    /// Address we're currently connecting/connected to
    std::string getConnectedAddress() const
    {
        std::unique_lock<Lock> guard(lock);
        return connectedAddress;
    }

    /// Current state of the connection
    ConnectionState getConnectionState() const
    {
        // No lock needed as performed atomically
        return connectionState;
    }

    /** Callback that will be called when a message is received if
        handlerMessage is not overridden.
    */
    typedef std::function<void (std::vector<zmq::message_t> &&)> MessageHandler;
    MessageHandler messageHandler;

    /** Function called when a message is received on the socket.  Default action
        is to call messageHandler.  Can be overridden or messageHandler can be
        assigned to.
    */
    virtual void handleMessage(std::vector<zmq::message_t> && message)
    {
        using namespace std;
        //cerr << "named socket got message " << message.at(0).toString()
        //     << endl;
        if (messageHandler)
            messageHandler(std::move(message));
        else throw ML::Exception("need to override either messageHandler "
                                 "or handleMessage");
    }

    /** Call the given function synchronously with locking set up such that no
        other socket operation can occur at the same time.  This allows for
        socket operations to be performed from any thread.  Note that the
        operation must not call any methods on this object or a locking
        error will occur.

        Function signature must match

        void fn(zmq::socket_t &)
    */
    template<typename Fn>
    void performSocketOpSync(Fn fn)
    {
        std::unique_lock<Lock> guard(lock);
        fn(*socket);
    }

    /** Send the given message synchronously. */
    void sendSync(std::vector<zmq::message_t> && message)
    {
        std::unique_lock<Lock> guard(lock);
        for (unsigned i = 0;  i < message.size();  ++i) {
            socket->send(message[i], i == message.size() - 1
                         ? 0 : ZMQ_SNDMORE);
        }
    }

private:
    // This lock is used to allow the synchronous methods to work without
    // needing ping-pong with the message loop.  Normally it should be
    // uncontended.
    typedef std::mutex Lock;
    mutable Lock lock;

    /** Function called by the endpoint connector to actually connect to
        an endpoint.  This may be called when we are already connected,
        which means that we should disconnect from the old endpoint and
        connect to the new one.
    */
    virtual bool doConnect(const std::string & endpointPath,
                           const std::string & entryName,
                           const Json::Value & epConfig)
    {
        using namespace std;

        //cerr << "   ((((((( doConnect for " << endpointPath << " " << entryName
        //     << endl;

        std::unique_lock<Lock> guard(lock);

        for (auto & entry: epConfig) {

            //cerr << "entry is " << entry << endl;

            if (!entry.isMember("zmqConnectUri"))
                continue;

            auto hs = entry["transports"][0]["hostScope"];
            if (!hs)
                continue;

            string hostScope = hs.asString();
            if (hs != "*") {
                utsname name;
                if (uname(&name))
                    throw ML::Exception(errno, "uname");
                if (hostScope != name.nodename)
                    continue;  // wrong host scope
            }

            string uri = entry["zmqConnectUri"].asString();

            if (connectedAddress != "") {
                // Already connected...
                if (connectedAddress == uri)
                    return true;
                else {
                    // Need to disconnect from the current address and connect to the new one
                    //cerr << "connectedAddress = " << connectedAddress << " uri = " << uri << endl;
                    socket->tryDisconnect(connectedAddress);
                    //throw ML::Exception("need to handle disconnect and reconnect to different "
                    //                    "address");
                }
            }
            
            connectedAddress = uri;
            connectedEndpointPath = endpointPath;
            connectedEntryName = entryName;
            socket->connect(uri);

            //cerr << "connection in progress to " << uri << endl;
            connectionState = CONNECTED;
            return true;
        }
        
        return false;
    }

    /// Zmq context we're working with
    zmq::context_t * context;

    /// Socket type to create; one of the ZMQ_ constants
    int socketType;
    
    /// Endpoint we're configured to connect to
    std::string connectedEndpoint;

    /// Address we're currently connecting/connected to
    std::string connectedAddress;

    /// Path of the endpoint we're currently connected to
    std::string connectedEndpointPath;

    /// Name of the entry under the endpoint name which we're connected to
    std::string connectedEntryName;

    /// File descriptor that our connection is on.  Mostly used for testing.
    int connectedFd;

    /// Current state of the connection
    ConnectionState connectionState;

    /// Handles actually connecting to the socket
    EndpointConnector connector;

    /// Socket that we connect
    std::unique_ptr<zmq::socket_t> socket;
};


/*****************************************************************************/
/* ZMQ NAMED SUBSCRIBER                                                      */
/*****************************************************************************/

/** A subscriber class built on top of the ZmqNamedSocket. */

struct ZmqNamedSubscriber: public ZmqNamedSocket {

    ZmqNamedSubscriber(zmq::context_t & context)
        : ZmqNamedSocket(context, ZMQ_SUB)
    {
    }

    /// Subscribe to the given message prefix
    void subscribe(const std::string & prefix)
    {
        auto doSubscribe = [&] (zmq::socket_t & socket)
            {
                subscribeChannel(socket, prefix);
            };
        
        performSocketOpSync(doSubscribe);
    }

    /// Subscribe to the given message prefix
    void subscribe(const std::vector<std::string> & prefixes)
    {
        auto doSubscribe = [&] (zmq::socket_t & socket)
            {
                for (const auto& prefix : prefixes)
                    subscribeChannel(socket, prefix);
            };

        performSocketOpSync(doSubscribe);
    }

};




/*****************************************************************************/
/* SERVICE PROVIDER WATCHER                                                  */
/*****************************************************************************/

/** This class keeps track of watching for service providers and making sure
    that there is a connection to each of them.

    It performs the following actions:
    * Watches the configuration node for a given service class
    * If a new service provider appears, calls a connect callback
    * If a service provider disappears, calls a disconnect callback

    It is designed to integrate into an existing message loop so that the
    notifications will be processed in the same thread as the other messages
    and locking is not required.
*/

struct ServiceProviderWatcher: public MessageLoop {

    ServiceProviderWatcher()
        : currentToken(1), changes(128)
    {
    }

    ~ServiceProviderWatcher()
    {
        using namespace std;
        //cerr << "shutting down service provider watcher" << endl;
        shutdown();
        //cerr << "done" << endl;
    }

    void init(std::shared_ptr<ConfigurationService> config)
    {
        this->config = config;

        changes.onEvent
            = std::bind(&ServiceProviderWatcher::handleServiceClassChange,
                        this,
                        std::placeholders::_1);

        addSource("ServiceProviderWatcher::changes", changes);
    }

    /** Type of a function that will be called when there is a change on
        the service providers.
    */
    typedef std::function<void (std::string path, bool)> ChangeHandler;
    ChangeHandler changeHandler;

    /** Notify whenever an instance of the given service class goes down or
        up.

        Whenever there is a change, the onChange handler will be called 
        from within the owning message loop.

        Returns a token that can be used to unregister the watch.
    */
    uint64_t watchServiceClass(const std::string & serviceClass,
                               ChangeHandler onChange)
    {
        using namespace std;
        //cerr << "watching service class " << serviceClass << endl;

        // Allocate a token, then push the message on to the thread to be
        // processed.
        uint64_t token = __sync_fetch_and_add(&currentToken, 1);

        std::unique_lock<Lock> guard(lock);

        auto & entry = serviceClasses[serviceClass];

        //cerr << "entry.watch = " << entry.watch << endl;

        if (!entry.watch) {
            // First time that we watch this service; set it up

            entry.watch.init([=] (const std::string & path,
                                  ConfigurationService::ChangeType change)
                             {
                                 using namespace std;
                                 //cerr << "changed path " << path << endl;

                                 changes.push(serviceClass);
                             });

            changes.push(serviceClass);
        }

        entry.entries[token].onChange = onChange;
        
        return token;
    }

    /** Stop watching the given service class. */
    void unwatchServiceClass(const std::string & serviceClass,
                             uint64_t token)
    {
        // TODO: synchronous?  Yes, as otherwise we can't guarantee anything
        // about keeping the entries valid

        throw ML::Exception("unwatchServiceClass: not done");

#if 0        
        int done = 0;

        messages.push([&] () { this->doUnwatchServiceClass(serviceClass, token); done = 1; futex_wake(done); });

        while (!done)
            futex_wait(done, 0);
#endif
    }

private:
    typedef std::mutex Lock;
    mutable Lock lock;
    
    // Shared variable for the tokens to give out to allow unwatching
    uint64_t currentToken;

    struct WatchEntry {
        ChangeHandler onChange;
    };

    struct ServiceClassEntry {
        /// Watch on the service providers node
        ConfigurationService::Watch watch;
        std::map<uint64_t, WatchEntry> entries; 

        /** Current set of known children of the service nodes.  Used to
            perform a diff to detect changes.
        */
        std::vector<std::string> knownChildren;
    };
    
    /// Map from service classes to list of watches
    std::map<std::string, ServiceClassEntry> serviceClasses;

    std::shared_ptr<ConfigurationService> config;

    /** Deal with a change in the service class node. */
    void handleServiceClassChange(const std::string & serviceClass)
    {
        using namespace std;

        //cerr << "handleServiceClassChange " << serviceClass << endl;
        
        std::unique_lock<Lock> guard(lock);

        auto & entry = serviceClasses[serviceClass];

        //cerr << "handling service class change for " << serviceClass << endl;

        vector<string> children
            = config->getChildren("serviceClass/" + serviceClass,
                                  entry.watch);

        //cerr << "children = " << children << endl;

        std::sort(children.begin(), children.end());

        auto & knownChildren = entry.knownChildren;

        // Perform a diff between previously and currently known children
        vector<string> addedChildren, deletedChildren;
        std::set_difference(children.begin(), children.end(),
                            knownChildren.begin(), knownChildren.end(),
                            std::back_inserter(addedChildren));
        std::set_difference(knownChildren.begin(), knownChildren.end(),
                            children.begin(), children.end(),
                            std::back_inserter(deletedChildren));

        knownChildren.swap(children);

        guard.unlock();

        for (auto & c: addedChildren) {
            for (auto & e: entry.entries) {
                e.second.onChange("serviceClass/" + serviceClass + "/" + c,
                                  true);
            }
        }

        for (auto & c: deletedChildren) {
            for (auto & e: entry.entries) {
                e.second.onChange("serviceClass/" + serviceClass + "/" + c,
                                  false);
            }
        }
    }

    /** We put zookeeper messages on this queue so that they get handled in
        the message loop thread and don't cause locking problems.
    */
    TypedMessageSink<std::string> changes;
};


/*****************************************************************************/
/* ZMQ NAMED MULTIPLE SUBSCRIBER                                             */
/*****************************************************************************/

/** Counterpart to the ZmqNamedPublisher.  It subscribes to all publishers
    of a given class.
*/

struct ZmqNamedMultipleSubscriber: public MessageLoop {

    ZmqNamedMultipleSubscriber(std::shared_ptr<zmq::context_t> context)
        : context(context)
    {
        needsPoll = true;
    }

    ~ZmqNamedMultipleSubscriber()
    {
        shutdown();
    }

    void init(std::shared_ptr<ConfigurationService> config)
    {
        this->config = config;

        serviceWatcher.init(config);

        addSource("ZmqNamedMultipleSubscriber::serviceWatcher", serviceWatcher);

        //debug(true);
    }

    void shutdown()
    {
        MessageLoop::shutdown();

        for (auto & sub: subscribers)
            sub.second->shutdown();
        subscribers.clear();
    }

    void connectAllServiceProviders(const std::string & serviceClass,
                                    const std::string & endpointName,
                                    const std::vector<std::string> & prefixes
                                    = std::vector<std::string>(),
                                    std::function<bool (std::string)> filter
                                    = nullptr,
                                    bool local = true)
    {
        auto onServiceChange = [=] (const std::string & service,
                                    bool created)
            {
                using namespace std;
                //cerr << "onServiceChange " << serviceClass << " " << endpointName
                //<< " " << service << " created " << created << endl;

                if (filter && !filter(service))
                    return;

                if (created)
                    connectService(serviceClass, service, endpointName, local);
                else
                    disconnectService(serviceClass, service, endpointName);
            };
        
        this->prefixes = prefixes;
        serviceWatcher.watchServiceClass(serviceClass, onServiceChange);
    }
    
    /** Callback that will be called when a message is received if
        handlerMessage is not overridden.
    */
    typedef std::function<void (std::vector<zmq::message_t> &&)> MessageHandler;
    MessageHandler messageHandler;

    /** Function called when a message is received on the socket.  Default action
        is to call messageHandler.  Can be overridden or messageHandler can be
        assigned to.
    */
    virtual void handleMessage(std::vector<zmq::message_t> && message)
    {
        if (messageHandler)
            messageHandler(std::move(message));
        else throw ML::Exception("need to override either messageHandler "
                                 "or handleMessage");
    }

    /** Connect to the given service. */
    void connectService(std::string serviceClass, std::string service,
                        std::string endpointName,
                        bool local = true)
    {
        using namespace std;

        Json::Value value = config->getJson(service);

        std::string location = value["serviceLocation"].asString();
        if(local && location != config->currentLocation) {
            std::cerr << "dropping " << location
                      << " != " << config->currentLocation << std::endl;
            return;
        }

        std::unique_lock<Lock> guard(lock);

        SubscriberMap::const_iterator found = subscribers.find(service);
        if(found != subscribers.end())
        {
           if(found->second->getConnectionState() == ZmqNamedSocket::CONNECTED)
           {
               std::cerr << "Already connected to service " << service << std::endl;
               return;
           }
           else
           {
             std::cerr << "we already had a connection entry to service " << service <<" - reuse " << std::endl;
             std::string path = value["servicePath"].asString();
             found->second->connectToEndpoint(path); 
             return ;
           }
        } 
         
        std::unique_ptr<ZmqNamedSubscriber> sub
            (new ZmqNamedSubscriber(*context));

        // Set up to handle the message
        sub->messageHandler = [=] (std::vector<zmq::message_t> && msg)
            {
                //cerr << "SUB MESSAGE HANDLER " << this << endl;
                this->handleMessage(std::move(msg));
            };
                                        
        sub->init(config);

        // TODO: put a watch in to reconnect if this changes
        std::string path = value["servicePath"].asString();

        //cerr << "(((((((((((( connecting to service " << service
        //     << " at endpoint " << path + "/" + endpointName << endl;
        
        //cerr << "+-+-+-+-+- connecting to endpoint " << path + "/" + endpointName << endl;
        //cerr << config->getChildren(path + "/" + endpointName) << endl;

        if (!prefixes.empty())
            sub->subscribe(prefixes);
        else sub->subscribe(""); // Subscribe to everything.

        sub->connectToEndpoint(path + "/" + endpointName);
        addSource(service, *sub);
        subscribers[service] = std::move(sub);
    }

    /** Disconnect from the given service. */
    void disconnectService(std::string serviceClass, std::string service,
                           std::string endpointName)
    {
        using namespace std;
        cerr << "need to disconenct from " << serviceClass << " "
             << service << " " << endpointName << endl;
//        cerr << "aborting as disconnect not done yet" << endl;
        SubscriberMap::const_iterator found = subscribers.find(service);
        if(found != subscribers.end())
        {
           found->second->disconnect();
        } 
        //abort();
    }

    /** This is responsible for watching the service providers and connecting
        to new ones when they pop up.
    */
    ServiceProviderWatcher serviceWatcher;
             
    std::shared_ptr<zmq::context_t> context;

    typedef std::mutex Lock;
    mutable Lock lock;

    /** Map of service name to subscribers */
    typedef std::map<std::string, std::unique_ptr<ZmqNamedSubscriber> > SubscriberMap; 
    SubscriberMap subscribers;

    std::shared_ptr<ConfigurationService> config;

    std::vector<std::string> prefixes;
};


} // namespace Datacratic

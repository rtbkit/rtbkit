/* zmq_endpoint.cc
   Jeremy Barnes, 9 November 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

*/

#include "zmq_endpoint.h"
#include "jml/utils/smart_ptr_utils.h"
#include <sys/utsname.h>
#include <thread>
#include "jml/arch/timers.h"
#include "jml/arch/info.h"

using namespace std;


namespace Datacratic {

/******************************************************************************/
/* ZMQ LOGS                                                                   */
/******************************************************************************/

Logging::Category ZmqLogs::print("ZMQ");
Logging::Category ZmqLogs::error("ZMQ Error", ZmqLogs::print);
Logging::Category ZmqLogs::trace("ZMQ Trace", ZmqLogs::print);



/*****************************************************************************/
/* ZMQ EVENT SOURCE                                                          */
/*****************************************************************************/

ZmqEventSource::
ZmqEventSource()
    : socket_(0), socketLock_(nullptr)
{
    needsPoll = false;
}

ZmqEventSource::
ZmqEventSource(zmq::socket_t & socket, SocketLock * socketLock)
    : socket_(&socket), socketLock_(socketLock)
{
    needsPoll = false;
    updateEvents();
}

void
ZmqEventSource::
init(zmq::socket_t & socket, SocketLock * socketLock)
{
    socket_ = &socket;
    socketLock_ = socketLock;
    needsPoll = false;
    updateEvents();
}

int
ZmqEventSource::
selectFd() const
{
    int res = -1;
    size_t resSize = sizeof(int);
    socket().getsockopt(ZMQ_FD, &res, &resSize);
    if (res == -1)
        THROW(ZmqLogs::error) << "no fd for zeromq socket" << endl;
    return res;
}

bool
ZmqEventSource::
poll() const
{
    if (currentEvents & ZMQ_POLLIN)
        return true;

    std::unique_lock<SocketLock> guard;

    if (socketLock_)
        guard = std::unique_lock<SocketLock>(*socketLock_);

    updateEvents();

    return currentEvents & ZMQ_POLLIN;
}

void
ZmqEventSource::
updateEvents() const
{
    size_t events_size = sizeof(currentEvents);
    socket().getsockopt(ZMQ_EVENTS, &currentEvents, &events_size);
}

bool
ZmqEventSource::
processOne()
{
    using namespace std;
    if (debug_)
        cerr << "called processOne on " << this << ", poll = " << poll() << endl;

    if (!poll())
        return false;

    std::vector<std::string> msg;

    // We process all events, as otherwise the select fd can't be guaranteed to wake us up
    for (;;) {
        {
            std::unique_lock<SocketLock> guard;
            if (socketLock_)
                guard = std::unique_lock<SocketLock>(*socketLock_);

            msg = recvAllNonBlocking(socket());

            if (msg.empty()) {
                if (currentEvents & ZMQ_POLLIN)
                    throw ML::Exception("empty message with currentEvents");
                return false;  // no more events
            }

            updateEvents();
        }

        if (debug_)
            cerr << "got message of length " << msg.size() << endl;
        handleMessage(msg);
    }

    return currentEvents & ZMQ_POLLIN;
}

void
ZmqEventSource::
handleMessage(const std::vector<std::string> & message)
{
    if (asyncMessageHandler) {
        asyncMessageHandler(message);
        return;
    }

    auto reply = handleSyncMessage(message);
    if (!reply.empty()) {
        sendAll(socket(), reply);
    }
}

std::vector<std::string>
ZmqEventSource::
handleSyncMessage(const std::vector<std::string> & message)
{
    if (!syncMessageHandler)
        THROW(ZmqLogs::error) << "no message handler set" << std::endl;

    return syncMessageHandler(message);
}
    

/*****************************************************************************/
/* ZMQ SOCKET MONITOR                                                        */
/*****************************************************************************/

//static int numMonitors = 0;

ZmqSocketMonitor::
ZmqSocketMonitor(zmq::context_t & context)
    : monitorEndpoint(new zmq::socket_t(context, ZMQ_PAIR)),
      monitoredSocket(0)
{
    //cerr << "creating socket monitor at " << this << endl;
    //__sync_fetch_and_add(&numMonitors, 1);
}

void
ZmqSocketMonitor::
shutdown()
{
    if (!monitorEndpoint)
        return;

    //cerr << "shutting down socket monitor at " << this << endl;
    
    connectedUri.clear();
    std::unique_lock<Lock> guard(lock);
    monitorEndpoint.reset();

    //cerr << __sync_add_and_fetch(&numMonitors, -1) << " monitors still active"
    //     << endl;
}

void
ZmqSocketMonitor::
disconnect()
{
    std::unique_lock<Lock> guard(lock);
    if (monitorEndpoint)
        monitorEndpoint->tryDisconnect(connectedUri.c_str());
}

void
ZmqSocketMonitor::
init(zmq::socket_t & socketToMonitor, int events)
{
    static int serial = 0;

    // Initialize the monitor connection
    connectedUri
        = ML::format("inproc://monitor-%p-%d",
                     this, __sync_fetch_and_add(&serial, 1));
    monitoredSocket = &socketToMonitor;
        
    //using namespace std;
    //cerr << "connecting monitor to " << connectedUri << endl;

    int res = zmq_socket_monitor(socketToMonitor, connectedUri.c_str(), events);
    if (res == -1)
        throw zmq::error_t();

    // Connect it in
    monitorEndpoint->connect(connectedUri.c_str());

    // Make sure we receive events from it
    ZmqBinaryTypedEventSource<zmq_event_t>::init(*monitorEndpoint);

    messageHandler = [=] (const zmq_event_t & event)
        {
            this->handleEvent(event);
        };
}

bool debugZmqMonitorEvents = false;

int
ZmqSocketMonitor::
handleEvent(const zmq_event_t & event)
{
    if (debugZmqMonitorEvents) {
        cerr << "got socket event " << printZmqEvent(event.event)
             << " at " << this
             << " " << connectedUri
             << " for socket " << monitoredSocket << endl;
    }

    auto doEvent = [&] (const EventHandler & handler,
                        const char * addr,
                        int param)
        {
            if (handler)
                handler(addr, param, event);
            else if (defaultHandler)
                defaultHandler(addr, param, event);
            else return 0;
            return 1;
        };

    switch (event.event) {

        // Bind
    case ZMQ_EVENT_LISTENING:
        return doEvent(bindHandler,
                       event.data.listening.addr,
                       event.data.listening.fd);

    case ZMQ_EVENT_BIND_FAILED:
        return doEvent(bindFailureHandler,
                       event.data.bind_failed.addr,
                       event.data.bind_failed.err);

        // Accept
    case ZMQ_EVENT_ACCEPTED:
        return doEvent(acceptHandler,
                       event.data.accepted.addr,
                       event.data.accepted.fd);
    case ZMQ_EVENT_ACCEPT_FAILED:
        return doEvent(acceptFailureHandler,
                       event.data.accept_failed.addr,
                       event.data.accept_failed.err);
        break;

        // Connect
    case ZMQ_EVENT_CONNECTED:
        return doEvent(connectHandler,
                       event.data.connected.addr,
                       event.data.connected.fd);
    case ZMQ_EVENT_CONNECT_DELAYED:
        return doEvent(connectFailureHandler,
                       event.data.connect_delayed.addr,
                       event.data.connect_delayed.err);
    case ZMQ_EVENT_CONNECT_RETRIED:
        return doEvent(connectRetryHandler,
                       event.data.connect_retried.addr,
                       event.data.connect_retried.interval);
            
        // Close and disconnection
    case ZMQ_EVENT_CLOSE_FAILED:
        return doEvent(closeFailureHandler,
                       event.data.close_failed.addr,
                       event.data.close_failed.err);
    case ZMQ_EVENT_CLOSED:
        return doEvent(closeHandler,
                       event.data.closed.addr,
                       event.data.closed.fd);

    case ZMQ_EVENT_DISCONNECTED:
        return doEvent(disconnectHandler,
                       event.data.disconnected.addr,
                       event.data.disconnected.fd);
            
    default:
        LOG(ZmqLogs::print)
            << "got unknown event type " << event.event << endl;
        return doEvent(defaultHandler, "", -1);
    }
}


/*****************************************************************************/
/* NAMED ZEROMQ ENDPOINT                                                     */
/*****************************************************************************/

ZmqNamedEndpoint::
ZmqNamedEndpoint(std::shared_ptr<zmq::context_t> context)
    : context_(context)
{
}

void
ZmqNamedEndpoint::
init(std::shared_ptr<ConfigurationService> config,
     int socketType,
     const std::string & endpointName)
{
    NamedEndpoint::init(config, endpointName);
    this->socketType = socketType;
    this->socket_.reset(new zmq::socket_t(*context_, socketType));
    setHwm(*socket_, 65536);
    
    addSource("ZmqNamedEndpoint::socket",
              std::make_shared<ZmqBinaryEventSource>
              (*socket_, [=] (std::vector<zmq::message_t> && message)
               {
                   handleRawMessage(std::move(message));
               }));
}

std::string
ZmqNamedEndpoint::
bindTcp(PortRange const & portRange, std::string host)
{
    std::unique_lock<Lock> guard(lock);

    if (!socket_)
        THROW(ZmqLogs::error) << "bind called before init" << std::endl;

    using namespace std;

    if (host == "")
        host = "*";

    int port = bindAndReturnOpenTcpPort(*socket_, portRange, host);

    auto getUri = [&] (const std::string & host)
        {
            return "tcp://" + host + ":" + to_string(port);
        };

    Json::Value config;

    auto addEntry = [&] (const std::string& addr,
                         const std::string& hostScope)
        {
            std::string uri;

            if(hostScope != "*") {
               uri = "tcp://" + addr + ":" + to_string(port);
            }
            else {
               uri = "tcp://" + ML::fqdn_hostname(to_string(port)) + ":" + to_string(port);
            }

            Json::Value & entry = config[config.size()];
            entry["zmqConnectUri"] = uri;

            Json::Value & transports = entry["transports"];
            transports[0]["name"] = "tcp";
            transports[0]["addr"] = addr;
            transports[0]["hostScope"] = hostScope;
            transports[0]["port"] = port;
            transports[1]["name"] = "zeromq";
            transports[1]["socketType"] = socketType;
            transports[1]["uri"] = uri;
        };

    if (host == "*") {
        auto interfaces = getInterfaces({AF_INET});
        for (unsigned i = 0;  i < interfaces.size();  ++i) {
            addEntry(interfaces[i].addr, interfaces[i].hostScope);
        }
        publishAddress("tcp", config);
        return getUri(host);
    }
    else {
        string host2 = addrToIp(host);
        // TODO: compute the host scope; don't just assume "*"
        addEntry(host2, "*");
        publishAddress("tcp", config);
        return getUri(host2);
    }
}

 

/*****************************************************************************/
/* NAMED ZEROMQ PROXY                                                        */
/*****************************************************************************/

ZmqNamedProxy::
ZmqNamedProxy() :
    context_(new zmq::context_t(1)),
    local(true),
    shardIndex(-1)
{
}

ZmqNamedProxy::
ZmqNamedProxy(std::shared_ptr<zmq::context_t> context, int shardIndex) :
    context_(context),
    local(true),
    shardIndex(shardIndex)
{
}

void
ZmqNamedProxy::
init(std::shared_ptr<ConfigurationService> config,
     int socketType,
     const std::string & identity)
{
    this->connectionType = NO_CONNECTION;
    this->connectionState = NOT_CONNECTED;

    this->config = config;
    socket_.reset(new zmq::socket_t(*context_, socketType));
    if (identity != "")
        setIdentity(*socket_, identity);
    setHwm(*socket_, 65536);

    serviceWatch.init(std::bind(&ZmqNamedProxy::onServiceNodeChange,
                                this,
                                std::placeholders::_1,
                                std::placeholders::_2));

    endpointWatch.init(std::bind(&ZmqNamedProxy::onEndpointNodeChange,
                                 this,
                                 std::placeholders::_1,
                                 std::placeholders::_2));
}

bool
ZmqNamedProxy::
connect(const std::string & endpointName,
        ConnectionStyle style)
{
    if (!config) {
        THROW(ZmqLogs::error)
            << "attempt to connect to " << endpointName
            << " without calling init()" << endl;
    }

    if (connectionState == CONNECTED)
        THROW(ZmqLogs::error) << "already connected" << endl;

    this->connectedService = endpointName;
    if (connectionType == NO_CONNECTION)
        connectionType = CONNECT_DIRECT;

    LOG(ZmqLogs::print) << "connecting to " << endpointName << endl;

    vector<string> children
        = config->getChildren(endpointName, endpointWatch);

    auto setPending = [&]
        {
            std::lock_guard<ZmqEventSource::SocketLock> guard(socketLock_);

            if (connectionState == NOT_CONNECTED)
                connectionState = CONNECTION_PENDING;
        };

    for (auto c: children) {
        ExcAssertNotEqual(connectionState, CONNECTED);
        string key = endpointName + "/" + c;
        Json::Value epConfig = config->getJson(key);
                
        for (auto & entry: epConfig) {

            if (!entry.isMember("zmqConnectUri"))
                return true;

            string uri = entry["zmqConnectUri"].asString();

            auto hs = entry["transports"][0]["hostScope"];
            if (!hs)
                continue;

            string hostScope = hs.asString();
            if (hs != "*") {
                utsname name;
                if (uname(&name)) {
                    THROW(ZmqLogs::error)
                        << "uname error: " << strerror(errno) << std::endl;
                }
                if (hostScope != name.nodename)
                    continue;  // wrong host scope
            }

            {
                std::lock_guard<ZmqEventSource::SocketLock> guard(socketLock_);
                socket().connect(uri.c_str());
                connectedUri = uri;
                connectionState = CONNECTED;
            }

            LOG(ZmqLogs::print) << "connected to " << uri << endl;
            onConnect(uri);
            return true;
        }

        setPending();
        return false;
    }

    if (style == CS_MUST_SUCCEED && connectionState != CONNECTED) {
        THROW(ZmqLogs::error)
            << "couldn't connect to any services of class " <<  serviceClass
            << endl;
    }

    setPending();
    return connectionState == CONNECTED;
}

bool
ZmqNamedProxy::
connectToServiceClass(const std::string & serviceClass,
                      const std::string & endpointName,
                      bool local_,
                      ConnectionStyle style)
{
    local = local_;

    // TODO: exception safety... if we bail don't screw around the auction
    ExcAssertNotEqual(connectionType, CONNECT_DIRECT);
    ExcAssertNotEqual(serviceClass, "");
    ExcAssertNotEqual(endpointName, "");

    this->serviceClass = serviceClass;
    this->endpointName = endpointName;

    if (connectionType == NO_CONNECTION)
        connectionType = CONNECT_TO_CLASS;

    if (!config) {
        THROW(ZmqLogs::error)
            << "attempt to connect to " << endpointName
            << " without calling init()" << endl;
    }

    if (connectionState == CONNECTED)
        THROW(ZmqLogs::error) << "attempt to double connect connection" << endl;

    vector<string> children
        = config->getChildren("serviceClass/" + serviceClass, serviceWatch);

    for (auto c: children) {
        string key = "serviceClass/" + serviceClass + "/" + c;
        Json::Value value = config->getJson(key);
        std::string name = value["serviceName"].asString();
        std::string path = value["servicePath"].asString();

        std::string location = value["serviceLocation"].asString();
        if (local && location != config->currentLocation) {
                LOG(ZmqLogs::trace)
                    << path << " / " << name << " dropped while connecting to "
                    << serviceClass << "/" << endpointName
                    << "(" << location << " != " << config->currentLocation << ")"
                    << std::endl;
            continue;
        }

        bool ok = connect(
                path + "/" + endpointName,
                style == CS_ASYNCHRONOUS ? CS_ASYNCHRONOUS : CS_SYNCHRONOUS);
        if (!ok) continue;

        shardIndex = value.get("shardIndex", -1).asInt();
        return true;
    }

    if (style == CS_MUST_SUCCEED && connectionState != CONNECTED) {
        THROW(ZmqLogs::error)
            << "couldn't connect to any services of class " << serviceClass
            << endl;
    }

    {
        std::lock_guard<ZmqEventSource::SocketLock> guard(socketLock_);

        if (connectionState == NOT_CONNECTED)
            connectionState = CONNECTION_PENDING;
    }

    return connectionState == CONNECTED;
}

void
ZmqNamedProxy::
onServiceNodeChange(const std::string & path,
                    ConfigurationService::ChangeType change)
{
    if (connectionState != CONNECTION_PENDING)
        return;  // no need to watch anymore

    connectToServiceClass(serviceClass, endpointName, local, CS_ASYNCHRONOUS);
}

void
ZmqNamedProxy::
onEndpointNodeChange(const std::string & path,
                     ConfigurationService::ChangeType change)
{
    if (connectionState != CONNECTION_PENDING)
        return;  // no need to watch anymore

    connect(connectedService, CS_ASYNCHRONOUS);
}


/******************************************************************************/
/* ZMQ MULTIPLE NAMED CLIENT BUS PROXY                                        */
/******************************************************************************/

void
ZmqMultipleNamedClientBusProxy::
connectAllServiceProviders(const std::string & serviceClass,
                           const std::string & endpointName,
                           bool local)
{
    if (connected) {
        THROW(ZmqLogs::error)
            << "already connected to "
                << serviceClass << " / " << endpointName
                << std::endl;
    }

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

/** Zookeeper makes the onServiceProvidersChanged calls re-entrant which is
    annoying to deal with. Instead, when a re-entrant call is detected we defer
    the call until we're done with the original call.
 */
bool
ZmqMultipleNamedClientBusProxy::
enterProvidersChanged(const std::string& path, bool local)
{
    std::lock_guard<ML::Spinlock> guard(providersChangedLock);

    if (!inProvidersChanged) {
        inProvidersChanged = true;
        return true;
    }

    LOG(ZmqLogs::trace)
        << "defering providers changed for " << path << std::endl;

    deferedProvidersChanges.emplace_back(path, local);
    return false;
}

void
ZmqMultipleNamedClientBusProxy::
exitProvidersChanged()
{
    std::vector<DeferedProvidersChanges> defered;

    {
        std::lock_guard<ML::Spinlock> guard(providersChangedLock);

        defered = std::move(deferedProvidersChanges);
        inProvidersChanged = false;
    }

    for (const auto& item : defered)
        onServiceProvidersChanged(item.first, item.second);
}

void
ZmqMultipleNamedClientBusProxy::
onServiceProvidersChanged(const std::string & path, bool local)
{
    if (!enterProvidersChanged(path, local)) return;

    // The list of service providers has changed

    std::vector<std::string> children
        = config->getChildren(path, serviceProvidersWatch);

    for (auto c: children) {
        Json::Value value = config->getJson(path + "/" + c);
        std::string name = value["serviceName"].asString();
        std::string path = value["servicePath"].asString();
        int shardIndex = value.get("shardIndex", -1).asInt();

        std::string location = value["serviceLocation"].asString();
        if (local && location != config->currentLocation) {
            LOG(ZmqLogs::trace)
                << path << " / " << name << " dropped ("
                    << location << " != " << config->currentLocation << ")"
                    << std::endl;
            continue;
        }

        watchServiceProvider(name, path, shardIndex);
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

    exitProvidersChanged();
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
struct ZmqMultipleNamedClientBusProxy::OnConnectCallback
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

    void operator() (std::string blah)
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


void
ZmqMultipleNamedClientBusProxy::
watchServiceProvider(const std::string & name, const std::string & path, int shardIndex)
{
    // Protects the connections map... I think.
    std::unique_lock<Lock> guard(connectionsLock);

    auto & c = connections[name];

    // already connected
    if (c) {
        LOG(ZmqLogs::trace)
            << path << " / " << name << " is already connected"
                << std::endl;
        return;
    }

    LOG(ZmqLogs::trace)
        << "connecting to " << path << " / " << name << std::endl;

    try {
        auto newClient = std::make_shared<ZmqNamedClientBusProxy>(zmqContext, shardIndex);
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


} // namespace Datacratic

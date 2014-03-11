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


/*****************************************************************************/
/* ZMQ EVENT SOURCE                                                          */
/*****************************************************************************/

ZmqEventSource::
ZmqEventSource()
    : socket_(0), socketLock_(nullptr)
{
    needsPoll = true;
}

ZmqEventSource::
ZmqEventSource(zmq::socket_t & socket, SocketLock * socketLock)
    : socket_(&socket), socketLock_(socketLock)
{
    needsPoll = true;
}

void
ZmqEventSource::
init(zmq::socket_t & socket, SocketLock * socketLock)
{
    socket_ = &socket;
    socketLock_ = socketLock;
    needsPoll = true;
}

int
ZmqEventSource::
selectFd() const
{
    int res = -1;
    size_t resSize = sizeof(int);
    socket().getsockopt(ZMQ_FD, &res, &resSize);
    if (res == -1)
        throw ML::Exception("no fd for zeromq socket");
    using namespace std;
    //cerr << "select FD is " << res << endl;
    return res;
}

bool
ZmqEventSource::
poll() const
{
    return getEvents(socket()).first;

#if 0
    using namespace std;

    

    zmq_pollitem_t toPoll = { socket(), 0, ZMQ_POLLIN };
    int res = zmq_poll(&toPoll, 1, 0);
    //cerr << "poll returned " << res << endl;
    if (res == -1)
        throw ML::Exception(errno, "zmq_poll");
    return res;
#endif
}

bool
ZmqEventSource::
processOne()
{
    using namespace std;
    if (debug_)
        cerr << "called processOne on " << this << ", poll = " << poll() << endl;

    std::vector<std::string> msg;

    /** NOTE: poll() will only work after we've tried (and failed) to
        pull a message off.
    */
    {
        std::unique_lock<SocketLock> guard;
        if (socketLock_)
            guard = std::unique_lock<SocketLock>(*socketLock_);

        msg = recvAllNonBlocking(socket());
    }

    if (!msg.empty()) {
        if (debug_)
            cerr << "got message of length " << msg.size() << endl;
        handleMessage(msg);
    }

    return poll();
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
    if (syncMessageHandler)
        return syncMessageHandler(message);
    throw ML::Exception("need to assign to or override one of the "
                        "message handlers");
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
        using namespace std;
        cerr << "got unknown event type " << event.event << endl;
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
        throw ML::Exception("need to call ZmqNamedEndpoint::init() before "
                            "bind");

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
    local(true)
{
}

ZmqNamedProxy::
ZmqNamedProxy(std::shared_ptr<zmq::context_t> context) :
    context_(context),
    local(true)
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
    if (!config)
        throw ML::Exception("attempt to connect to named service "
                            + endpointName + " without calling init()");

    if (connectionState == CONNECTED)
        throw ML::Exception("already connected");

    this->connectedService = endpointName;
    if (connectionType == NO_CONNECTION)
        connectionType = CONNECT_DIRECT;

    cerr << "connecting to " << endpointName << endl;

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
        //cerr << "got key " << key << endl;
        Json::Value epConfig = config->getJson(key);

        //cerr << "epConfig for " << key << " is " << epConfig
        //     << endl;
                
        for (auto & entry: epConfig) {

            //cerr << "entry is " << entry << endl;

            if (!entry.isMember("zmqConnectUri"))
                return true;

            string uri = entry["zmqConnectUri"].asString();

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

            {
                std::lock_guard<ZmqEventSource::SocketLock> guard(socketLock_);
                socket().connect(uri.c_str());
                connectedUri = uri;
                connectionState = CONNECTED;
            }

            cerr << "connected to " << uri << endl;
            onConnect(uri);
            return true;
        }

        setPending();
        return false;
    }

    if (style == CS_MUST_SUCCEED && connectionState != CONNECTED)
        throw ML::Exception("couldn't connect to any services of class "
                            + serviceClass);

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

    //cerr << "serviceClass = " << serviceClass << endl;

    this->serviceClass = serviceClass;
    this->endpointName = endpointName;

    if (connectionType == NO_CONNECTION)
        connectionType = CONNECT_TO_CLASS;

    if (!config)
        throw ML::Exception("attempt to connect to named service "
                            + endpointName + " without calling init()");

    if (connectionState == CONNECTED)
        throw ML::Exception("attempt to double connect connection");

    vector<string> children
        = config->getChildren("serviceClass/" + serviceClass, serviceWatch);

    for (auto c: children) {
        string key = "serviceClass/" + serviceClass + "/" + c;
        //cerr << "getting " << key << endl;
        Json::Value value = config->getJson(key);
        std::string name = value["serviceName"].asString();
        std::string path = value["servicePath"].asString();

        std::string location = value["serviceLocation"].asString();
        if (local && location != config->currentLocation) {
            std::cerr << "dropping " << location << " != " << config->currentLocation << std::endl;
            continue;
        }

        //cerr << "name = " << name << " path = " << path << endl;
        if (connect(path + "/" + endpointName,
                    style == CS_ASYNCHRONOUS ? CS_ASYNCHRONOUS : CS_SYNCHRONOUS))
            return true;
    }

    if (style == CS_MUST_SUCCEED && connectionState != CONNECTED)
        throw ML::Exception("couldn't connect to any services of class "
                            + serviceClass);

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
    //cerr << "******* CHANGE TO SERVICE NODE " << path << endl;

    if (connectionState != CONNECTION_PENDING)
        return;  // no need to watch anymore

    connectToServiceClass(serviceClass, endpointName, local, CS_ASYNCHRONOUS);
}

void
ZmqNamedProxy::
onEndpointNodeChange(const std::string & path,
                     ConfigurationService::ChangeType change)
{
    //cerr << "******* CHANGE TO ENDPOINT NODE " << path << endl;

    if (connectionState != CONNECTION_PENDING)
        return;  // no need to watch anymore

    connect(connectedService, CS_ASYNCHRONOUS);
}


} // namespace Datacratic

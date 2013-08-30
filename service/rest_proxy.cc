/* rest_proxy.cc
   Jeremy Barnes, 14 November 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

*/

#include "rest_proxy.h"
#include "jml/arch/exception_handler.h"

using namespace std;
using namespace ML;

namespace Datacratic {

/*****************************************************************************/
/* REST PROXY                                                                */
/*****************************************************************************/

RestProxy::
RestProxy()
    : operationQueue(1024),
      numMessagesOutstanding_(0),
      currentOpId(1)
{
    // What to do when we get a new entry in the queue?
    operationQueue.onEvent = std::bind(&RestProxy::handleOperation,
                                       this, std::placeholders::_1);
    
}

RestProxy::
RestProxy(const std::shared_ptr<zmq::context_t> & context)
    : operationQueue(1024),
      connection(context),
      numMessagesOutstanding_(0),
      currentOpId(1)
{
    // What to do when we get a new entry in the queue?
    operationQueue.onEvent = std::bind(&RestProxy::handleOperation,
                                       this, std::placeholders::_1);
    
}

RestProxy::
~RestProxy()
{
    shutdown();
}

void
RestProxy::
sleepUntilIdle()
{
    for (;;) {
        int o = numMessagesOutstanding_;
        //cerr << "numMessagesOustanding = " << o << endl;
        if (!o)
            return;
        ML::futex_wait(numMessagesOutstanding_, o, 0.01);
    }
}

void
RestProxy::
shutdown()
{
    // Stop processing messages
    MessageLoop::shutdown();

    connection.shutdown();
}

void
RestProxy::
init(std::shared_ptr<ConfigurationService> config,
     const std::string & serviceName,
     const std::string & endpointName)
{
    serviceName_ = serviceName;

    connection.init(config, ZMQ_XREQ);
    connection.connect(serviceName + "/" + endpointName);
    
    addSource("RestProxy::operationQueue", operationQueue);

    // What to do when we get something back from zeromq?
    addSource("RestProxy::handleZmqResponse",
              std::make_shared<ZmqEventSource>
              (connection.socket(),
               std::bind(&RestProxy::handleZmqResponse,
                         this,
                         std::placeholders::_1)));
}

void
RestProxy::
initServiceClass(std::shared_ptr<ConfigurationService> config,
                 const std::string & serviceClass,
                 const std::string & serviceEndpoint,
                 bool local)
{
    connection.init(config, ZMQ_XREQ);
    connection.connectToServiceClass(serviceClass, serviceEndpoint, local);
    
    addSource("RestProxy::operationQueue", operationQueue);

    // What to do when we get something back from zeromq?
    addSource("RestProxy::handleZmqResponse",
              std::make_shared<ZmqEventSource>
              (connection.socket(),
               std::bind(&RestProxy::handleZmqResponse,
                         this,
                         std::placeholders::_1)));
}

void
RestProxy::
push(const RestRequest & request, const OnDone & onDone)
{
    Operation op;
    op.request = request;
    op.onDone = onDone;
    if (operationQueue.tryPush(std::move(op)))
        ML::atomic_inc(numMessagesOutstanding_);
    else
        throw ML::Exception("queue is full");
}

void
RestProxy::
push(const OnDone & onDone,
     const std::string & method,
     const std::string & resource,
     const RestParams & params,
     const std::string & payload)
{
    RestRequest request(method, resource, params, payload);
    push(request, onDone);
}

void
RestProxy::
handleOperation(const Operation & op)
{
    // Gets called when someone calls our API to make something happen;
    // this is run by the main worker thread to actually do the work.
    // It forwards the request off to the master banker.
    uint64_t opId = 0;
    if (op.onDone)
        opId = currentOpId++;

    //cerr << "sending with payload " << op.request.payload
    //     << " and response id " << opId << endl;

    if (trySendMessage(connection.socket(),
                       std::to_string(opId),
                       op.request.verb,
                       op.request.resource,
                       op.request.params.toBinary(),
                       op.request.payload)) {
        if (opId)
            outstanding[opId] = op.onDone;
        else {
            int no = __sync_add_and_fetch(&numMessagesOutstanding_, -1);
            if (no == 0)
                futex_wake(numMessagesOutstanding_);
        }
    }
    else {
        if (op.onDone) {
            ML::Set_Trace_Exceptions notrace(false);
            string exc_msg = ("connection to '" + serviceName_
                              + "' is unavailable");
            op.onDone(make_exception_ptr<ML::Exception>(exc_msg), 0, "");
        }
        int no = __sync_add_and_fetch(&numMessagesOutstanding_, -1);
        if (no == 0)
            futex_wake(numMessagesOutstanding_);
    }
}

void
RestProxy::
handleZmqResponse(const std::vector<std::string> & message)
{
    // Gets called when we get a response back from the master banker in
    // response to one of our calls.

    // We call the callback associated with this code.

    //cerr << "response is " << message << endl;

    uint64_t opId = boost::lexical_cast<uint64_t>(message.at(0));
    int responseCode = boost::lexical_cast<int>(message.at(1));
    std::string body = message.at(2);

    ExcAssert(opId);

    auto it = outstanding.find(opId);
    if (it == outstanding.end()) {
        cerr << "unknown op ID " << endl;
        return;
    }
    try {
        if (responseCode >= 200 && responseCode < 300) 
            it->second(nullptr, responseCode, body);
        else
            it->second(std::make_exception_ptr(ML::Exception(body)),
                       responseCode, "");
    } catch (const std::exception & exc) {
        cerr << "warning: exception handling banker result: "
             << exc.what() << endl;
    } catch (...) {
        cerr << "warning: unknown exception handling banker result"
             << endl;
    }

    outstanding.erase(it);
    
    ML::atomic_dec(numMessagesOutstanding_);
}


/******************************************************************************/
/* MULTI REST PROXY                                                           */
/******************************************************************************/

void
MultiRestProxy::
shutdown()
{
    if (!connected) return;

    MessageLoop::shutdown();

    lock_guard<ML::Spinlock> guard(connectionsLock);

    for (auto& conn: connections) {
        if (!conn.second) continue;
        conn.second->shutdown();
    }

    connections.clear();
    connected = false;
}

namespace {

RestProxy::OnDone
makeResponseFn(
        const std::string& serviceName, const MultiRestProxy::OnResponse& fn)
{
    return [=] (std::exception_ptr ex, int code, const std::string& msg) {
        if (fn) fn(serviceName, ex, code, msg);
    };
}

} // namespace anonymous



void
MultiRestProxy::
push(const RestRequest & request, const OnResponse & onResponse)
{
    lock_guard<ML::Spinlock> guard(connectionsLock);

    for (const auto& conn : connections) {
        if (!conn.second) continue;

        auto onDone = makeResponseFn(conn.first, onResponse);
        conn.second->push(request, onDone);
    }
}


void
MultiRestProxy::
push(   const OnResponse & onResponse,
        const string & method,
        const string & resource,
        const RestParams & params,
        const string & payload)
{
    lock_guard<ML::Spinlock> guard(connectionsLock);

    for (const auto& conn : connections) {
        if (!conn.second) continue;

        auto onDone = makeResponseFn(conn.first, onResponse);
        conn.second->push(onDone , method, resource, params, payload);
    }
}


void
MultiRestProxy::
connectAllServiceProviders(
        const string& serviceClass, const string& endpointName, bool local)
{
    ExcCheck(!connected, "Already connectoed to a service provider");

    this->serviceClass = serviceClass;
    this->endpointName = endpointName;
    this->localized = local;

    serviceProvidersWatch.init(
            [=] (const string&, ConfigurationService::ChangeType) {
                onServiceProvidersChanged("serviceClass/" + serviceClass, local);
            });

    onServiceProvidersChanged("serviceClass/" + serviceClass, local);
    connected = true;
}

void
MultiRestProxy::
connectServiceProvider(const string& serviceName)
{
    {
        lock_guard<ML::Spinlock> guard(connectionsLock);

        auto& conn = connections[serviceName];
        if (conn) return;

        shared_ptr<RestProxy> newConn(new RestProxy(context));
        newConn->init(config, serviceName, endpointName);
        conn = std::move(newConn);

        addSource("MultiRestProxy::" + serviceName, conn);
    }

    onConnect(serviceName);
}

void
MultiRestProxy::
onServiceProvidersChanged(const string& path, bool local)
{
    vector<string> children = config->getChildren(path, serviceProvidersWatch);

    for (const auto& child : children) {
        Json::Value value = config->getJson(path + "/" + child);

        string location = value["serviceLocation"].asString();
        if (local && location != config->currentLocation) {
            cerr << "dropping " << location
                << " != " << config->currentLocation
                << endl;
            continue;
        }

        connectServiceProvider(value["serviceName"].asString());
    }

    vector<string> disconnected;
    {
        lock_guard<ML::Spinlock> guard(connectionsLock);

        for (const auto& conn : connections) {
            if (!conn.second) continue;

            auto it = find(children.begin(), children.end(), conn.first);
            if (it != children.end()) continue;

            removeSource(conn.second.get());
            disconnected.push_back(conn.first);
        }

        // We don't have to worry about invalidating iterators anymore.
        for (const auto& conn : disconnected)
            connections.erase(conn);
    }

    // Lock has been released and it's now safe to trigger the callbacks.
    for (const auto& conn : disconnected)
        onDisconnect(conn);
}

} // namespace Datacratic

/* rest_multiproxy.cc
   Wolfgang Sourdeau, January 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   An extended derivative of RestProxy, designed to handle multiple
   connections in the same message loop.
*/

#include <memory>
#include "rest_multi_proxy.h"

using namespace std;
using namespace ML;

namespace RTBKIT {

/*****************************************************************************/
/* REST MULTIPROXY                                                                */
/*****************************************************************************/

RestMultiProxy::
RestMultiProxy(const std::shared_ptr<zmq::context_t> & context)
    : operationQueue(1024),
      context_(context),
      numMessagesOutstanding_(0),
      currentOpId(1)
{
    // What to do when we get a new entry in the queue?
    operationQueue.onEvent = std::bind(&RestMultiProxy::handleOperation,
                                       this, std::placeholders::_1);
    
}

RestMultiProxy::
~RestMultiProxy()
{
    shutdown();
}

void
RestMultiProxy::
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
RestMultiProxy::
shutdown()
{
    sleepUntilIdle();

    // Stop processing messages
    MessageLoop::shutdown();

    vector<string> keys;
    for (auto & it: connections) {
        keys.push_back(it.first);
        /* FIXME: the ZmqNamedProxy destructor does invoke shutdown already,
           why do we need it here to avoid a SIGABORT */
        it.second->shutdown();
        // delete it.second;
    }
    for (string & key: keys) {
        connections.erase(key);
    }
}

void
RestMultiProxy::
init(std::shared_ptr<ConfigurationService> config,
     const vector<string> & serviceNames)
{
    for (const string & serviceName: serviceNames) {
        auto connection = std::make_shared<ZmqNamedProxy>(context_);
        connection->init(config, ZMQ_XREQ);
        connection->connect(serviceName + "/zeromq");
        connections[serviceName] = connection;

        // What to do when we get something back from zeromq?
        addSource("RestMultiProxy::handleZmqResponse",
                  std::make_shared<ZmqEventSource>
                  (connection->socket(),
                   std::bind(&RestMultiProxy::handleZmqResponse,
                             this,
                             std::placeholders::_1)));
    }

    addSource("RestMultiProxy::operationQueue", operationQueue);
}

#if 0
void
RestMultiProxy::
initServiceClass(std::shared_ptr<ConfigurationService> config,
                 const std::string & serviceClass,
                 const std::string & serviceEndpoint)
{
    connection.init(config, ZMQ_XREQ);
    connection.connectToServiceClass(serviceClass, serviceEndpoint);
    
    addSource("RestMultiProxy::operationQueue", operationQueue);

    // What to do when we get something back from zeromq?
    addSource("RestMultiProxy::handleZmqResponse",
              std::make_shared<ZmqEventSource>
              (connection.socket(),
               std::bind(&RestMultiProxy::handleZmqResponse,
                         this,
                         std::placeholders::_1)));
}
#endif

void
RestMultiProxy::
push(const string & serviceName, const RestRequest & request, OnDone onDone)
{
    Operation op;
    op.connection = connections[serviceName];
    op.request = request;
    op.onDone = onDone;
    operationQueue.push(std::move(op));
    ML::atomic_inc(numMessagesOutstanding_);
}

void
RestMultiProxy::
push(const string & serviceName,
     const OnDone & onDone,
     const std::string & method,
     const std::string & resource,
     const RestParams & params,
     const std::string & payload)
{
    Operation op;
    op.connection = connections[serviceName];
    op.onDone = onDone;
    op.request.verb = method;
    op.request.resource = resource;
    op.request.params = params;
    op.request.payload = payload;
    operationQueue.push(std::move(op));
    ML::atomic_inc(numMessagesOutstanding_);
}

void
RestMultiProxy::
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

    sendMessage(op.connection->socket(), std::to_string(opId),
                op.request.verb,
                op.request.resource,
                op.request.params.toBinary(),
                op.request.payload);
    
    if (opId)
        outstanding[opId] = op.onDone;
    else {
        int no = __sync_add_and_fetch(&numMessagesOutstanding_, -1);
        if (no == 0)
            futex_wake(numMessagesOutstanding_);
    }
}

void
RestMultiProxy::
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

} // namespace RTBKIT

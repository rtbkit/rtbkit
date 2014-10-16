/* rest_proxy.h                                                    -*- C++ -*-
   Jeremy Banres, 14 November 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

*/

#pragma once

#include "soa/service/message_loop.h"
#include "soa/service/zmq_endpoint.h"
#include "soa/service/typed_message_channel.h"
#include "soa/service/rest_service_endpoint.h"

namespace Datacratic {


/*****************************************************************************/
/* HELPER FUNCTIONS                                                          */
/*****************************************************************************/

/** Pass through a REST response to the given function. */

template<typename Result>
void decodeRestResponseJson(const std::string & functionName,
                            std::exception_ptr exc,
                            int resultCode,
                            const std::string & body,
                            std::function<void (std::exception_ptr,
                                                Result &&)> onDone)
{
    Result result;

    try {
        if (exc) {
            onDone(exc, std::move(result));
            return;
        }
        else if (resultCode < 200 || resultCode >= 300) {
            onDone(std::make_exception_ptr
                   (ML::Exception("%s REST request failed: %d: %s",
                                  functionName.c_str(),
                                  resultCode,
                                  body.c_str())),
                   std::move(result));
            return;
        }
        else {
            onDone(nullptr, std::move(static_cast<Result>(Result::fromJson(Json::parse(body)))));
        }
    } catch (...) {
        onDone(std::current_exception(), std::move(result));
    }
}

template<typename Result>
std::function<void (std::exception_ptr, int, std::string)>
makeRestResponseJsonDecoder(std::string functionName,
                            std::function<void (std::exception_ptr,
                                                Result &&)> onDone)
{
    return [=] (std::exception_ptr exc,
                int resultCode,
                std::string body)
        {
            if (!onDone)
                return;
            decodeRestResponseJson<Result>(functionName,
                                           exc, resultCode, body,
                                           onDone);
        };
}


/*****************************************************************************/
/* REST PROXY                                                                */
/*****************************************************************************/

/** Proxy that handles a set of outstanding asynchronous operations to a
    rest endpoint and deals with getting the results back.
*/

struct RestProxy: public MessageLoop {

    RestProxy();

    RestProxy(const std::shared_ptr<zmq::context_t> & context);
    
    ~RestProxy();

    void sleepUntilIdle();

    void shutdown();

    /** Initialize and connect to the given service on the "zeromq" endpoint. */
    void init(std::shared_ptr<ConfigurationService> config,
              const std::string & serviceName,
              const std::string & endpointName = "zeromq");
    
    /** Initialize and connect to an instance of the given service class. */
    void initServiceClass(std::shared_ptr<ConfigurationService> config,
                          const std::string & serviceClass,
                          const std::string & endpointName,
                          bool local = true);
    
    typedef std::function<void (std::exception_ptr,
                                int responseCode, const std::string &)> OnDone;

    /** Push the given request to be performed.  When the result comes
        back, it will be passed to OnDone.
    */
    void push(const RestRequest & request, const OnDone & onDone);

    /** Push the given request to be performed. */
    void push(const OnDone & onDone,
              const std::string & method,
              const std::string & resource,
              const RestParams & params = RestParams(),
              const std::string & payload = "");

    size_t numMessagesOutstanding() const
    {
        return numMessagesOutstanding_;
    }

protected:
    std::string serviceName_;
    std::string endpointName_;

    struct Operation {
        RestRequest request;
        OnDone onDone;
    };

    TypedMessageSink<Operation> operationQueue;
    ZmqNamedProxy connection;

    std::map<uint64_t, OnDone> outstanding;
    int numMessagesOutstanding_;  // atomic so can be read with no lock
    uint64_t currentOpId;

    void handleOperation(const Operation & op);
    void handleZmqResponse(const std::vector<std::string> & message);
};


/******************************************************************************/
/* MULTI REST PROXY                                                           */
/******************************************************************************/

/** Provides a way to connect to all services under a given service class
    through a REST interface. Note that this sets a watch on the service class
    such that whenever a service goes down or comes up the rest proxy will be
    notified and will adjust itself accordingly.

    Note: This class does not and should not provide a sleepUntilIdle interface
    as it is very prone to deadlock when coupled with services being shutdown
    and restarted asynchronously.

    \todo Provides just about the same pattern as ZmqMultipleNamedClientBusProxy
    so there's common functionality here that could be merged.
 */
struct MultiRestProxy : public MessageLoop
{
    MultiRestProxy(std::shared_ptr<zmq::context_t> context) :
        connected(false),
        context(std::move(context))
    {}

    ~MultiRestProxy() { shutdown(); }


    void init(std::shared_ptr<ConfigurationService> config)
    {
        this->config = std::move(config);
    }

    void shutdown();


    typedef std::function<void (const std::string& serviceName)> ConnectionHandler;

    /** Called whenever we connect a new service. */
    ConnectionHandler connectHandler;
    virtual void onConnect(const std::string& serviceName)
    {
        if (connectHandler) connectHandler(serviceName);
    }

    /** Called whenever a service we were connected is disconnected. */
    ConnectionHandler disconnectHandler;
    virtual void onDisconnect(const std::string& serviceName)
    {
        if (disconnectHandler) disconnectHandler(serviceName);
    }


    /** Connects our class to every service under the given service class. */
    void connectAllServiceProviders(
            const std::string& serviceClass,
            const std::string& endpointName,
            bool local = true);


    typedef std::function<void (
            const std::string& serviceName,
            std::exception_ptr ex,
            int responseCode,
            const std::string& payload)> OnResponse;

    /** Send a REST message to every connected service. The response callback
        will be invoked once for every answer we receive such that if we're
        connected to 3 services then we can receive up to 5 messages.
     */
    void push(const RestRequest & request, const OnResponse & onResponse);
    void push(
            const OnResponse & onResponse,
            const std::string & method,
            const std::string & resource,
            const RestParams & params = RestParams(),
            const std::string & payload = "");

private:

    bool connected;

    std::shared_ptr<zmq::context_t> context;
    std::shared_ptr<ConfigurationService> config;

    std::string serviceClass;
    std::string endpointName;
    bool localized;

    ML::Spinlock connectionsLock;

    typedef std::map<std::string, std::shared_ptr<RestProxy> > ConnectionsMap;
    ConnectionsMap connections;

    ConfigurationService::Watch serviceProvidersWatch;

    void onServiceProvidersChanged(const std::string& path, bool local);
    void connectServiceProvider(const std::string& serviceName);
    void disconnectServiceProvider(const std::string& serviceName);

};

} // namespace Datacratic

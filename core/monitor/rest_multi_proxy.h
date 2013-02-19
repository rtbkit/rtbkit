/* rest_multiproxy.h                                               -*- C++ -*-
   Wolfgang Sourdeau, January 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   An extended derivative of RestProxy, designed to handle multiple
   connections in the same message loop.
*/

/* TODO: make rest_proxy derive from this class */

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/shared_ptr.hpp>

#include "soa/service/typed_message_channel.h"
#include "soa/service/rest_service_endpoint.h"


namespace Datacratic {
    class ZmqNamedProxy;
}

namespace RTBKIT {
    using namespace Datacratic;

/*****************************************************************************/
/* REST PROXY                                                                */
/*****************************************************************************/

/** Proxy that handles a set of outstanding asynchronous operations to a
    rest endpoint and deals with getting the results back.
*/

struct RestMultiProxy: public MessageLoop {

    RestMultiProxy(const std::shared_ptr<zmq::context_t> & context);
    
    ~RestMultiProxy();

    void sleepUntilIdle();

    void shutdown();

    /** Initialize and connect to the given services on the "zeromq"
     * endpoint. */
    void init(std::shared_ptr<ConfigurationService> config,
              const std::vector<std::string> & serviceNames);

#if 0
    /** Initialize and connect to an instance of the given service classes. */
    void initServiceClass(std::shared_ptr<ConfigurationService> config,
                          const std::string & serviceClass,
                          const std::string & endpointName);
#endif

    typedef std::function<void (std::exception_ptr,
                                int responseCode, const std::string &)>
        OnDone;

    /** Push the given request to be performed.  When the result comes
        back, it will be passed to OnDone.
    */
    void push(const std::string & serviceName, const RestRequest & request,
              OnDone onDone);

    /** Push the given request to be performed. */
    void push(const std::string & serviceName,
              const OnDone & onDone,
              const std::string & method,
              const std::string & resource,
              const RestParams & params = RestParams(),
              const std::string & payload = "");

    size_t numMessagesOutstanding() const
    {
        return numMessagesOutstanding_;
    }

protected:

    struct Operation {
        std::shared_ptr<ZmqNamedProxy> connection;
        RestRequest request;
        OnDone onDone;
    };

    TypedMessageSink<Operation> operationQueue;

    std::shared_ptr<zmq::context_t> context_;
    std::unordered_map<std::string,
                       std::shared_ptr<ZmqNamedProxy>> connections;

    std::map<uint64_t, OnDone> outstanding;
    int numMessagesOutstanding_;  // atomic so can be read with no lock
    uint64_t currentOpId;

    void handleOperation(const Operation & op);
    void handleZmqResponse(const std::vector<std::string> & message);
};

} // namespace RTBKIT

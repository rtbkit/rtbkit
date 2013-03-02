/* monitor_provider.h                                              -*- C++ -*-
   Wolfgang Sourdeau, January 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
   
   Rest endpoint queried by the monitor in the provider processes
*/

#pragma once

#include <memory>

#include "soa/types/date.h"
#include "soa/service/rest_proxy.h"

namespace zmq {
    struct context_t;
} // namespace zmq

namespace Json {
    struct Value;
} // namespace Json

namespace RTBKIT {
    using namespace Datacratic;

struct MonitorProvider
{
    /* this method returns the service identifier to use when sending status
       information to the Monitor */
    virtual std::string getProviderName() const = 0;

    /* this method returns the service status: "true" indicates that all the
       service-specific conditions are fulfilled, "false" otherwise */
    virtual Json::Value getProviderIndicators() const = 0;
};

struct MonitorProviderClient : public RestProxy
{
    MonitorProviderClient(const std::shared_ptr<zmq::context_t> & context,
                          MonitorProvider & provider);

    ~MonitorProviderClient();
 
    void init(std::shared_ptr<ConfigurationService> & config,
              const std::string & serviceName = "monitor");

    /** shutdown the MessageLoop but make sure all requests have been
        completed beforehand. */
    void shutdown();

    /** this method is invoked periodically to query the MonitorProvider and
     * "POST" the result to the Monitor */
    void postStatus();

    /** method executed when we receive the response from the Monitor */
    void onResponseReceived(std::exception_ptr ext,
                            int responseCode, const std::string & body);

    /** monitored service proxy */
    MonitorProvider & provider_;

    /** flag enabling the inhibition of requests to the Monitor service */
    bool inhibit_;

    /** monitored service name */
    std::string restUrlPath_;

    /** bound instance of onResponseReceived */
    RestProxy::OnDone onDone;

    /** whether a request roundtrip to the Monitor is currently active */
    bool pendingRequest;

    /** the mutex used when pendingRequest is tested and modified */
    typedef std::unique_lock<std::mutex> Guard;
    mutable std::mutex requestLock;
};

} // namespace RTBKIT

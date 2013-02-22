/* monitor_provider.h                                              -*- C++ -*-
   Wolfgang Sourdeau, January 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
   
   Rest endpoint queried by the monitor in the provider processes
*/

#pragma once

#include <memory>

#include "soa/types/date.h"

#include "soa/service/rest_request_router.h"
#include "soa/service/rest_service_endpoint.h"

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
    /* this method returns the service status: "true" indicates that all the
       service-specific conditions are fulfilled, "false" otherwise */
    virtual Json::Value getMonitorIndicators() = 0;
};

struct MonitorProviderEndpoint
    : public ServiceBase, public RestServiceEndpoint
{
    MonitorProviderEndpoint(ServiceBase & parentService,
                            MonitorProvider & provider);
    ~MonitorProviderEndpoint();

    void init();
    void shutdown();
    void bindTcp();

    /* this method returns the json body, based on "lastStatus_" when the
     * endpoint is queried */
    std::string restGetServiceStatus();

    /* this method is executed periodically and set "lastStatus_" to the value
       returned by MonitorProvider::serviceStatus */
    void refreshStatus();

    std::string endpointName_;

    RestRequestRouter router_;
    Json::Value lastStatus_;

    MonitorProvider & provider_;

    typedef std::unique_lock<std::mutex> Guard;
    std::mutex lock;
};

} // namespace RTBKIT

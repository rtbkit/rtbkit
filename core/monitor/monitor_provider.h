/* monitor_provider.h                                              -*- C++ -*-
   Wolfgang Sourdeau, January 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
   
   Rest endpoint queried by the monitor in the provider processes
*/

#pragma once

#include <memory>

#include "monitor_indicator.h"
#include "soa/types/date.h"
#include "soa/service/rest_proxy.h"

namespace zmq {
    struct context_t;
} // namespace zmq

namespace RTBKIT {
    using namespace Datacratic;

struct MonitorProvider
{
    /* this method returns the service identifier to use when sending status
       information to the Monitor */
    virtual std::string getProviderClass() const = 0;

    /* this method returns the service status: "true" indicates that all the
       service-specific conditions are fulfilled, "false" otherwise */
    virtual MonitorIndicator getProviderIndicators() const = 0;
};

struct MonitorProviderClient : public MultiRestProxy
{
    MonitorProviderClient(const std::shared_ptr<zmq::context_t> & context,
                          MonitorProvider & provider);

    ~MonitorProviderClient();
 
    void init(
            std::shared_ptr<ConfigurationService> & config,
            const std::string & serviceClass = "monitor",
            bool localized = true);

    /** shutdown the MessageLoop but make sure all requests have been
        completed beforehand. */
    void shutdown();

    /** this method is invoked periodically to query the MonitorProvider and
     * "POST" the result to the Monitor */
    void postStatus();

    /** monitored service proxy */
    MonitorProvider & provider_;

    /** flag enabling the inhibition of requests to the Monitor service */
    bool inhibit_;

    /** monitored service name */
    std::string restUrlPath_;
};

} // namespace RTBKIT

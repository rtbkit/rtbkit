/* monitor.h                                                       -*- C++ -*-
   Wolfgang Sourdeau, January 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
   
   Monitor class
*/

#pragma once

#include "soa/types/date.h"
#include "soa/service/service_base.h"
#include "soa/service/rest_request_router.h"
#include "soa/service/rest_service_endpoint.h"

#include "monitor_provider_proxy.h"

namespace RTBKIT {

struct Monitor : public ServiceBase,
                 public RestServiceEndpoint,
                 public MonitorProvidersSubscriber
{
    Monitor(std::shared_ptr<ServiceProxies> proxies,
            const std::string & serviceName = "monitor",
            int checkTimeout = 2);

    void init();

    /** return a JSON payload matching the result of "getMonitorStatus" */
    std::string restGetMonitorStatus() const;

    /** determines whether the system is working properly or not */
    bool getMonitorStatus() const;

    /** number of seconds during which the value in lastStatus is valid */
    int checkTimeout_;

    RestRequestRouter router;

    /** the last time lastStatus was modified */
    Date lastCheck;

    /** the status of the system the last time it was checked */
    bool lastStatus;

    /* callback invoked when the status of all monitor providers have been
       received */
    void onProvidersStatusLoaded(const MonitorProviderResponses & responses);

    /* helper that parse each provider response */
    bool checkProviderResponse(const MonitorProviderResponse & response) const;
};

} // namespace RTBKIT

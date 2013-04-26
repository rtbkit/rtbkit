/* monitor.h                                                       -*- C++ -*-
   Wolfgang Sourdeau, January-March 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
   
   MonitorEndpoint class
*/

#pragma once

#include <string>
#include <vector>

#include "soa/types/date.h"
#include "soa/service/service_base.h"
#include "soa/service/rest_request_router.h"
#include "soa/service/rest_service_endpoint.h"


namespace RTBKIT {

struct MonitorEndpoint : public Datacratic::ServiceBase,
                         public Datacratic::RestServiceEndpoint
{
    MonitorEndpoint(std::shared_ptr<Datacratic::ServiceProxies> proxies,
                    const std::string & serviceName = "monitor");
    ~MonitorEndpoint();

    void init(const std::vector<std::string> & providerNames);

    std::pair<std::string, std::string>
    bindTcp(const std::string& host = "");

    Datacratic::RestRequestRouter router;

    void checkServiceIndicators() const;

    /* MonitorClient interface */

    /** determines whether the system is working properly or not */
    bool getMonitorStatus() const;

    int checkTimeout_;

    /* MonitorProvider interface */
    struct MonitorProviderStatus {
        Datacratic::Date lastCheck;
        bool lastStatus;
    };

    bool postServiceIndicators(const std::string & providerName,
                               const std::string & indicatorsStr);

    std::vector<std::string> providerNames_;
    std::unordered_map<std::string, MonitorProviderStatus> providersStatus_;
};

} // namespace RTBKIT

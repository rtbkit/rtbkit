/* monitor.h                                                       -*- C++ -*-
   Wolfgang Sourdeau, January-March 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.

   MonitorEndpoint class
*/

#pragma once

#include <string>
#include <vector>

#include "monitor_indicator.h"
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

    void init(const std::vector<std::string> & providerClasses);

    std::pair<std::string, std::string>
    bindTcp(const std::string& host = "");

    /* MonitorClient interface */

    /** determines whether the system is working properly or not */
    bool getMonitorStatus() const;


    /** Human readable dump of the state of the various components */
    void dump(std::ostream& stream = std::cerr) const;

    bool postServiceIndicators(const std::string & providerName,
                               const std::string & indicatorsStr);

    Datacratic::RestRequestRouter router;
    int checkTimeout_;

    /* MonitorProvider interface */
    struct MonitorProviderStatus
    {
        Datacratic::Date lastCheck;
        bool lastStatus;
        std::string lastMessage;
    };

    std::vector<std::string> providerClasses_;

    struct ClassStatus : public std::map<std::string, MonitorProviderStatus>
    {
        bool getClassStatus(double checkTimeout) const;
        void dump(double checkTimeout, std::ostream& stream = std::cerr) const;
    };

    std::map<std::string, ClassStatus> providersStatus_;

    bool disabled;

    Datacratic::ConfigurationService::Watch selfWatch;
};

} // namespace RTBKIT

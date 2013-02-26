/* monitor_provider.cc
   Wolfgang Sourdeau, January 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
   
   Rest endpoint queried by the monitor in the provider processes
*/

#include <iostream>
#include <boost/algorithm/string/trim.hpp>
#include "soa/service/service_base.h"
#include "rtbkit/common/port_ranges.h"
#include "monitor_provider.h"

using namespace std;

namespace RTBKIT {

MonitorProviderEndpoint::
MonitorProviderEndpoint(ServiceBase & parentService,
                        MonitorProvider & provider)
    : ServiceBase("monitor-provider", parentService),
      RestServiceEndpoint(parentService.getZmqContext()),
      endpointName_(parentService.serviceName() + "/monitor-provider"),
      lastStatus_(Json::Value()),
      provider_(provider)
{
}

MonitorProviderEndpoint::
~MonitorProviderEndpoint()
{
    shutdown();
}

void
MonitorProviderEndpoint::
init()
{
    registerServiceProvider(endpointName_, { "monitorProvider" });

    auto config = getServices()->config;
    config->removePath(serviceName_);
    RestServiceEndpoint::init(config, endpointName_);

    /* rest router */
    onHandleRequest = router_.requestHandler();
    router_.description = "API for the Datacratic Monitor Service";
    router_.addHelpRoute("/", "GET");

    RestRequestRouter::OnProcessRequest statusRoute
        = [&] (const RestServiceEndpoint::ConnectionId & connection,
               const RestRequest & request,
               const RestRequestParsingContext & context) {
        // cerr << "request received..." << endl;

        connection.sendResponse(200, this->restGetServiceStatus(),
                                "application/json");
        return RestRequestRouter::MR_YES;
    };
    router_.addRoute("/status", "GET", "Return the status of the service",
                     statusRoute, Json::Value());

    /* refresh timer */
    addPeriodic("MonitorProviderEndpoint::refreshStatus", 1.0,
                std::bind(&MonitorProviderEndpoint::refreshStatus, this),
                true /* single threaded */);
}

void
MonitorProviderEndpoint::
shutdown()
{
    unregisterServiceProvider(endpointName_, { "monitorProvider" });
    RestServiceEndpoint::shutdown();
}

void
MonitorProviderEndpoint::
bindTcp()
{
    RestServiceEndpoint::bindTcp(PortRanges::zmq.monitorProvider, PortRanges::http.monitorProvider);
}

string
MonitorProviderEndpoint::
restGetServiceStatus()
{
    return boost::trim_copy(lastStatus_.toString());
}

void
MonitorProviderEndpoint::
refreshStatus()
{
    // Guard(lock);
    // cerr << "refresh" << endl;
    lastStatus_ = provider_.getMonitorIndicators();
}

} // namespace RTBKIT

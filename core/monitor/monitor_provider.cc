/* monitor_provider.cc
   Wolfgang Sourdeau, January 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
   
   Rest endpoint queried by the monitor in the provider processes
*/

#include <iostream>
#include <boost/algorithm/string/trim.hpp>
#include <jml/arch/exception_handler.h>
#include "soa/service/service_base.h"
#include "rtbkit/common/port_ranges.h"
#include "monitor_provider.h"

using namespace std;

namespace RTBKIT {

MonitorProviderClient::
MonitorProviderClient(const std::shared_ptr<zmq::context_t> & context,
                      MonitorProvider & provider)
        : RestProxy(context),
          provider_(provider),
          pendingRequest(false)
{
    restUrlPath_ = "/v1/services/" + provider.getProviderName();
    onDone = std::bind(&MonitorProviderClient::onResponseReceived, this,
                       placeholders::_1, placeholders::_2, placeholders::_3);
}

MonitorProviderClient::
~MonitorProviderClient()
{
    shutdown();
}

void
MonitorProviderClient::
init(std::shared_ptr<ConfigurationService> & config,
     const std::string & serviceName)
{
    addPeriodic("MonitorProviderClient::postStatus", 1.0,
                std::bind(&MonitorProviderClient::postStatus, this),
                true);

    RestProxy::init(config, serviceName);
}

void
MonitorProviderClient::
shutdown()
{
    sleepUntilIdle();
    RestProxy::shutdown();
}

void
MonitorProviderClient::
postStatus()
{
    Guard(requestLock);

    if (pendingRequest) {
        fprintf(stderr, "MonitorProviderClient::checkStatus: last request is"
                " still active\n");
    }
    else {
        string payload = provider_.getProviderIndicators().toString();
        pendingRequest = true;
        push(onDone, "POST", restUrlPath_, RestParams(), payload);
    }
}

void
MonitorProviderClient::
onResponseReceived(exception_ptr ext, int responseCode, const string & body)
{
    bool newStatus(false);

    if (responseCode == 200) {
        ML::Set_Trace_Exceptions notrace(false);
        try {
            Json::Value parsedBody = Json::parse(body);
            if (parsedBody.isMember("status") && parsedBody["status"] == "ok") {
                newStatus = true;
            }
        }
        catch (const Json::Exception & exc) {
        }
    }

    pendingRequest = false;
}

} // namespace RTBKIT

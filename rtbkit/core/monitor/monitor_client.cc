/* monitor_client.cc
   Wolfgang Sourdeau, January 2013
   Copyright (c) 2013 Datacratic.  All rights reserved. */

#include <jml/arch/exception_handler.h>

#include "monitor_client.h"
#include "jml/utils/exc_check.h"

using namespace std;

namespace RTBKIT {

Logging::Category MonitorClient::print("[LOG] MonitorClient");
Logging::Category MonitorClient::error("[ERROR] MonitorClient", MonitorClient::print);
Logging::Category MonitorClient::trace("[TRACE] MonitorClient", MonitorClient::print);

MonitorClient::
~MonitorClient()
{
    shutdown();
}

void
MonitorClient::
init(std::shared_ptr<ConfigurationService> & config,
     const std::string & serviceName)
{
    addPeriodic("MonitorClient::checkStatus", 0.5,
                std::bind(&MonitorClient::checkStatus, this),
                true);

    RestProxy::initServiceClass(config, serviceName, "zeromq", true);
}

void
MonitorClient::
shutdown()
{
    sleepUntilIdle();
    RestProxy::shutdown();
}

void
MonitorClient::
checkStatus()
{
    if (!testMode) {
        Guard(requestLock);

        if (lastSuccess.plusSeconds(checkTimeout_) < Date::now()) {
            push(onDone, "GET", "/v1/status");
        } else ; // too soon, lastSuccess  still considered good
    }
}

void
MonitorClient::
onResponseReceived(exception_ptr ext, int responseCode, const string & body)
{
    if (responseCode == 200) {
        ML::Set_Trace_Exceptions notrace(false);
        try {
            Json::Value parsedBody = Json::parse(body);
            if (parsedBody.isMember("status") && parsedBody["status"] == "ok") {
                lastSuccess = lastCheck;
            }
        }
        catch (const Json::Exception & exc) {
        }
    }

    lastCheck = Date::now();
}

bool
MonitorClient::
getStatus(double tolerance)
    const
{
    if (testMode) return testResponse;
    ExcCheckLessEqual(checkTimeout_, tolerance / 2, "Check timeout must be less or equal to tolerance divided by two");
    return Date::now().secondsSince(lastSuccess) < tolerance;
}

} // RTB

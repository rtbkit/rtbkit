/* monitor_client.cc
   Wolfgang Sourdeau, January 2013
   Copyright (c) 2013 Datacratic.  All rights reserved. */

#include <jml/arch/exception_handler.h>

#include "monitor_client.h"

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

    addPeriodic("MonitorClient::checkTimeout", checkTimeout_ / 2,
                std::bind(&MonitorClient::checkTimeout, this),
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

        if (pendingRequest) {
             LOG(print) << "checkStatus: last request is still active\n";
        }
        else {
            pendingRequest = true;
            push(onDone, "GET", "/v1/status");
        }
    }
}

void MonitorClient::
checkTimeout()
{
    if(lastCheck.plusSeconds(checkTimeout_) < Date::now()) {
        if(pendingRequest) {
            // We timed out, output a message that we timed out and reset pending
            LOG(print) << "checkTimeout: last request dropped" << endl;
            pendingRequest = false;
        }
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
    pendingRequest = false;
}

bool
MonitorClient::
getStatus(double tolerance)
    const
{
    if (testMode) return testResponse;
    return Date::now().secondsSince(lastSuccess) < tolerance;
}

} // RTB

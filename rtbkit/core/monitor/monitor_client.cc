/* monitor_client.cc
   Wolfgang Sourdeau, January 2013
   Copyright (c) 2013 Datacratic.  All rights reserved. */

#include <jml/arch/exception_handler.h>

#include "monitor_client.h"

using namespace std;

namespace RTBKIT {

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
    addPeriodic("MonitorClient::checkStatus", 1.0,
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

        if (pendingRequest) {
            cerr << "MonitorClient::checkStatus: last request is still active\n";
        }
        else {
            pendingRequest = true;
            push(onDone, "GET", "/v1/status");
        }
    }
}

void
MonitorClient::
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

    lastStatus = newStatus;
    lastCheck = Date::now();
    pendingRequest = false;
}

bool
MonitorClient::
getStatus()
    const
{
    bool status;

    if (testMode)
        status = testResponse;
    else {
        Date now = Date::now();

        status = (lastStatus
                  && (lastCheck.plusSeconds(checkTimeout_) >= now));
    }

    return status;
}

} // RTB

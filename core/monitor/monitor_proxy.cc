/* monitor_proxy.cc
   Wolfgang Sourdeau, January 2013
   Copyright (c) 2013 Datacratic.  All rights reserved. */

#include <jml/arch/exception_handler.h>

#include "monitor_proxy.h"

using namespace std;

namespace RTBKIT {

MonitorProxy::
~MonitorProxy()
{
    shutdown();
}

void
MonitorProxy::
init(std::shared_ptr<ConfigurationService> & config,
     const std::string & serviceName)
{
    addPeriodic("MonitorProxy::checkStatus", 1.0,
                std::bind(&MonitorProxy::checkStatus, this),
                true);

    RestProxy::init(config, serviceName);
}

void
MonitorProxy::
shutdown()
{
    RestProxy::shutdown();
    sleepUntilIdle();
}

void
MonitorProxy::
checkStatus()
{
    if (!testMode) {
        Guard(requestLock);

        if (pendingRequest) {
            cerr << "MonitorProxy::checkStatus: last request is still active\n";
        }
        else {
            pendingRequest = true;
            push(onDone, "GET", "/status");
        }
    }
}

void
MonitorProxy::
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
MonitorProxy::
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

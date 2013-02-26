/* monitor.cc
   Wolfgang Sourdeau, January 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
   
   Main monitor class
*/

#include <boost/algorithm/string/trim.hpp>
#include <jml/arch/exception_handler.h>

#include "monitor.h"

using namespace std;
using namespace RTBKIT;

/* MONITOR */

Monitor::
Monitor(std::shared_ptr<ServiceProxies> proxies,
        const std::string & serviceName,
        int checkTimeout)
  : ServiceBase(serviceName, proxies),
    RestServiceEndpoint(proxies->zmqContext),
    checkTimeout_(checkTimeout), lastStatus(false)
{
}

void
Monitor::
init()
{
    registerServiceProvider(serviceName_, { "monitor" });

    auto config = getServices()->config;
    config->removePath(serviceName_);
    RestServiceEndpoint::init(config, serviceName_);

    /* rest router */
    onHandleRequest = router.requestHandler();
    router.description = "API for the Datacratic Monitor Service";
    router.addHelpRoute("/", "GET");

    RestRequestRouter::OnProcessRequest statusRoute
        = [&] (const RestServiceEndpoint::ConnectionId & connection,
               const RestRequest & request,
               const RestRequestParsingContext & context) {
        // cerr << "request received..." << endl;

        connection.sendResponse(200, this->restGetMonitorStatus(),
                                "application/json");
        return RestRequestRouter::MR_YES;
    };
    router.addRoute("/status", "GET",
                    "Return the aggregate status of the monitored services",
                    statusRoute, Json::Value());

    RestRequestRouter::OnProcessRequest serviceDumpRoute
        = [&,config] (const RestServiceEndpoint::ConnectionId & connection,
                      const RestRequest & request,
                      const RestRequestParsingContext & context) {
        connection.sendResponse(200, config->jsonDump().toString(),
                                "application/json");
        return RestRequestRouter::MR_YES;
    };
    router.addRoute("/service-dump", "GET",
                    "Return the service entries from ZooKeeper",
                    serviceDumpRoute, Json::Value());
}

string
Monitor::
restGetMonitorStatus()
    const
{
    Json::Value jsonResponse;
    jsonResponse["status"] = getMonitorStatus() ? "ok" : "failure";

    return boost::trim_copy(jsonResponse.toString());
}

bool
Monitor::
getMonitorStatus()
    const
{
    Date now = Date::now();

    // cerr << "getMonitorStatus" << endl;

    return (lastStatus
            && (checkTimeout_ == -1
                || now <= lastCheck.plusSeconds(checkTimeout_)));
}

void
Monitor::
onProvidersStatusLoaded(const MonitorProviderResponses & responses)
{
    bool newStatus(true);

    for (const MonitorProviderResponse & response: responses) {
        newStatus &= checkProviderResponse(response);
        if (!newStatus)
            break;
    }

    // cerr << "received status responses" << endl;

    lastCheck = Date::now();
    lastStatus = newStatus;
}

bool
Monitor::
checkProviderResponse(const MonitorProviderResponse & response)
    const
{
    bool status(false);

    if (response.code == 200) {
        ML::Set_Trace_Exceptions notrace(false);
        try {
            Json::Value parsedBody = Json::parse(response.body);
            if (parsedBody.isMember("status") && parsedBody["status"] == "ok") {
                status = true;
            }
        }
        catch (...) {
            cerr << "exception during parsing of (supposedly) json response:"
                 << response.body << endl;
        }
    }

    if (!status) {
        Date now = Date::now();
        cerr << now.printClassic()
             << " - service '" << response.serviceName
             << "' is in failure mode" << endl;
    }

    return status;
}

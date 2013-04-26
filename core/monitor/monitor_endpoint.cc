/* monitor.cc
   Wolfgang Sourdeau, January 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
   
   Main monitor class
*/

#include <boost/algorithm/string/trim.hpp>
#include <jml/arch/exception_handler.h>

#include "soa/service/rest_request_binding.h"

#include "monitor_endpoint.h"

using namespace std;
using namespace ML;
using namespace Datacratic;
using namespace RTBKIT;


/* MONITOR */

MonitorEndpoint::
MonitorEndpoint(shared_ptr<ServiceProxies> proxies,
                const string & serviceName)
    : ServiceBase(serviceName, proxies),
      RestServiceEndpoint(proxies->zmqContext),
      checkTimeout_(2)
{
}

MonitorEndpoint::
~MonitorEndpoint()
{
    shutdown();
}

void
MonitorEndpoint::
init(const vector<string> & providerNames)
{
    addPeriodic("MonitorEndpoint::checkServiceIndicators", 1.0,
                bind(&MonitorEndpoint::checkServiceIndicators, this),
                true);

    providerNames_ = providerNames;
    registerServiceProvider(serviceName_, { "monitor" });

    auto config = getServices()->config;
    config->removePath(serviceName_);
    RestServiceEndpoint::init(config, serviceName_);

    /* rest router */
    onHandleRequest = router.requestHandler();
    router.description = "API for the Datacratic Monitor Service";
    router.addHelpRoute("/", "GET");

    auto & versionNode = router.addSubRouter("/v1", "version 1 of API");

    addRouteSyncReturn(versionNode,
                       "/status",
                       {"GET"},
                       "Return the health status of the system",
                       "",
                       [] (bool status) {
                           Json::Value jsonResponse;
                           jsonResponse["status"] = status ? "ok" : "failure";
                           
                           return jsonResponse;
                       },
                       &MonitorEndpoint::getMonitorStatus,
                       this);

    addRouteSyncReturn(versionNode,
                       "/service-dump",
                       {"GET"},
                       "Return the service entries from ZooKeeper",
                       "Dictionary of the the Zookeeper service entries",
                       [] (Json::Value jsonValue) {
                           return move(jsonValue);
                       },
                       &ConfigurationService::jsonDump,
                       &*config);

    auto & servicesNode
        = versionNode.addSubRouter("/services",
                                   "Operations on service states");

    for (string & providerName: providerNames_) {
        auto providerPost = [&,providerName]
            (const RestServiceEndpoint::ConnectionId & connection,
             const RestRequest & request,
             const RestRequestParsingContext & context) {
            if (this->postServiceIndicators(providerName, request.payload))
                connection.sendResponse(204, "", "application/json");
            else
                connection.sendResponse(403, "Error", "application/json");
            return RestRequestRouter::MR_YES;
        };

        servicesNode.addRoute("/" + providerName, "POST",
                              "service REST url",
                              providerPost, Json::Value());

        auto & providerStatus = providersStatus_[providerName];
        providerStatus.lastStatus = false;
        providerStatus.lastCheck = Date::fromSecondsSinceEpoch(0);
    }
}

std::pair<std::string, std::string>
MonitorEndpoint::
bindTcp(const std::string& host)
{
    return RestServiceEndpoint::bindTcp(
            getServices()->ports->getRange("monitor.zmq"),
            getServices()->ports->getRange("monitor.http"),
            host);
}

void
MonitorEndpoint::
checkServiceIndicators()
    const
{
    Date now = Date::now();

    for (const auto & it: providersStatus_) {
        const MonitorProviderStatus & status = it.second;
        if (!status.lastStatus) {
            fprintf (stderr, "%s: status of service '%s' is wrong\n",
                     now.printClassic().c_str(), it.first.c_str());
        }
        if (status.lastCheck.plusSeconds((double) checkTimeout_) <= now) {
            fprintf (stderr,
                     "%s: status of service '%s' was last updated at %s\n",
                     now.printClassic().c_str(), it.first.c_str(),
                     status.lastCheck.printClassic().c_str());
        }
    }
}

bool
MonitorEndpoint::
getMonitorStatus()
    const
{
    bool monitorStatus(true);
    Date now = Date::now();

    for (const auto & it: providersStatus_) {
        const MonitorProviderStatus & status = it.second;
        if (!status.lastStatus) {
            monitorStatus = false;
        }
        if (status.lastCheck.plusSeconds((double) checkTimeout_) <= now) {
            monitorStatus = false;
        }
    }

    return monitorStatus;
}

bool
MonitorEndpoint::
postServiceIndicators(const string & providerName,
                      const string & indicatorsStr)
{
    bool rc(false);
    bool status;

    ML::Set_Trace_Exceptions notrace(false);
    try {
        Json::Value indicators = Json::parse(indicatorsStr);
        if (indicators.type() == Json::objectValue) {
            status = (indicators.isMember("status")
                      && indicators["status"] == "ok");
            rc = true;
        }
    }
    catch (...) {
        cerr << "exception during parsing of (supposedly) json response: "
             << indicatorsStr << endl;
    }

    if (rc) {
        auto & providerStatus = providersStatus_[providerName];
        providerStatus.lastStatus = status;
        providerStatus.lastCheck = Date::now();
    }

    return rc;
}

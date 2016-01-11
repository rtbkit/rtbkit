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
      checkTimeout_(10),
      disabled(false)
{
}

MonitorEndpoint::
~MonitorEndpoint()
{
    shutdown();
}

void
MonitorEndpoint::
init(const vector<string> & providerClasses)
{
    addPeriodic("MonitorEndpoint::checkServiceIndicators", 10.0, [=] (uint64_t)
            {
                cerr << endl << endl;
                dump();
            });

    registerServiceProvider(serviceName_, { "monitor" });

    auto config = getServices()->config;
    config->removePath(serviceName_);
    RestServiceEndpoint::init(config, serviceName_);
    selfWatch.init([&](const std::string &path,
                       ConfigurationService::ChangeType change)
        {
            auto children = config->getChildren("serviceClass/" + serviceName_,
                                                selfWatch);
        });

    auto children = config->getChildren("serviceClass/" + serviceName_,
                                        selfWatch);


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

    providerClasses_ = providerClasses;

    for (const string & providerClass: providerClasses_) {

        auto providerPost = [&,providerClass]
            (const RestServiceEndpoint::ConnectionId & connection,
             const RestRequest & request,
             const RestRequestParsingContext & context) {

            if (this->postServiceIndicators(providerClass, request.payload))
                connection.sendResponse(204, "", "application/json");
            else
                connection.sendResponse(403, "Error", "application/json");
            return RestRequestRouter::MR_YES;
        };

        servicesNode.addRoute("/" + providerClass, "POST",
                              "service REST url",
                              providerPost, Json::Value());

        // Create the entry for our class with empty service list.
        (void) providersStatus_[providerClass];
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

bool
MonitorEndpoint::
getMonitorStatus()
    const
{
    if (disabled) return true;

    // If any of the classes are sick then the system is considered sick.
    for (const auto & it : providersStatus_)
        if (!it.second.getClassStatus(checkTimeout_)) return false;

    return !disabled;
}

bool
MonitorEndpoint::ClassStatus::
getClassStatus(double checkTimeout) const
{
    Date now = Date::now();

    // If any of the services are healthy then the class is considered healthy.
    for (const auto& it: *this) {
        const MonitorProviderStatus& status = it.second;

        if (!status.lastStatus) continue;
        if (status.lastCheck.plusSeconds(checkTimeout) <= now) continue;

        return true;
    }

    return false;
}

bool
MonitorEndpoint::
postServiceIndicators(const string & providerClass,
                      const string & indicatorsStr)
{
    MonitorIndicator ind;

    ML::Set_Trace_Exceptions notrace(false);
    try {
        Json::Value indJson = Json::parse(indicatorsStr);
        ind = MonitorIndicator::fromJson(indJson);
        ExcCheck(!ind.serviceName.empty(), "service name can't be empty");
    }
    catch (...) {
        cerr << "exception during parsing of (supposedly) json response: "
             << indicatorsStr << endl;
        return false;
    }

    MonitorProviderStatus status;
    status.lastCheck = Date::now();
    status.lastStatus = ind.status;
    status.lastMessage = ind.message;

    providersStatus_[providerClass][ind.serviceName] = status;
    return true;
}

void
MonitorEndpoint::
dump(ostream& stream) const
{
    for (const auto& it: providersStatus_) {
        stream << it.first << ": " << endl;
        it.second.dump(checkTimeout_, stream);
    }
}

void
MonitorEndpoint::ClassStatus::
dump(double timeout, ostream& stream) const
{
    Date now = Date::now();

    for (const auto& it: *this) {
        const MonitorProviderStatus& status = it.second;
        bool isTimeout = status.lastCheck.plusSeconds(timeout) <= now;

        stream <<
            ML::format("    %-20s %-3s %-20s %s\n",
                    it.first,
                    (status.lastStatus && !isTimeout ? "OK" : "ERR"),
                    status.lastCheck.printClassic(),
                    (isTimeout ? "Timeout" : status.lastMessage));
    }
}

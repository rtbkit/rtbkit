/* router_rest_api.cc                                              -*- C++ -*-
   Jeremy Barnes, 7 November 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Class to add an http monitor to a router.
*/

#include "router_rest_api.h"
#include "router.h"
#include "jml/utils/json_parsing.h"

using namespace std;
using namespace ML;

namespace RTBKIT {


/******************************************************************************/
/* ROUTER MONITOR CONNECTION                                                  */
/******************************************************************************/

RouterRestApiConnection::
RouterRestApiConnection(const string& name,
                        ServiceBase& parent,
                        Router* router) :
    HttpMonitorHandler(name, parent, router),
    router(router)
{
}

void
RouterRestApiConnection::
doGet(const std::string& resource)
{
    if (header.resource == "/stats")
        sendResponse(router->getStats());
    else if (header.resource == "/agents") {
        sendResponse(router->getAllAgentInfo());
    }
    else if (header.resource.find("/agent/") == 0) {
        string agentName(header.resource, 7);
        sendResponse(router->getAgentInfo(agentName));
    }
    else if (header.resource.find("/strategy/") == 0) {
        string strategyName(header.resource, 10);
        sendResponse(router->getStrategyInfo(strategyName));
    }
    else if (header.resource.find("/campaign/") == 0) {
        string campaignName(header.resource, 10);
        sendResponse(router->getCampaignInfo(campaignName));
    }
    else {
        sendErrorResponse(
                          404, "unknown GET resource '" + header.resource + "'");
    }
}

void
RouterRestApiConnection::
doPost(const std::string& resource, const std::string& payload)
{
    if (header.resource == "/validateConfig") {
        try {
            AgentConfig config;
            config.parse(payload);
            Json::Value response;
            response["status"] = "OK";
            response["config"] = config.toJson();
            sendResponse(response);
        } catch (const std::exception & exc) {
            Json::Value response;
            response["error"]
                = "exception processing request "
                + header.verb + " "
                + header.resource;
            response["exception"] = exc.what();
            sendErrorResponse(400, response);
        }
    }
    else {
        sendErrorResponse(404,
                          "unknown GET resource '" + header.resource
                          + "'");
    }
}

} // namespace RTBKIT

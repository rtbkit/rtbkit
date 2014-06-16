/* configuration_service.h                                         -*- C++ -*-
   Jeremy Barnes, 26 November 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.
*/

#pragma once

#include "soa/service/rest_service_endpoint.h"
#include "soa/service/service_base.h"
#include "soa/service/rest_request_router.h"

#include "rtbkit/core/monitor/monitor_provider.h"

namespace RTBKIT {

using namespace Datacratic;

/*****************************************************************************/
/* AGENT CONFIGURATION SERVICE                                               */
/*****************************************************************************/

/** Service that maintains and broadcasts the bidding agent configuration
    to all services that are listening for it.

    AGENTS send (via a REST interface) their configuration on startup and
    every time it changes.  They also send a heartbeat every second to
    say that they are still alive.

    SERVICES (router, post auction loop, and anything that needs to know
    how the agents are configured) connect via zeromq.  They will be
    sent all configurations on connection, and will be sent any changed
    configurations once they are changed.
*/

struct AgentConfigurationService : public RestServiceEndpoint,
                                   public ServiceBase,
                                   public MonitorProvider
{

    AgentConfigurationService(std::shared_ptr<ServiceProxies> services,
                              const std::string & serviceName = "agentConfigurationService");

    ~AgentConfigurationService()
    {
        shutdown();
        agents.shutdown();
        listeners.shutdown();
    }

    void init();

    void start()
    {
        RestServiceEndpoint::start();
        monitorProviderClient.start();
        //listeners.start();
    }

    void shutdown()
    {
        RestServiceEndpoint::shutdown();
        agents.shutdown();
        listeners.shutdown();
        monitorProviderClient.shutdown();
    }

    void unsafeDisableMonitor() {
        monitorProviderClient.disable();
    }

    /// Handler for POST /v1/agents/<name>/heartbeat
    void handleAgentHeartbeat(const std::string & agent);
    
    /// Handler for POST /v1/agents/<name>/
    void handleAgentConfig(const std::string & agent,
                           const Json::Value & config);

    void handleDeleteConfig(const std::string & agent);

    /// Handler for GET /v1/agents/<name>/
    Json::Value handleGetAgent(const std::string & agent) const;

    /// Handler for GET /v1/agents/
    std::vector<std::string> handleGetAgentList() const;

    /// Handler for GET /v1/agents/all
    Json::Value handleGetAllAgents() const;

    /// Bind to tcp/ip ports and listen
    void bindTcp();

    RestRequestRouter router;

    ZmqNamedClientBus agents;

    /** Services connect via a zeromq bus as they need to be pushed
        notifications of changes.
    */
    ZmqNamedClientBus listeners;

    struct ListenerInfo {
    };
    
    std::unordered_map<std::string, ListenerInfo> listenerInfo;

    struct AgentInfo {
        Json::Value config;
        std::string configStr;
        Date lastHeartbeat;
    };

    std::unordered_map<std::string, AgentInfo> agentInfo;

    /* Reponds to Monitor requests */
    MonitorProviderClient monitorProviderClient;

    /* MonitorProvider interface */
    std::string getProviderClass() const;
    MonitorIndicator getProviderIndicators() const;
};

} // namespace RTBKIT


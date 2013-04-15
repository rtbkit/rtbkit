/* agent_configuration_service.cc
   Jeremy Barnes, 26 November 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Implementation of the configuration service.
*/

#include "jml/utils/string_functions.h"
#include "agent_configuration_service.h"
#include "soa/service/rest_request_binding.h"

using namespace std;
using namespace ML;

namespace RTBKIT {


/*****************************************************************************/
/* AGENT CONFIGURATION SERVICE                                               */
/*****************************************************************************/

AgentConfigurationService::
AgentConfigurationService(std::shared_ptr<ServiceProxies> services,
                          const std::string & serviceName)
    : RestServiceEndpoint(services->zmqContext), 
      ServiceBase(serviceName, services),
      agents(services->zmqContext),
      listeners(services->zmqContext),
      monitorProviderClient(services->zmqContext, *this)
{
}

void
AgentConfigurationService::
init()
{
    getServices()->config->removePath(serviceName());

    registerServiceProvider(serviceName(), { "rtbAgentConfiguration" });
    

    //registerService();

    agents.onConnection = [=] (const std::string & agent)
        {
            cerr << "got new agent " << hexify_string(agent) << endl;
        };

    agents.onDisconnection = [=] (const std::string & agent)
        {
            cerr << "lost agent " << hexify_string(agent) << endl;
            // Broadcast the disconnection to all listeners
            for (auto & l: listenerInfo)
                listeners.sendMessage(l.first, "CONFIG", agent, "");
        };

    listeners.onConnection = [=] (const std::string & listener)
        {
            cerr << "got new listener " << hexify_string(listener) << endl;

            listenerInfo.insert(make_pair(listener, ListenerInfo()));

            // we got a new listener...
            for (auto & a: agentInfo) {
                if (!a.second.config.isNull())
                    listeners.sendMessage(listener, "CONFIG", a.first,
                                          a.second.config.toString());
            }
        };

    listeners.onDisconnection = [=] (const std::string & listener)
        {
            cerr << "lost listener " << hexify_string(listener) << endl;

            listenerInfo.erase(listener);
        };

    listeners.clientMessageHandler = [=] (const std::vector<std::string> & message)
        {
            cerr << "listeners got client message " << message << endl;
            throw ML::Exception("unexpected listener message");
            //const std::string & agent = message.at(2);
            //Json::Value j = Json::parse(message.at(3));

            //handleAgentConfig(agent, j);
        };

    agents.clientMessageHandler = [=] (const std::vector<std::string> & message)
        {
            //cerr << "agent message " << message << endl;
            const std::string & agent = message.at(2);
            Json::Value j = Json::parse(message.at(3));

            handleAgentConfig(agent, j);
        };

    RestServiceEndpoint::init(getServices()->config, serviceName() + "/rest");
    listeners.init(getServices()->config, serviceName() + "/listen");
    agents.init(getServices()->config, serviceName() + "/agents");

    onHandleRequest = router.requestHandler();

    router.description = "API for the Datacratic Bidding Agent Configuration Service";

    router.addHelpRoute("/", "GET");

    auto & versionNode = router.addSubRouter("/v1", "version 1 of API");
    auto & agentsNode
        = versionNode.addSubRouter("/agents",
                                   "Operations on agents");
    
    addRouteSyncReturn(agentsNode,
                       "/",
                       {"GET"},
                       "List all agents that are configured",
                       "Array of names",
                       [] (const std::vector<std::string> & v) { return jsonEncode(v); },
                       &AgentConfigurationService::handleGetAgentList,
                       this);
    
    addRouteSyncReturn(agentsNode,
                       "/all",
                       {"GET"},
                       "Return the configuration of the all agents",
                       "Dictionary of configuration of all agents",
                       [] (const Json::Value & config) { return config; },
                       &AgentConfigurationService::handleGetAllAgents,
                       this);


    auto & agent
        = agentsNode.addSubRouter(Rx("/([^/]*)", "/<agentName>"),
                                  "operations on an individual agent");
    
    RequestParam<std::string> agentKeyParam(-2, "<agentName>", "agent to operate on");
    
    addRouteSyncReturn(agent,
                       "/config",
                       {"GET"},
                       "Return the configuration of the given agent",
                       "Representation of the named agent",
                       [] (const Json::Value & config) { return config; },
                       &AgentConfigurationService::handleGetAgent,
                       this,
                       agentKeyParam);

    addRouteSync(agent,
                 "/config",
                 {"POST"},
                 "Set the configuration of the given agent",
                 &AgentConfigurationService::handleAgentConfig,
                 this,
                 agentKeyParam,
                 JsonParam<Json::Value>("", "Configuration block for agent"));

    addRouteSync(agent,
                 "/heartbeat",
                 {"POST"},
                 "Send a heartbeat for the agent",
                 &AgentConfigurationService::handleAgentHeartbeat,
                 this,
                 agentKeyParam);


    addSource("AgentConfigurationService::agents", agents);
    addSource("AgentConfigurationService::listeners", listeners);

    monitorProviderClient.init(getServices()->config);
}

void
AgentConfigurationService::
bindTcp()
{
    RestServiceEndpoint::bindTcp(
            getServices()->ports->getRange("agentConfiguration.zmq"),
            getServices()->ports->getRange("agentConfiguration.http"));
    listeners.bindTcp(getServices()->ports->getRange("configuration"));
    agents.bindTcp();
}

Json::Value
AgentConfigurationService::
handleGetAgent(const std::string & agent) const
{
    auto it = agentInfo.find(agent);
    if (it == agentInfo.end())
        return Json::Value();
    return it->second.config;
}

Json::Value
AgentConfigurationService::
handleGetAllAgents() const
{
    Json::Value result;
    for (auto & c: agentInfo)
        result[c.first] = c.second.config;
    return result;
}

std::vector<std::string>
AgentConfigurationService::
handleGetAgentList() const
{
    vector<string> result;
    for (auto & c: agentInfo)
        result.push_back(c.first);
    return result;
}

void
AgentConfigurationService::
handleAgentConfig(const std::string & agent,
                  const Json::Value & config)
{
    auto & info = agentInfo[agent];
    info.lastHeartbeat = Date::now();
    
    // If the configuration didn't change, we don't need to broadcast it
    if (info.config == config)
        return;

    info.config = config;

    // Broadcast the configuration to all listeners
    for (auto & l: listenerInfo)
        listeners.sendMessage(l.first, "CONFIG", agent, config.toString());
}

void
AgentConfigurationService::
handleAgentHeartbeat(const std::string & agent)
{
}

/** MonitorProvider interface */
string
AgentConfigurationService::
getProviderName()
    const
{
    return serviceName();
}

Json::Value
AgentConfigurationService::
getProviderIndicators()
    const
{
    Json::Value value;

    /* MB health check:
       - no error occurred in last save (implying Redis conn is alive) */
    Date now = Date::now();
    bool status(true);
    value["status"] = status ? "ok" : "failure";

    return value;
}

} // namespace RTBKIT

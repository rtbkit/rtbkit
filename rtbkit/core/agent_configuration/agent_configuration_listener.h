/* configuration_listener.h                                        -*- C++ -*-
   Jeremy Barnes, 26 November 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Agent that listens for configuration.
*/

#pragma once

#include "soa/service/zmq_endpoint.h"
#include "rtbkit/core/router/router_types.h"
#include "soa/gc/rcu_protected.h"


namespace RTBKIT {


/*****************************************************************************/
/* AGENT CONFIG ENTRY                                                        */
/*****************************************************************************/

/** A single entry in the agent info structure. */
struct AgentConfigEntry {
    std::string name;
    std::shared_ptr<const AgentConfig> config;

    bool valid() const { return !!config; }

    /** JSON version for debugging. */
    Json::Value toJson() const;
};


/** A read-only structure with information about all of the agents so
    that auctions can scan them without worrying about data dependencies.
    Uses RCU.
*/
struct AllAgentConfig : public std::vector<AgentConfigEntry> {
    std::unordered_map<std::string, int> agentIndex;
    std::unordered_map<AccountKey, std::vector<int> > accountIndex;
};


/*****************************************************************************/
/* AGENT CONFIGURATION LISTENER                                              */
/*****************************************************************************/

struct AgentConfigurationListener: public MessageLoop {

    AgentConfigurationListener(std::shared_ptr<zmq::context_t> context)
        : allAgents(new AllAgentConfig()), configEndpoint(context)
    {
    }

    ~AgentConfigurationListener()
    {
        delete allAgents;
        shutdown();
    }

    /** Type of function to be called on a configuration change.  If an agent
        has disappeared, the config pointer will be null.
    */
    typedef std::function<void (std::string, std::shared_ptr<const AgentConfig>)>
    OnConfigChange;

    OnConfigChange onConfigChange;

    void init(std::shared_ptr<ConfigurationService> config)
    {
        configEndpoint.init(config);
        configEndpoint.connectToServiceClass("rtbAgentConfiguration",
                                             "listen");
        configEndpoint.messageHandler
            = [=] (const std::vector<std::string> & message)
            {
                if (message.size() > 1) {
                    using namespace std;
                    cerr << "got configuration message " << message[1] << endl;
                }
                this->onMessage(message);
            };
        addSource("AgentConfigurationListener::configEndpoint", configEndpoint);
    }

    void start()
    {
        MessageLoop::start();
    }

    void shutdown()
    {
        MessageLoop::shutdown();
        configEndpoint.shutdown();
    }

    /*************************************************************************/
    /* AGENT INTERACTIONS                                                    */
    /*************************************************************************/

    typedef std::function<void (const AgentConfigEntry & info)> OnAgentFn;
    /** Call the given callback for each agent. */
    void forEachAgent(const OnAgentFn & onAgent) const;

    /** Call the given callback for each agent that is bidding on the given
        account.
    */
    void forEachAccountAgent(const AccountKey & account,
                             const OnAgentFn & onAgent) const;

    /** Find the information entry for the given agent.  All elements are
        guaranteed to be valid until the object is destroyed.
    */
    AgentConfigEntry getAgentEntry(const std::string & agent) const;

private:
    void onMessage(const std::vector<std::string> & message);

    AllAgentConfig * allAgents;
    mutable GcLock allAgentsGc;

    ZmqNamedClientBusProxy configEndpoint;
};

struct AgentBridge {
    AgentBridge(std::shared_ptr<zmq::context_t> context) :
        agents(context) {
    }

    void shutdown() {
        agents.shutdown();
    }

    /// Messages to the agents go out on this
    ZmqNamedClientBus agents;

    /** Send the given message to the given bidding agent. */
    template<typename... Args>
    void sendAgentMessage(const std::string & agent,
                          const std::string & messageType,
                          const Date & date,
                          Args... args)
    {
        agents.sendMessage(agent, messageType, date,
                           std::forward<Args>(args)...);
    }

    /** Send the given message to the given bidding agent. */
    template<typename... Args>
    void sendAgentMessage(const std::string & agent,
                          const std::string & eventType,
                          const std::string & messageType,
                          const Date & date,
                          Args... args)
    {
        agents.sendMessage(agent, eventType, messageType, date,
                           std::forward<Args>(args)...);
    }
};


} // namespace RTBKIT


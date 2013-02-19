/* agent_configuration_listener.cc
   Jeremy Barnes, 26 November 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Listener for configuration.
*/

#include "agent_configuration_listener.h"
#include "agent_config.h"

namespace RTBKIT {


/*****************************************************************************/
/* AGENT CONFIGURATION LISTENER                                              */
/*****************************************************************************/


void
AgentConfigurationListener::
forEachAgent(const OnAgentFn & onAgent) const
{
    GcLock::SharedGuard guard(allAgentsGc);
    const AllAgentConfig * ac = allAgents;
    if (!ac) return;

    std::for_each(ac->begin(), ac->end(), onAgent);
}

void
AgentConfigurationListener::
forEachAccountAgent(const AccountKey & account,
                    const OnAgentFn & onAgent) const
{
    GcLock::SharedGuard guard(allAgentsGc);
    const AllAgentConfig * ac = allAgents;
    if (!ac) return;

    auto it = ac->accountIndex.find(account);
    if (it == ac->accountIndex.end())
        return;

    for (auto jt = it->second.begin(), jend = it->second.end();
         jt != jend;  ++jt)
        onAgent(ac->at(*jt));
}

AgentConfigEntry
AgentConfigurationListener::
getAgentEntry(const std::string & agent) const
{
    GcLock::SharedGuard guard(allAgentsGc);
    const AllAgentConfig * ac = allAgents;
    if (!ac) return AgentConfigEntry();

    auto it = ac->agentIndex.find(agent);
    if (it == ac->agentIndex.end())
        return AgentConfigEntry();
    return ac->at(it->second);
}

void
AgentConfigurationListener::
onMessage(const std::vector<std::string> & message)
{
    using namespace std;

    const std::string & topic = message.at(0);
    if (topic != "CONFIG") {
        cerr << "unknown message for agent configuration listener" << endl;
        cerr << message;
        return;
    }
    const std::string & agent = message.at(1);
    const std::string & configStr = message.at(2);

    std::shared_ptr<AgentConfig> config;

    if (!configStr.empty()) {
        Json::Value j = Json::parse(configStr);
        config.reset(new AgentConfig(AgentConfig::createFromJson(j)));
    }

    /* Now, update the current configuration list */

    GcLock::SharedGuard guard(allAgentsGc);
    AllAgentConfig * ac = allAgents;
    
    /* Create a new object and copy the old ones in */
    std::unique_ptr<AllAgentConfig> newConfig(new AllAgentConfig());
    bool found = false;
    for (auto & c: *ac) {
        if (c.name == agent) {
            found = true;
            if (config) {
                AgentConfigEntry ce = c;
                ce.config = config;
                newConfig->emplace_back(ce);
            }
            else continue;
        }
        else newConfig->emplace_back(c);

        int i = newConfig->size() - 1;
        newConfig->agentIndex[c.name] = i;
        newConfig->accountIndex[newConfig->back().config->account].push_back(i);
    }
    if (!found && config) {
        AgentConfigEntry ce;
        ce.name = agent;
        ce.config = config;
        newConfig->emplace_back(ce);

        int i = newConfig->size() - 1;
        newConfig->agentIndex[agent] = i;
        newConfig->accountIndex[newConfig->back().config->account].push_back(i);
    }

    if (ML::cmp_xchg(allAgents, ac, (AllAgentConfig *)newConfig.get())) {
        newConfig.release();
        ExcAssertNotEqual(ac, allAgents);
        if (ac)
            allAgentsGc.defer([=] () { delete ac; });
    }
    else {
        throw ML::Exception("cmp_exch failed for AgentConfigurationListener");
    }

    if (onConfigChange)
        onConfigChange(agent, config);
}


} // namespace RTBKIT

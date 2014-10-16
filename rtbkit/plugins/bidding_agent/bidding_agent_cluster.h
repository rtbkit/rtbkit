/*
 * bidding_agent_cluster.h
 *
 *  Created on: Oct 16, 2013
 *      Author: rtbkit
 */

#ifndef BIDDING_AGENT_CLUSTER_H_
#define BIDDING_AGENT_CLUSTER_H_

#include <mutex>
#include <memory>
#include <string>
#include <utility>
#include <functional>
#include <unordered_map>
#include "soa/service/message_loop.h"
#include "soa/service/service_base.h"
#include "rtbkit/plugins/bidding_agent/bidding_agent.h"

namespace RTBKIT {

/**
 * Tiny wrapper meant to allow a cluster of bidding agents
 * to share a same poll loop.
 * The shared poll loop belong to the cluster (which is a MessageLoop)
 * Limited groupped operations are supported: (init, shutdown)
 */
class BiddingAgentCluster: 
    public Datacratic::ServiceBase,
    public Datacratic::MessageLoop
{
public:
    typedef std::shared_ptr<BiddingAgent> agent_ptr;

    BiddingAgentCluster(std::shared_ptr<ServiceProxies> proxies,
                        const std::string & name = "bidding_agent_cluster",
                        int numThreads = 1,
                        double maxAddedLatency = 0.0005)
        : ServiceBase (name, proxies)
        , MessageLoop (numThreads,maxAddedLatency)
    {
    }

    BiddingAgentCluster(ServiceBase & parent,
                        const std::string & name = "bidding_agent_cluster",
                        int numThreads = 1,
                        double maxAddedLatency = 0.0005)
        : ServiceBase (name, parent)
        , MessageLoop (numThreads,maxAddedLatency)
    {
    }

    virtual ~BiddingAgentCluster () { }

    /** call init() on every agent in the cluster */
    void init ()
    {
        for (auto& agent: agents_)
            if (agent.second.get())
                agent.second->init ();
    }

    /** call shutdown() on every agent in the cluster; */
    void shutdown ()
    {
        for (auto& agent: agents_)
            if (agent.second.get())
                agent.second->shutdown();
    }

    /** attempt to add new_member in the cluster. return true when success */
    bool join (const agent_ptr& new_member, const std::string& name)
    {
        std::lock_guard<std::mutex> l(mtx_);
        auto it = agents_.insert ({name, new_member});
        if (it.second)
            this->addSource (name, new_member);
        return it.second ;
    }

    /** attempt to remove member from the cluster. return true when success */
    bool leave (const std::string& name)
    {
        std::lock_guard<std::mutex> l(mtx_);
        auto it = agents_.find(name);
        if (it != std::end(agents_)) {
            removeSourceSync(it->second.get());
            it->second->shutdown();
            agents_.erase(it);
            return true;
        }

        return false;
    }

    /** returns true if agent is member of our cluster */
    bool is_member (const std::string & name) const
    {
        std::lock_guard<std::mutex> l(mtx_);
        return agents_.find (name) != agents_.end();
    }

    /** probably useless */
    agent_ptr operator[] (const std::string& name) const
    {
        std::lock_guard<std::mutex> l(mtx_);
        auto it = agents_.find(name);
        return it != agents_.end() ? (*it).second : agent_ptr();
    }

private:
    mutable std::mutex mtx_;
    std::unordered_map<std::string, std::shared_ptr<BiddingAgent>> agents_;
};

} /* namespace RTBKIT */
#endif /* BIDDING_AGENT_CLUSTER_H_ */

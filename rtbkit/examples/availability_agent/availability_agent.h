/** availability_agent.h                                 -*- C++ -*-
    Rémi Attab, 27 Jul 2012
    Copyright (c) 2012 Recoset.  All rights reserved.

    Service thingy for the sprocket.

*/

#ifndef __rtb__availability_agent_h__
#define __rtb__availability_agent_h__

#include "availability_check.h"
#include "soa/service/service_base.h"
#include "rtbkit/plugins/bidding_agent/bidding_agent.h"
#include "jml/arch/cmp_xchg.h"
#include "jml/arch/atomic_ops.h"

#include <memory>
#include <thread>


namespace Datacratic {


/******************************************************************************/
/* AVAILABILITY SERVICE                                                       */
/******************************************************************************/

/** Agent used to sample a bid request stream (without actually bidding) and use
    the recorded bid requests to make availability calculations.
*/
struct AvailabilityAgent : public ServiceBase
{
    AvailabilityAgent(
            const std::shared_ptr<ServiceProxies> services,
            const std::string& name = "availability_agent");

    ~AvailabilityAgent() { shutdown(); }

    void start();
    void shutdown();

    /** The size of the ring buffer used to hold the bid requests for
        availability calculation. See AvailabilityCheck for more details.
    */
    void setRequestBufferSize(size_t size) { ringBufferSize = size; }

    /** Portion of the router's bid request stream that should be directed
        toward our agent. It's also used to calculate the qps at the router
        level so it's important that it set low enough so that the agent can
        keep up with the bid request stream (the maxInFlight is set very high).
    */
    void setBidProbability(double prob) { bidProbability = prob; }

    /** Reports the effectiveness of each filters employed by the router. */
    Json::Value checkConfig(const RTBKIT::AgentConfig& config);
    Json::Value checkConfig(const std::string& config);
    Json::Value checkConfig(Json::Value config);

    // Required the make the template gods happy. Remove at your own peril.
    Json::Value checkConfigJson(const Json::Value& config) {
        return checkConfig(config);
    }


    /** Stats for the agent. */
    struct Stats {

        Stats() :
            checks(0), newRequests(0), ringSize(0),
            relativeQps(0), absoluteQps(0)
        {}

        uint32_t checks;
        uint32_t newRequests;
        uint64_t ringSize;

        /** relativeQps is used as the accumulator to calculate the absoluteQps
            and is sampled periodically. Because it's not stable it should not
            be dumped to carbon or the console. Use the absoluteQps instead
            which is adjusted for the bidProbability.
        */
        uint64_t relativeQps;
        uint64_t absoluteQps;

        Json::Value toJson() const
        {
            Json::Value json(Json::objectValue);

            json["checks"] = checks;
            json["newRequests"] = newRequests;
            json["ringSize"] = ringSize;
            json["relativeQps"] = relativeQps;
            json["absoluteQps"] = absoluteQps;

            return json;
        }

    };
    Stats getStats() const { return stats; }

    /** Write a stat to the CLI */
    void dumpToCLI(const std::string& stat, uint64_t value) const;

private:
    Stats stats;

    void addBaseConfig(Json::Value& config);

    void doConfig();
    void onBidRequest(
            double timestamp,
            Id id,
            std::shared_ptr<RTBKIT::BidRequest> bidRequest,
            const RTBKIT::Bids & bids,
            double timeLeftMs,
            Json::Value augmentations);

    void doQpsThread();

    bool serviceRunning;
    size_t ringBufferSize;
    double bidProbability;

    std::unique_ptr<AvailabilityCheck> checker;
    RTBKIT::BiddingAgent proxy;

    std::unique_ptr<std::thread>(qpsThread);
};


} // Recoset

#endif // __rtb__availability_agent_h__

/** availability_agent.cc                                 -*- C++ -*-
    Rémi Attab, 27 Jul 2012
    Copyright (c) 2012 Recoset.  All rights reserved.

    42

*/


#include "availability_agent.h"
#include "jml/arch/format.h"
#include "jml/arch/timers.h"

#include <boost/bind.hpp>
#include <iostream>
#include <array>

using namespace std;
using namespace ML;
using namespace RTBKIT;


namespace Datacratic {


AvailabilityAgent::
AvailabilityAgent(
        const std::shared_ptr<ServiceProxies> services, const string& name) :
    ServiceBase(name, services),
    serviceRunning(false),
    ringBufferSize(100000),
    bidProbability(0.1),
    proxy(*this)
{
    proxy.strictMode(false);
    proxy.onBidRequest = boost::bind(
            &AvailabilityAgent::onBidRequest, this, _1, _2, _3, _4, _5, _6);

    proxy.onError = [&](double ts, string desc, vector<string> err) {
        cerr << "ERROR: (" << ts << ", " << desc << ") -> { ";
        for (size_t i = 0; i < err.size(); ++i)
            cerr << err[i] << " ";
        cerr << " }" << endl;

    };

    recordHit("up");
}


void
AvailabilityAgent::
start()
{
    if (serviceRunning) return;
    serviceRunning = true;

    checker.reset(new AvailabilityCheck(ringBufferSize));
    checker->onEvent = [&](const string& ev, StatEventType type, float val) {
        recordEvent(ev.c_str(), type, val);

        if (ev == "newRequests") ML::atomic_inc(stats.newRequests);
        else if (ev == "checks") ML::atomic_inc(stats.checks);
        else if (ev == "ringSize") stats.ringSize = val;
        else if (type != ET_OUTCOME) dumpToCLI(ev, val);
    };

    qpsThread.reset(new thread([&]{ doQpsThread(); }));

    proxy.init();
    proxy.start();
    doConfig();
}


void
AvailabilityAgent::
shutdown()
{
    if (!serviceRunning) return;
    serviceRunning = false;

    proxy.shutdown();

    qpsThread->join();
    qpsThread.reset();

    checker.reset();
}

void
AvailabilityAgent::
addBaseConfig(Json::Value& config)
{
    config["account"] = AccountKey("availability_agent").toJson();
    config["maxInFlight"] = 100000;
    config["bidProbability"] = bidProbability;

    config["roundRobin"] = Json::Value(Json::objectValue);
    config["roundRobin"]["group"] = "availability_agent";
    config["roundRobin"]["weight"] = 1;

    auto makeCreative = [&](int id, const string& name, int width, int height)
        -> Json::Value
    {
        Json::Value creative(Json::objectValue);
        creative["id"] = id;
        creative["name"] = name;
        creative["width"] = width;
        creative["height"] = height;
        return creative;
    };

    config["creatives"] = Json::Value(Json::arrayValue);
    config["creatives"].append(makeCreative(1, "p1", 300, 250));
    config["creatives"].append(makeCreative(2, "p2", 728, 90));
    config["creatives"].append(makeCreative(3, "p3", 160, 600));
}


Json::Value
AvailabilityAgent::
checkConfig(const std::string& str)
{
    Json::Value json;
    Json::Reader reader;
    reader.parse(str, json);

    addBaseConfig(json);
    return checkConfig(AgentConfig::createFromJson(json));
}

Json::Value
AvailabilityAgent::
checkConfig(Json::Value json)
{
    addBaseConfig(json);
    return checkConfig(AgentConfig::createFromJson(json));
}

Json::Value
AvailabilityAgent::
checkConfig(const AgentConfig& config)
{
    Json::Value result = checker->checkConfig(config);
    uint64_t routerQps = stats.absoluteQps;

    double total = result["total"].asInt();
    double biddable = result["biddable"].asInt();

    double biddablePct = total > 0.0 ? biddable / total : 0.0;
    uint64_t biddableQps = biddablePct * routerQps;

    result["routerQps"] = routerQps;
    result["biddablePct"] = biddablePct;
    result["biddableQps"] = biddableQps;

    return result;
}


// Lifted from the rtb/agent/strategy/abstract_strategy.coffee
void
AvailabilityAgent::
doConfig()
{
    Json::Value config(Json::objectValue);
    addBaseConfig(config);
    proxy.doConfigJson(config);
}


void
AvailabilityAgent::
onBidRequest(
        double timestamp,
        Id id,
        std::shared_ptr<BidRequest> bidRequest,
        const Bids & bids,
        double timeLeftMs,
        Json::Value augmentations)
{
    proxy.doBid(id, bids, Json::Value());

    checker->addRequest(*bidRequest);
    ML::atomic_inc(stats.relativeQps);
}

void
AvailabilityAgent::
doQpsThread()
{
    // Used to smooth the absolute qps because otherwise it's very spiky.
    array<uint64_t, 120> avgBuffer;
    fill(avgBuffer.begin(), avgBuffer.end(), 0);
    size_t pos = 0;
    bool full = false;

    while(serviceRunning) {
        this_thread::sleep_for(chrono::seconds(1));

        // Atomically sample the relative qps.
        uint64_t relative;
        do {
            relative = stats.relativeQps;
        } while (!ML::cmp_xchg(stats.relativeQps, relative, (uint64_t)0));

        // Updated the ring buffer.
        avgBuffer[pos] = ((double)relative) / bidProbability;
        pos = (pos + 1) % avgBuffer.size();
        full = full || pos == 0;

        // Calc absolute using avg of relative samplings.
        auto itFirst = avgBuffer.begin();
        auto itLast = full ? avgBuffer.end() : avgBuffer.begin() + pos;
        uint64_t sum = accumulate(itFirst, itLast, 0);
        stats.absoluteQps = sum / avgBuffer.size();

        recordLevel(stats.absoluteQps, "qps");
    }
}

void
AvailabilityAgent::
dumpToCLI(const string& stat, uint64_t value) const
{
    cerr << ML::format("%'16ld %s\n", value, stat);
};


} // namepsace Recoset

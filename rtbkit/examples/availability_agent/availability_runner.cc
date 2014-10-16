/** availability_runner.cc                                 -*- C++ -*-
    Rémi Attab, 30 Jul 2012
    Copyright (c) 2012 Recoset.  All rights reserved.

    Command line interface for the availability agent.

*/

#include "availability_agent.h"
#include "soa/service/http_monitor.h"
#include "soa/service/process_stats.h"
#include "soa/service/rest_request_router.h"
#include "soa/service/rest_request_binding.h"
#include "soa/service/rest_service_endpoint.h"
#include "soa/service/service_utils.h"
#include "jml/arch/timers.h"

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <thread>
#include <chrono>

using namespace std;
using namespace ML;
using namespace Datacratic;

/******************************************************************************/
/* AVAILABILITY MONITOR                                                       */
/******************************************************************************/

struct AvailabilityMonitor : public ServiceBase, public RestServiceEndpoint
{
    AvailabilityMonitor(AvailabilityAgent& agent_) :
        ServiceBase("monitor", agent_),
        RestServiceEndpoint(getServices()->zmqContext),
        agent(agent_)
    {
        init(getServices()->config, "monitor");

        addRouteSyncReturn(
                restRouter,
                "/ping",
                {"GET"},
                "Returns stats to indicate that the service is running correctly.",
                "Accumulated stats of the service since it started.",
                [] (const AvailabilityAgent::Stats& stats) {return stats.toJson();},
                &AvailabilityAgent::getStats,
                &agent);

        addRouteSyncReturn(
                restRouter,
                "/query",
                {"POST"},
                "Checks the given config against the availability checker.",
                "The result of the availability check",
                [] (const Json::Value& ret) { return ret; },
                &AvailabilityAgent::checkConfigJson,
                &agent,
                JsonParam<Json::Value>("", "Configuration to check"));
    }

private:

    // override of RestServiceEndpoint.
    virtual void handleRequest(
            const ConnectionId & conn, const RestRequest & request) const
    {
        restRouter.handleRequest(conn, request);
    }

    AvailabilityAgent& agent;
    RestRequestRouter restRouter;
};


/******************************************************************************/
/* MAIN                                                                       */
/******************************************************************************/

int main(int argc, char** argv)
{
    ServiceProxyArguments serviceArgs;

    float bidProbability = 1.0;
    size_t ringSize = 100000;

    string restHost = "";

    using namespace boost::program_options;

    options_description checkOptions("Checker Options");
    checkOptions.add_options()
        ("bid-probability", value<float>(&bidProbability),
                "Bid probability passed to the router")

        ("ring-size", value<size_t>(&ringSize),
                "Number of bid requests to keep around");


    options_description restOptions("REST Options");
    restOptions.add_options()
        ("rest-host", value<string>(&restHost),
                "Host of the REST interface for the service");


    options_description allOptions("All Options");
    allOptions
        .add(serviceArgs.makeProgramOptions())
        .add(checkOptions)
        .add(restOptions);
    allOptions.add_options() ("help,h", "Print this message");

    variables_map vm;
    store(command_line_parser(argc, argv).options(allOptions).run(), vm);
    notify(vm);

    if (vm.count("help")) {
        cerr << allOptions << endl;
        return 1;
    }

    /** Agent Setup */
    auto proxies = serviceArgs.makeServiceProxies();
    AvailabilityAgent agent(proxies, "availability_agent");

    if (vm.count("bid-probability")) agent.setBidProbability(bidProbability);
    if (vm.count("ring-size")) agent.setRequestBufferSize(ringSize);

    agent.start();

    AvailabilityMonitor monitor(agent);
    monitor.bindTcp(
            proxies->ports->getRange("availability.zmq"),
            proxies->ports->getRange("availability.http"),
            restHost);
    monitor.start();

    /** Process Stats loop */
    ProcessStats lastProcStats;
    AvailabilityAgent::Stats lastAgentStats = agent.getStats();

    while (true) {
        this_thread::sleep_for(chrono::seconds(10));

        auto onStats = [&](const string& stat, double value) {
            agent.recordLevel(value, stat);
            agent.dumpToCLI(stat, value);
        };

        ProcessStats curStats;
        ProcessStats::logToCallback(onStats, lastProcStats, curStats, "process");
        lastProcStats = curStats;

        AvailabilityAgent::Stats stats = agent.getStats();
        agent.dumpToCLI("ringSize", stats.ringSize);
        agent.dumpToCLI("qps", stats.absoluteQps);
        agent.dumpToCLI("checks", stats.checks - lastAgentStats.checks);
        agent.dumpToCLI(
                "newRequests", stats.newRequests - lastAgentStats.newRequests);
        lastAgentStats = stats;

        cerr << endl;
    };
}

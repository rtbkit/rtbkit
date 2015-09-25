/* bid_request_endpoint_test.cc
   Eric Robert, 9 May 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Tool to test a bid request endpoint.
*/

#include <memory>
#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include "rtbkit/core/router/router.h"
#include "rtbkit/core/agent_configuration/agent_configuration_service.h"
#include "rtbkit/core/banker/null_banker.h"
#include "rtbkit/plugins/bidding_agent/bidding_agent_cluster.h"
#include "rtbkit/testing/test_agent.h"
#include "soa/service/message_loop.h"
#include "jml/utils/file_functions.h"

using namespace Datacratic;
using namespace RTBKIT;
using namespace std ;

static Json::Value loadJsonFromFile(const std::string & filename) {
    ML::File_Read_Buffer buf(filename);
    return Json::parse(std::string(buf.start(), buf.end()));
}

string tid()
{
    ostringstream oss ;
    oss << "0x" << hex <<  this_thread::get_id();
    return oss.str();
}


int main(int argc, char ** argv)
{
    using namespace boost::program_options;

    int recordCount = 1000;
    std::string recordFile;
    std::string exchangeConfigurationFile = "rtbkit/examples/router-config.json";

    options_description endpoint_options("Bid request endpoint options");
    endpoint_options.add_options()
    ("record-request,r", value<std::string>(&recordFile),
     "filename prefix to record incoming bid request")
    ("record-count,n", value<int>(&recordCount),
     "number of bid request recorded by file")
    ("exchange-configuration,x", value<std::string>(&exchangeConfigurationFile),
     "configuration file with exchange data");

    options_description all_opt;
    all_opt
    .add(endpoint_options);
    all_opt.add_options()
    ("help,h", "print this message");

    variables_map vm;
    store(command_line_parser(argc, argv)
          .options(all_opt)
          .run(),
          vm);
    notify(vm);

    if(vm.count("help")) {
        std::cerr << all_opt << std::endl;
        exit(1);
    }

    std::shared_ptr<ServiceProxies> proxies(new ServiceProxies());

    // The agent config service lets the router know how our agent is
    // configured
    AgentConfigurationService agentConfig(proxies, "config");
    agentConfig.unsafeDisableMonitor();
    agentConfig.init();
    agentConfig.bindTcp();
    agentConfig.start();

    // We need a router for our exchange connector to work
    Router router(proxies, "router");
    router.unsafeDisableMonitor();
    router.init();

    // Set a null banker that blindly approves all bids so that we can
    // bid.
    router.setBanker(std::make_shared<NullBanker>(true));

    // Configure exchange connectors
    Json::Value exchangeConfiguration = loadJsonFromFile(exchangeConfigurationFile);
    router.initExchanges(exchangeConfiguration);
    router.initFilters();

    // Start the router up
    router.bindTcp();
    router.start();


    router.forAllExchanges([&](std::shared_ptr<ExchangeConnector> const & item) {
        item->enableUntil(Date::positiveInfinity());
        if(!recordFile.empty()) {
            std::string filename = item->exchangeName() + "-" + recordFile;
            item->startRequestLogging(filename, recordCount);
        }
    });

    RTBKIT::AgentConfig config;
    config.maxInFlight = 20000;
    config.creatives.push_back(RTBKIT::Creative::sampleLB);
    config.creatives.push_back(RTBKIT::Creative::sampleWS);
    config.creatives.push_back(RTBKIT::Creative::sampleBB);

    // create a router proxy and start it.
    // will start polling in a dedicated thread
    BiddingAgentCluster cluster (proxies, "BAC");
    cluster.start ();

    // create/start/configure 1st agent
    auto bob = make_shared<TestAgent>(proxies, "BOB") ;
    bob->onBidRequest = [&] (
                            double timestamp,
                            const Id & id,
                            std::shared_ptr<BidRequest> br,
                            Bids bids,
                            double timeLeftMs,
                            const Json::Value & augmentations,
                            const WinCostModel & wcm)
    {
        Bid& bid = bids[0];
        bid.bid(bid.availableCreatives[0], USD_CPM(1.234));
        bob->doBid(id, bids, Json::Value(), wcm);
        ML::atomic_inc(bob->numBidRequests);
        cerr << bob->serviceName() << "(" << tid() << ") bid count=" << bob->numBidRequests << endl;
    };


    // create/start/configure 2nd agent
    auto alice = make_shared<TestAgent>(proxies, "ALICE") ;
    alice->onBidRequest = [&] (
                              double timestamp,
                              const Id & id,
                              std::shared_ptr<BidRequest> br,
                              Bids bids,
                              double timeLeftMs,
                              const Json::Value & augmentations,
                              const WinCostModel & wcm)
    {
        Bid& bid = bids[0];
        bid.bid(bid.availableCreatives[0], USD_CPM(1.233));
        alice->doBid(id, bids, Json::Value(), wcm);
        ML::atomic_inc(bob->numBidRequests);
        cerr << alice->serviceName() << "(" << tid() << ") bid count=" << alice->numBidRequests << endl;
    };

    // join agents to our cluster
    // joined agents will be added as source to
    // our the cluster's poll loop;
    cluster.join (bob, "BOB");
    cluster.join (alice, "ALICE");


    // for each agent A in cluster, will call A.init()
    cluster.init();

    config.account = {"testCampaign", "bobStrategy"};
    bob->doConfig (config);

    config.account = {"testCampaign", "aliceStrategy"};
    alice->doConfig(config);

    for(;;)
    {
        ML::sleep(10.0);

        std::cerr << "----------[ REPORT ]----------" << std::endl;
        proxies->events->dump(std::cerr);
    }
}


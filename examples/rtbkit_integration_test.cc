/* router_integration_test.cc
   Jeremy Barnes, 21 November 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Overall integration test for the router stack.
*/

#include "rtbkit/core/router/router.h"
#include "rtbkit/core/post_auction/post_auction_loop.h"
#include "rtbkit/core/agent_configuration/agent_configuration_service.h"
#include "rtbkit/core/banker/master_banker.h"
#include "rtbkit/core/banker/slave_banker.h"
#include "jml/utils/rng.h"
#include "rtbkit/core/monitor/monitor.h"
#include "jml/utils/pair_utils.h"
#include "jml/utils/environment.h"
#include "jml/arch/timers.h"
#include "soa/service/testing/redis_temporary_server.h"
#include "testing/generic_exchange_connector.h"
#include "testing/mock_exchange.h"
#include "rtbkit/testing/test_agent.h"
#include <boost/thread.hpp>
#include <netdb.h>
#include <memory>


using namespace std;
using namespace ML;
using namespace Redis;
using namespace Datacratic;
using namespace RTBKIT;


/******************************************************************************/
/* COMPONENTS                                                                 */
/******************************************************************************/

/** Initializes the various components of the RTBKit stack. */
struct Components
{

    std::shared_ptr<ServiceProxies> proxies;

    // See init for an inline description of the various components.

    RedisTemporaryServer redis;
    Router router1, router2;
    PostAuctionLoop postAuctionLoop;
    MasterBanker masterBanker;
    SlaveBudgetController budgetController;
    AgentConfigurationService agentConfig;
    Monitor monitor;
    MonitorProviderProxy monitorProxy;
    TestAgent agent;

    // \todo Add a PAL event subscriber.
    // \todo Add an augmentor.

    vector<unique_ptr<GenericExchangeConnector> > exchangeConnectors;
    vector<int> exchangePorts;


    Components(std::shared_ptr<ServiceProxies> proxies)
        : proxies(proxies),
          router1(proxies, "router1"),
          router2(proxies, "router2"),
          postAuctionLoop(proxies, "pas1"),
          masterBanker(proxies, "masterBanker"),
          agentConfig(proxies, "agentConfigurationService"),
          monitor(proxies, "monitor"),
          monitorProxy(proxies->zmqContext, monitor),
          agent(proxies, "agent1")
    {
    }

    void shutdown()
    {
        postAuctionLoop.shutdown();
        router1.shutdown();
        router2.shutdown();
        masterBanker.shutdown();
        agentConfig.shutdown();
        agent.shutdown();
        monitor.shutdown();
        monitorProxy.shutdown();
        budgetController.shutdown();
        cerr << "done shutdown" << endl;
    }

    void init()
    {
        const string agentUri = "tcp://127.0.0.1:1234";

        // Setup and agent configuration service which is used to notify all
        // interested services of changes to the agent configuration.
        agentConfig.init();
        agentConfig.bindTcp();
        agentConfig.start();

        // Setup a master banker used to keep the canonical budget of the
        // various bidding agent accounts. The data contained in this service is
        // periodically persisted to redis.
        masterBanker.init(std::make_shared<RedisBankerPersistence>(redis));
        masterBanker.bindTcp();
        masterBanker.start();

        // Setup a slave banker that we can use to manipulate and peak at the
        // budgets during the test.
        budgetController.init(proxies->config);
        budgetController.start();

        // Each router contains a slave masterBanker which is periodically
        // synced with the master banker.
        auto makeSlaveBanker = [=] (const std::string & name)
            {
                auto res = std::make_shared<SlaveBanker>
                (proxies->zmqContext, proxies->config, name);
                res->start();
                return res;
            };

        // Setup a post auction loop (PAL) which handles all exchange events
        // that don't need to be processed in real-time (wins, loss, etc).
        postAuctionLoop.init();
        postAuctionLoop.setBanker(makeSlaveBanker("pas1"));
        postAuctionLoop.bindTcp();
        postAuctionLoop.start();

        // Setup two routers which will manage the bid request stream coming
        // from the exchange, the augmentations coming from the augmentors (to
        // be added to the test) and the bids coming from the agents. Along the
        // way it also applies various filters based on agent configuration
        // while ensuring that all the real-time constraints are respected.

        router1.init();
        router1.setBanker(makeSlaveBanker("router1"));
        router1.bindTcp();
        router1.start();

        router2.init();
        router2.setBanker(makeSlaveBanker("router2"));
        router2.bindTcp();
        router2.start();


        // Setup an exchange connector for each router which will act as the
        // middle men between the exchange and the router.

        exchangeConnectors.emplace_back(
                new GenericExchangeConnector(&router1, Json::Value()));

        exchangeConnectors.emplace_back(
                new GenericExchangeConnector(&router2, Json::Value()));

        for (auto& connector : exchangeConnectors) {
            connector->enableUntil(Date::positiveInfinity());

            int port = connector->init(-1, "localhost", 2 /* threads */);
            exchangePorts.push_back(port);
        }

        // Setup a monitor which ensures that any instability in the system will
        // throttle the bid request stream. In other words, it's the component
        // in charge of not making you bankrupt.
        monitor.init();
        monitor.bindTcp();
        monitor.start();

        // The MonitorProxy queries all specified services once per second and
        // feed the Monitor with the aggregate result
        monitorProxy.init(proxies->config,
                          {"router1", "router2", "pas1", "masterBanker"});
        monitorProxy.start();

        // Our bidding agent which listens to the bid request stream from all
        // available routers and decide who gets to see your awesome pictures of
        // kittens.
        agent.start(agentUri, "test-agent");
        agent.configure();


    }
};


/******************************************************************************/
/* SETUP                                                                      */
/******************************************************************************/

/** Transfer a given budget to each router for a given account. */
void allocateBudget(
        SlaveBudgetController& budgetController,
        const AccountKey& account,
        Amount budget)
{
    budgetController.addAccountSync(account);
    budgetController.setBudgetSync(account[0], budget);
    budgetController.topupTransferSync(account, USD(10));

    cerr << budgetController.getAccountSummarySync(account[0], -1) << endl;

    // Syncing is done periodically so we have to wait a bit before the router
    // will have a budget available. Necessary because the bid request stream
    // for this test isn't infinit.
    cerr << "sleeping so that the slave accounts can sync up" << endl;
    ML::sleep(2.1);

    auto summary = budgetController.getAccountSummarySync(account[0], -1);
    cerr << summary << endl;

    ExcAssertEqual(
            summary.subAccounts["testStrategy"].subAccounts["router1"].budget,
            USD(0.10));

    ExcAssertEqual(
            summary.subAccounts["testStrategy"].subAccounts["router2"].budget,
            USD(0.10));
}

/** Some debugging output for the banker. */
void dumpAccounts(
        SlaveBudgetController& budgetController,
        const AccountKey & name, const AccountSummary & a)
{
    cerr << name << ": " << endl;
    cerr << budgetController.getAccountSync(name) << endl;

    for (auto & sub: a.subAccounts) {
        dumpAccounts(budgetController, name.childKey(sub.first), sub.second);
    }
};


/******************************************************************************/
/* MAIN                                                                       */
/******************************************************************************/

/** Small example problem that spins up the various components of the RTBKit
    stack, sends a bunch of bid requests through it then print some stats.

 */
int main(int argc, char ** argv)
{
    // Controls the length of the test.
    enum {
        nExchangeThreads = 10,
        nBidRequestsPerThread = 200
    };

    auto proxies = std::make_shared<ServiceProxies>();

    // If we had a zookeeper instance running, we could use it to do service
    // discovery. Since we don't, ServiceProxies will just default to using a
    // local service map.
    if (false) proxies->useZookeeper("zookeeper.datacratic.com", "stats");

    // If we had a carbon instance running, we could use it to log events. Since
    // we don't, ServiceProxies will just default to using a local equivalent.
    if (false) proxies->logToCarbon("carbon.datacratic.com", "stats");


    // Setups up the various component of the RTBKit stack. See Components::init
    // for more details.
    Components components(proxies);
    components.init();

    // This lambda implements our incredibly sophisticated bidding strategy.
    components.agent.onBidRequest = [&] (
            double timestamp,
            const Id & id,
            std::shared_ptr<BidRequest> br,
            const Json::Value & spots,
            double timeLeftMs,
            const Json::Value & augmentations)
        {
            Json::Value response;
            response[0]["creative"] = spots[0]["creatives"][0];
            response[0]["price"] = 10000;
            response[0]["priority"] = 1;

            Json::Value metadata;
            components.agent.doBid(id, response, metadata);

            ML::atomic_inc(components.agent.numBidRequests);
        };

    // Setup an initial budgeting for the test.
    allocateBudget(
            components.budgetController,
            {"testCampaign", "testStrategy"},
            USD(1000));


    boost::thread_group threads;


    int numDone = 0;
    Date start = Date::now();

    // Uses MockExchange to simulates a very basic exchange for the test.
    auto doExchangeThread = [&] (int threadId)
        {
            string exchangeName = string("exchange-") + to_string(threadId);

            MockExchange exchange(proxies, exchangeName);
            exchange.init(threadId, components.exchangePorts);
            exchange.start(nBidRequestsPerThread);

            ML::atomic_inc(numDone);
        };


    // Start up the exchange thread which should let bid requests flow through
    // our stack.
    for (unsigned i = 0;  i < nExchangeThreads;  ++i)
        threads.create_thread(std::bind(doExchangeThread, i));


    // Dump the budget stats while we wait for the test to finish.
    while (numDone < nExchangeThreads) {
        auto summary = components.budgetController.getAccountSummarySync(
                {"testCampaign"}, -1);
        cerr <<  summary << endl;

        dumpAccounts(components.budgetController, {"testCampaign"}, summary);
        ML::sleep(1.0);
    }


    // Time to start cleaning up.

    Date end = Date::now();

    stringstream eventDump;
    components.proxies->events->dump(eventDump);
    cerr << eventDump.str();

    exit(0);
}


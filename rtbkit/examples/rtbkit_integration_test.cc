/* router_integration_test.cc
   Jeremy Barnes, 21 November 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Overall integration test for the router stack.
*/

#include <memory>

#include "augmentor_ex.h"

#include "rtbkit/core/router/router.h"
#include "rtbkit/core/post_auction/post_auction_service.h"
#include "rtbkit/core/agent_configuration/agent_configuration_service.h"
#include "rtbkit/core/banker/master_banker.h"
#include "rtbkit/core/banker/slave_banker.h"
#include "rtbkit/core/monitor/monitor_endpoint.h"
#include "jml/utils/rng.h"
#include "jml/utils/pair_utils.h"
#include "jml/utils/environment.h"
#include "jml/arch/timers.h"
#include "jml/utils/testing/watchdog.h"
#include "soa/service/testing/redis_temporary_server.h"
#include "rtbkit/testing/generic_exchange_connector.h"
#include "rtbkit/testing/mock_exchange.h"
#include "rtbkit/testing/test_agent.h"
#include "rtbkit/examples/mock_exchange_connector.h"
#include "rtbkit/plugins/adserver/mock_adserver_connector.h"
#include "rtbkit/plugins/adserver/mock_win_source.h"
#include "rtbkit/plugins/adserver/mock_event_source.h"
#include "rtbkit/plugins/bid_request/mock_bid_source.h"
#include <boost/thread.hpp>
#include <netdb.h>
#include <memory>


using namespace std;
using namespace ML;
using namespace Redis;
using namespace Datacratic;
using namespace RTBKIT;


const size_t numAccounts(50); /* number of accounts and agents */

#define ZMQ_APP_LAYER 1

/******************************************************************************/
/* SETUP                                                                      */
/******************************************************************************/

void setupAgent(TestAgent& agent)
{
    // Since we're writting a simple test, this allows us to omit callbacks in
    // the bidding agent class.
    agent.strictMode(false);


    // Indicate to the router that we want our bid requests to be augmented
    // with our frequency cap augmentor example.
    {
        AugmentationConfig augConfig;

        // Name of the requested augmentor.
        augConfig.name = "frequency-cap-ex";

        // If the augmentor was unable to augment our bid request then it
        // should be filtered before it makes it to our agent.
        augConfig.required = true;

        // Config parameter sent used by the augmentor to determine which
        // tag to set.
        augConfig.config = Json::Value(42);

        // Instruct to router to filter out all bid requests who have not
        // been tagged by our frequency cap augmentor.
        augConfig.filters.include.push_back("pass-frequency-cap-ex");

        agent.config.addAugmentation(augConfig);
    }

    // Notify the world about our config change.
    agent.doConfig(agent.config);

    // This lambda implements our incredibly sophisticated bidding strategy.
    agent.onBidRequest = [&] (
            double timestamp,
            const Id & id,
            std::shared_ptr<BidRequest> br,
            Bids bids,
            double timeLeftMs,
            const Json::Value & augmentations,
            const WinCostModel & wcm)
        {
            ExcAssertGreater(bids.size(), 0);

            Bid& bid = bids[0];
            ExcAssertGreater(bid.availableCreatives.size(), 0);

            bid.bid(bid.availableCreatives[0], USD_CPM(1));

            agent.doBid(id, bids, Json::Value(), wcm);
            ML::atomic_inc(agent.numBidRequests);
        };
}


/** Transfer a given budget to each router for a given account. */
void allocateBudget(
        SlaveBudgetController& budgetController,
        const AccountKey& account,
        Amount budget)
{
    budgetController.addAccountSync(account);
    budgetController.setBudgetSync(account[0], budget);
    budgetController.topupTransferSync(account, USD(10));

    // cerr << budgetController.getAccountSummarySync(account[0], -1) << endl;
}

void testBudget(SlaveBudgetController & budgetController,
                const AccountKey & account)
{
    auto summary = budgetController.getAccountSummarySync(account[0], -1);

    if (summary.subAccounts[account[1]].subAccounts["router1"].budget
        != USD(0.10)) {
        cerr << summary << endl;
        throw ML::Exception("USD(0.10) not available in budget of router1");
    }

    if (summary.subAccounts[account[1]].subAccounts["router2"].budget
        != USD(0.10)) {
        cerr << summary << endl;
        throw ML::Exception("USD(0.10) not available in budget of router2");
    }
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
/* COMPONENTS                                                                 */
/******************************************************************************/

/** Initializes the various components of the RTBKit stack. */
struct Components
{

    std::shared_ptr<ServiceProxies> proxies;

    // See init for an inline description of the various components.

    RedisTemporaryServer redis;
    Router router1, router2;
    PostAuctionService postAuctionLoop;
    MasterBanker masterBanker;
    SlaveBudgetController budgetController;
    AgentConfigurationService agentConfig;
    MonitorEndpoint monitor;
    vector<shared_ptr<TestAgent> > agents;
    FrequencyCapAugmentor augmentor1, augmentor2;

    // \todo Add a PAL event subscriber.

    MockAdServerConnector winStream;
    int winStreamPort;
    int eventStreamPort;

    vector<unique_ptr<MockExchangeConnector> > exchangeConnectors;
    vector<int> exchangePorts;


    Components(std::shared_ptr<ServiceProxies> proxies)
        : proxies(proxies),
          router1(proxies, "router1"),
          router2(proxies, "router2"),
          postAuctionLoop(proxies, "pal1"),
          masterBanker(proxies, "masterBanker"),
          agentConfig(proxies, "agentConfigurationService"),
          monitor(proxies, "monitor"),
          augmentor1(proxies, "fca1"),
          augmentor2(proxies, "fca2"),
          winStream("mockStream", proxies)
    {
    }

    void shutdown()
    {
        router1.shutdown();
        router2.shutdown();
        winStream.shutdown();
        postAuctionLoop.shutdown();

        budgetController.shutdown();

        for (shared_ptr<TestAgent> & agent: agents) {
            agent->shutdown();
        }
        augmentor1.shutdown();
        augmentor2.shutdown();
        agentConfig.shutdown();

        monitor.shutdown();

        // Waiting a little bit that SlaveBanker from the Router and PAS stop
        // sending requests to the master banker before shutting it down
        ML::sleep(2);
        masterBanker.shutdown();

        cerr << "done shutdown" << endl;
    }

    void init(size_t numAgents)
    {
        const string agentUri = "tcp://127.0.0.1:1234";

        // Setup a monitor which ensures that any instability in the system will
        // throttle the bid request stream. In other words, it ensures you won't
        // go bankrupt.
        monitor.init({
                    "rtbRequestRouter",
                    "rtbPostAuctionService",
                    "rtbBanker",
                    "rtbDataLogger",
                    "rtbAgentConfiguration"});
        monitor.bindTcp();
        monitor.start();

        // Setup and agent configuration service which is used to notify all
        // interested services of changes to the agent configuration.
        agentConfig.init();
        agentConfig.bindTcp();
        agentConfig.start();

        // Setup a master banker used to keep the canonical budget of the
        // various bidding agent accounts. The data contained in this service is
        // periodically persisted to redis.
        masterBanker.init(std::make_shared<RedisBankerPersistence>(redis));
        auto bankerAddr = masterBanker.bindTcp().second;
        masterBanker.start();

        cerr << "bankerAddr: " + bankerAddr + "\n";
        ML::sleep(5);

        // Setup a slave banker that we can use to manipulate and peak at the
        // budgets during the test.
#if ZMQ_APP_LAYER
        budgetController.setApplicationLayer(make_application_layer<ZmqLayer>(proxies));
#else
        auto appLayer = make_application_layer<HttpLayer>("http://127.0.0.1:15500");
        budgetController.setApplicationLayer(appLayer);
#endif

        budgetController.start();

        // Each router contains a slave masterBanker which is periodically
        // synced with the master banker.
        auto makeSlaveBanker = [=] (const std::string & name) {
            auto res = std::make_shared<SlaveBanker>(name);
#if ZMQ_APP_LAYER
            auto appLayer = make_application_layer<ZmqLayer>(proxies);
#else
            cerr << "bankerAddr: " + bankerAddr + "\n";
            auto appLayer = make_application_layer<HttpLayer>("http://127.0.0.1:15500");
#endif
            res->setApplicationLayer(appLayer);
            res->start();
            return res;
        };

        // Setup a post auction loop (PAL) which handles all exchange events
        // that don't need to be processed in real-time (wins, loss, etc).
        postAuctionLoop.init(8);
        postAuctionLoop.setBanker(makeSlaveBanker("pal1"));
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
                new MockExchangeConnector("mock-1", proxies));

        exchangeConnectors.emplace_back(
                new MockExchangeConnector("mock-2", proxies));

        auto ports = proxies->ports->getRange("mock-exchange");

        for (auto& connector : exchangeConnectors) {
            connector->enableUntil(Date::positiveInfinity());

            int port = connector->init(ports, "localhost", 2 /* threads */);
            exchangePorts.push_back(port);
        }

        router1.addExchange(*exchangeConnectors[0]);
        router1.initFilters();
        router2.addExchange(*exchangeConnectors[1]);
        router2.initFilters();
        
        // Setup an ad server connector that also acts as a midlle men between
        // the exchange's wins and the post auction loop.
        winStream.init(winStreamPort = 12340, eventStreamPort = 12341);
        winStream.start();

        // Our bidding agent which listens to the bid request stream from all
        // available routers and decide who gets to see your awesome pictures of
        // kittens.
        for (size_t i = 0; i < numAgents; i++) {
            AccountKey key{"testCampaign" + to_string(i),
                           "testStrategy" + to_string(i)};
            auto agent = make_shared<TestAgent>(proxies,
                                                "testAgent" + to_string(i),
                                                key);
            agents.push_back(agent);

            agent->init();
            agent->start();
            agent->configure();

            // Some extra customization for our agent to make it extra
            // special. See setupAgent for more details.
            setupAgent(*agent);

            // Setup an initial budgeting for the test.
            allocateBudget(budgetController, key, USD(1000));
        }

        // Our augmentor which does frequency capping for our agent.
        augmentor1.init();
        augmentor1.start();
        augmentor2.init();
        augmentor2.start();
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
    Watchdog watchdog(200.0);

    // Controls the length of the test.
    enum {
        nExchangeThreads = 10,
        nBidRequestsPerThread = 100
    };

    auto proxies = std::make_shared<ServiceProxies>();

    // If we had a zookeeper instance running, we could use it to do service
    // discovery. Since we don't, ServiceProxies will just default to using a
    // local service map.
    if (false) proxies->useZookeeper("zookeeper.rtbkit.org", "stats");

    // If we had a carbon instance running, we could use it to log events. Since
    // we don't, ServiceProxies will just default to using a local equivalent.
    if (false) proxies->logToCarbon("carbon.rtbkit.org", "stats");


    Components components(proxies);

    // Setups up the various component of the RTBKit stack. See
    // Components::init for more details.
    components.init(numAccounts);

    // Syncing is done periodically so we have to wait a bit before the router
    // will have a budget available. Necessary because the bid request stream
    // for this test isn't infinit.
    auto ensureBudgetSync = [&] (const shared_ptr<Banker> & banker) {
        shared_ptr<SlaveBanker> slave = static_pointer_cast<SlaveBanker>(banker);
        while (slave->getNumReauthorized() == 0) {
            ML::sleep(0.5);
        }
    };
    ensureBudgetSync(components.postAuctionLoop.getBanker());
    ensureBudgetSync(components.router1.getBanker());
    ensureBudgetSync(components.router2.getBanker());

    ML::sleep(2.1);

    cerr << "testing budgets\n";
    for (int i = 0; i < numAccounts; i++) {
        AccountKey key{"testCampaign" + to_string(i),
                       "testStrategy" + to_string(i)};
        testBudget(components.budgetController, key);
    }
    cerr << "budgets tested\n";

    // Start up the exchange threads which should let bid requests flow through
    // our stack.
    MockExchange exchange(proxies, "mock-exchange");

    for(auto i = 0; i != nExchangeThreads; ++i) {
        NetworkAddress bids(components.exchangePorts[i % components.exchangePorts.size()]);
        NetworkAddress wins(components.winStreamPort);
        NetworkAddress events(components.eventStreamPort);
        exchange.add(new MockBidSource(bids, nBidRequestsPerThread), new MockWinSource(wins), new MockEventSource(events));
    }

    // Dump the budget stats while we wait for the test to finish. Only the
    // first one is fetched to avoid flooding the console with unreadable
    // data.
    while (!exchange.isDone()) {
        auto summary = components.budgetController.getAccountSummarySync(
            {"testCampaign0"}, -1);
        cerr <<  summary << endl;

        dumpAccounts(components.budgetController, {"testCampaign0"}, summary);
        ML::sleep(1.0);

        auto doCheckBanker = [&] (const string & label,
                                  const shared_ptr<Banker> & banker) {
            shared_ptr<SlaveBanker> slave = static_pointer_cast<SlaveBanker>(banker);
            cerr << ("banker rqs : " + label + " "
                     + to_string(slave->getNumReauthorized())
                     + " last delay: " + to_string(slave->getLastReauthorizeDelay())
                     + "\n");
        };

        doCheckBanker("pal", components.postAuctionLoop.getBanker());
        doCheckBanker("router1", components.router1.getBanker());
        doCheckBanker("router2", components.router2.getBanker());
    }

    cerr << "SHUTDOWN\n";
    _exit(0);
    // Test is done; clean up time.
    components.shutdown();

    components.proxies->events->dump(cerr);
}


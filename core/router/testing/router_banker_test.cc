/* router_banker_test.cc
   Sunil Rottoo, 4th April 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.
   
   Test for the banker class when used in the router.
 */

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "jml/arch/format.h"
#include "jml/arch/exception_handler.h"
#include "jml/utils/guard.h"
#include "rtbkit/plugins/bidding_agent/bidding_agent.h"
#include "rtbkit/core/banker/master_banker.h"
#include "rtbkit/core/banker/slave_banker.h"
#include "rtbkit/common/auction.h"
#include "rtbkit/core/router/router.h"
#include "jml/utils/environment.h"
#include "jml/arch/timers.h"
#include <future>

using namespace std;
using namespace ML;
using namespace Datacratic;
using namespace RTBKIT;

Env_Option<string> tmpDir("TMP", "./tmp");
int numBidRequests;
int numErrors;
int numGotConfig;
int numWins;
int numLosses;
int numNoBudgets;
int numTooLates;
int numAuctions;

class RouterTester;

struct TestAgent: public BiddingAgent {
    TestAgent(RouterTester &routerTester);

    // we only want to send requests only the router is configured
    future<bool> getConfigFuture()
    {
        return configPromise_.get_future();
    }

    future<bool> getCompleteFuture()
    {
        return completePromise_.get_future();
    }

    void setDefaultConfig()
    {
        AgentConfig config;
        config.campaign = "TestCampaign";
        config.strategy = "strategy1";
        config.maxInFlight = 20000;
        config.creatives.push_back(Creative::sampleLB);
        config.creatives.push_back(Creative::sampleWS);
        config.creatives.push_back(Creative::sampleBB);
        this->config = config;
    }

    void defaultError(double timestamp, const std::string & error,
            const std::vector<std::string> & message)
    {
         cerr << "agent got error: " << error << " from message: "
              << message << endl;
    }

    void defaultNeedConfig(double)
    {
        cerr << "need config" << endl;
    }

    void defaultGotConfig(double)
    {
        configPromise_.set_value(true);
        haveGotConfig = true;
    }

    void defaultAckHeartbeat(double)
    {
        cerr << "ack heartbeat" << endl;
        ++numHeartbeats;
    }

    void finishBid(const BiddingAgent::BidResultArgs & args);

    void step1Check();
    void step2Check();
    void step3Check();

    void defaultWin(const BiddingAgent::BidResultArgs & args)
    {
        Guard guard(lock);
        numWins++;
        //cerr << "Got a default win for bid " << args.auctionId << " num wins " << numWins <<endl;
        finishBid(args);
    }

    void injectWins();

    void defaultLoss(const BiddingAgent::BidResultArgs & args)
    {
        cerr << "default loss" << endl;
    }

    void defaultNoBudget(const BiddingAgent::BidResultArgs & args);
    void defaultBid(double timestamp, const Id & id,
            std::shared_ptr<BidRequest> br, const Json::Value & imp,
            double timeLeftMs);

    void setupCallbacks()
    {
        onError
            = boost::bind(&TestAgent::defaultError, this, _1, _2, _3);
        onNeedConfig
            = boost::bind(&TestAgent::defaultNeedConfig, this, _1);
        onGotConfig
            = boost::bind(&TestAgent::defaultGotConfig, this, _1);
        onAckHeartbeat
            = boost::bind(&TestAgent::defaultAckHeartbeat, this, _1);
         onWin
            = boost::bind(&TestAgent::defaultWin, this, _1);
        onLoss
            = boost::bind(&TestAgent::defaultLoss, this, _1);
        onNoBudget
            = boost::bind(&TestAgent::defaultNoBudget, this, _1);
        onBidRequest
             = boost::bind(&TestAgent::defaultBid, this, _1, _2, _3, _4, _5);
    }

    void configure()
    {
        doConfig(config.toJson());
    }

    void doBid(const Id & id, const Json::Value & response,
            const Json::Value & metadata)
    {
        if (response.size() != 0)
        {
            recordBid(id);
        }
        BiddingAgent::doBid(id, response, metadata);
    }

    void recordBid(const Id & id)
    {
        Guard guard(lock);
        if (!awaitingStatus.insert(id).second)
            throw ML::Exception("auction already in progress");

        numBidsOutstanding = awaitingStatus.size();
    }

    AgentConfig config;
    promise<bool> configPromise_;
    promise<bool> completePromise_;
    RouterTester &tester_;bool haveGotConfig;
    int numHeartbeats;
    int numBidRequests;
    int numErrors;
    int numGotConfig;
    int numWins;
    int numLosses;
    int numNoBudgets;
    int numTooLates;
    int numAuctions;
    int expectedWins_, expectedNoBudgets_;
    int totalRequests_;
    uint64_t bidPrice_;
    uint64_t winPrice_;

    unsigned int step_; // What is the step of the test plan
    typedef ML::Spinlock Lock;
    typedef boost::lock_guard<Lock> Guard;
    mutable Lock lock;

    std::set<Id> awaitingStatus;
    int numBidsOutstanding;

};
struct RouterTester {

    RouterTester(const std::string & agentUri, const std::string & agentName)
        : agentUri(agentUri), agentName(agentName),
          router(std::make_shared<ServiceProxies>(), "router"),
          agent(*this),campaign_("TestCampaign"),
          strategy_("strategy1"),numAuctionsDone_(0)
    {
        router.bindAgents(agentUri);
    }

    ~RouterTester()
    {
        shutdown();
    }

    void start()
    {
        agent.start(agentUri, agentName);
        router.start();
        agent.configure();
        future<bool> configFuture = agent.getConfigFuture();
        // Wait till the agent has registered - otherwise injection of bids will fail
        // router.enterSimulationMode();

        configFuture.get();
        // Set the budget for the test campaign of $10
        setBudget(campaign_, 10000000);
        // transfer $1 to strategy1
        // Since this is all asynchronous we want to make sure
        bool budgetSet = false;
        Json::Value result;
        unsigned int numTries = 0;
        while (!budgetSet && numTries < 10)
        {
            result = router.topupTransferSync({campaign_, strategy_}, 1000000);
            Json::Value strat = result["strategy"];
            if (strat["available"]["micro-USD"] == 1000000)
            {
                cerr << "Amount has been transferred to strategy" << endl;
                budgetSet = true;
                break;
            }
            ML::sleep(0.1);
            numTries++;
        }
        if (!budgetSet)
            throw ML::Exception(
                    "Failed to transfer to strategy after 10 tries");
    }

    Json::Value setBudget(const std::string &campaign, uint64_t amountInMicros,
            unsigned int totalTries = 10)
    {
        bool budgetSet = false;
        Json::Value result;
        unsigned int numTries = 0;
        while (!budgetSet && numTries < totalTries)
        {
            // Since this is all being done asynchronous
            result = router.setBudget(campaign, amountInMicros);
            //cerr << "result = " << result << endl;
            if (result["available"]["micro-USD"] == 10000000)
            {
                cerr << "budget is set to correct value " << endl;
                budgetSet = true;
                break;
            }
            ML::sleep(0.1);
            numTries++;
        }
        if (!budgetSet)
        {
            throw ML::Exception(
                    "Failed to set budget after "
                            + boost::lexical_cast<string>(totalTries)
                            + " tries");
        }
        return result;
    }

    void sleepUntilIdle()
    {
        router.sleepUntilIdle();
    }

    void shutdown()
    {
        router.shutdown();
        //sleepUntilIdle();
        agent.shutdown();
    }

    std::shared_ptr<Auction> createAuction(unsigned int id)
    {
        auto handleAuction = [&] (std::shared_ptr<Auction> auction)
        {
            ML::atomic_inc(numAuctionsDone_);
        };

        Date start = Date::now();
        Date expiry = start.plusSeconds(0.05);
        std::shared_ptr<BidRequest> request(new BidRequest());
        request->auctionId = Id(id);
        AdSpot spot1(request->auctionId);
        spot1.formats.push_back(Format(300, 250));
        request->imp.push_back(spot1);
        string current = request->toJsonStr();
        std::shared_ptr<Auction> auction(
                new Auction(handleAuction, request, current, start, expiry,
                        id));
        return auction;
    }

    // This method will submit 11 auctions for which we expect 10 to pass and 1 to fail with
    // no budget.
    void runStep(unsigned int step, unsigned expectedWins,
            unsigned expectedNoBudgets, uint64_t bidPrice, uint64_t winPrice)
    {
    	cerr << "Running step " << step << endl;
    	agent.bidPrice_ = bidPrice;
    	agent.winPrice_ = winPrice;
    	agent.expectedWins_ += expectedWins;
    	agent.expectedNoBudgets_ += expectedNoBudgets;
    	agent.totalRequests_ += expectedWins + expectedNoBudgets ;
    	cerr << "Total Requests: " << agent.totalRequests_ << endl;
    	agent.step_ = step ;
        unsigned auctionStart = numAuctionsDone_;
        unsigned auctionEnd = auctionStart + expectedWins + expectedNoBudgets;
        cerr << "AuctionStart: " << auctionStart << " auctionEnd:" << auctionEnd
                << endl;
        //-----------------------------------------------------------------------------------
        // we run this part in a separate thread because this function is being
        // called as part of a callback and we do not want to block the sender which we will
        // because of the numerous calls to sleep.
        // Please note that we capture by copy since this is run as a detached thread and
        // references to auctionStart and auctionEnd would cause problems
        //------------------------------------------------------------------------------------
        std::thread([this,auctionStart,auctionEnd]()
        {
            //cerr << "inject auction from thread " << std::this_thread::get_id() << endl;
                cerr << "AuctionStart: " << auctionStart << " auctionEnd:" << auctionEnd << endl;
                for (unsigned i = auctionStart; i < auctionEnd; ++i)
                {
                    ML::sleep(0.1);
                    Date start = Date::now();
                    this->router.injectAuction(this->createAuction(i+1),start.secondsSinceEpoch() + 2.0);
                }
            }).detach();
    }

    string agentUri;
    string agentName;
    Router router;
    TestAgent agent;
    string campaign_, strategy_;
    uint64_t numAuctionsDone_;
};

TestAgent::TestAgent(RouterTester &routerTester)
    :  BiddingAgent(routerTester.router.getZmqContext()),tester_(routerTester),	numBidRequests(0), numErrors(0), numGotConfig(0),numWins(0),
      numLosses(0), numNoBudgets(0), numTooLates(0), numAuctions(0),expectedWins_(0),
      expectedNoBudgets_(0),totalRequests_(0)
{
    setDefaultConfig();
    setupCallbacks();
}

void TestAgent::step1Check()
{
    // Now check that value that the banker should have
    const Campaigns &theCampaigns
        = dynamic_cast<RedisBanker &>(*tester_.router.getBanker())
        .getCampaigns();
    Campaigns::const_iterator cIt = theCampaigns.find("TestCampaign");
    BOOST_CHECK(cIt != theCampaigns.end());
    // Check that we have $9 available
    BOOST_CHECK_EQUAL(cIt->second.available_, USD(9));
    // Get the first strategy
    Strategies::const_iterator sIt1 = cIt->second.strategies_.find("strategy1");
    BOOST_CHECK(sIt1 != cIt->second.strategies_.end());
    BOOST_CHECK_EQUAL(sIt1->second.committed_, MicroUSD(0));
    BOOST_CHECK_EQUAL(sIt1->second.available_, MicroUSD(500000));
    BOOST_CHECK_EQUAL(sIt1->second.spent_, MicroUSD(500000));
}

void TestAgent::step2Check()
{
    // Now check that value that the banker should have
    const Campaigns &theCampaigns
        = dynamic_cast<RedisBanker &>(*tester_.router.getBanker())
        .getCampaigns();
    Campaigns::const_iterator cIt = theCampaigns.find("TestCampaign");
    BOOST_CHECK(cIt != theCampaigns.end());
    // Check that we have $9 available
    BOOST_CHECK_EQUAL(cIt->second.available_, USD(9));
    // Get the first strategy
    Strategies::const_iterator sIt1 = cIt->second.strategies_.find("strategy1");
    BOOST_CHECK(sIt1 != cIt->second.strategies_.end());
    BOOST_CHECK_EQUAL(sIt1->second.committed_, MicroUSD(0));
    BOOST_CHECK_EQUAL(sIt1->second.available_, MicroUSD(250000));
    BOOST_CHECK_EQUAL(sIt1->second.spent_,     MicroUSD(750000));
}

void TestAgent::defaultNoBudget(const BiddingAgent::BidResultArgs & args)
{
    Guard guard(lock);
    numNoBudgets++;
    // Those that have no budgets we do not want to inject wins for
    awaitingStatus.erase(args.auctionId);
    if (step_ == 1)
    {
        injectWins();
    } else if (step_ == 2 && numNoBudgets == 6)
    {
        //cerr << "(step 2):we will inject wins for :" << endl;
        //copy(awaitingStatus.begin(), awaitingStatus.end(),ostream_iterator<Id>(cerr,","));
        //cerr << endl;
        injectWins();
    } else if (step_ == 3 && numNoBudgets == 7)
    {
        //cerr << "(step 2):we will inject wins for :" << endl;
        //copy(awaitingStatus.begin(), awaitingStatus.end(),ostream_iterator<Id>(cerr,","));
        //cerr << endl;
        injectWins();
    }
}

void TestAgent::injectWins()
{
    for (auto it = awaitingStatus.begin(); it != awaitingStatus.end(); ++it)
    {
        UserIds userIds;
        Date now = Date::now();
        ML::sleep(0.2);
        //cerr << " We have received all bid requests. We can now schedule wins for all of them " << endl;
        //cerr << "Injecting win " << *it << " with spot it " << *it << endl;
        tester_.router.injectWin(Id(*it), Id(*it), MicroUSD(winPrice_),
                                 now.plusSeconds(0.05),
                                 Json::Value(), userIds,
                                 tester_.campaign_, tester_.strategy_,
                                 now.plusSeconds(0.15));

    }
    awaitingStatus.clear();
}

void TestAgent::step3Check()
{
    // Now check that value that the banker should have
    const Campaigns &theCampaigns
        = dynamic_cast<RedisBanker &>(*tester_.router.getBanker())
        .getCampaigns();
    Campaigns::const_iterator cIt = theCampaigns.find("TestCampaign");
    BOOST_CHECK(cIt != theCampaigns.end());
    // Check that we have $9 available
    BOOST_CHECK_EQUAL(cIt->second.available_, USD(9));
    // Get the first strategy
    Strategies::const_iterator sIt1 = cIt->second.strategies_.find("strategy1");
    BOOST_CHECK(sIt1 != cIt->second.strategies_.end());
    BOOST_CHECK_EQUAL(sIt1->second.committed_, USD(0));
    BOOST_CHECK_EQUAL(sIt1->second.available_, USD(0));
    BOOST_CHECK_EQUAL(sIt1->second.spent_, MicroUSD(1000000));
}

void TestAgent::defaultBid(double timestamp, const Id & id,
        std::shared_ptr<BidRequest> br, const Json::Value & imp,
        double timeLeftMs)
{
    Json::Value response;
    //cerr << "imp = " << imp << endl;
    response[0u]["creative"] = imp[0u]["creatives"][0u];
    response[0u]["price"] = bidPrice_; //100000000;
    response[0u]["surplus"] = 1;

    Json::Value metadata;
    doBid(id, response, metadata);
    ML::atomic_inc(numBidRequests);
}

void TestAgent::finishBid(const BiddingAgent::BidResultArgs & args)
{
    // cerr << "finishBid for bid " << args.auctionId << endl;
    // If we have accounted for all requests we can signal
    if (numWins == expectedWins_ && numNoBudgets == expectedNoBudgets_)
    {
        if (step_ == 1)
        {
            step1Check();
            // Now run another series trying to run down the budget. We expect 5 wins and 5 no budgets
            tester_.runStep(2, 5, 5, bidPrice_, winPrice_);
        } else if (step_ == 2)
        {
            cerr << "Step 2 was finished " << endl;
            step2Check();
            // At the start of step 3 we have 250 000 micro $available we want to run this down
            // by submitting 100 bid requests with a bid price of 2500 micro$ and win price of 2500
            // Please note that we submit an additional request to test that we cannot increase the budget
            bidPrice_ = 2500000;
            winPrice_ = 2500;
            tester_.runStep(3, 100, 1, bidPrice_, winPrice_);
        } else
        {
            cerr << "Step 3 was finished " << endl;
            step3Check();
            // Now try to set the budget to something that is less than the amount spent
            // we expect this to fail after synchronously since this checked in the
            // calling thread. Therefore we should only need to do this once
            BOOST_CHECK_THROW(tester_.setBudget(tester_.campaign_, 900000, 1),
                    ML::Exception);
            completePromise_.set_value(true);
        }
    }
}

BOOST_AUTO_TEST_CASE( test_banker_via_router )
{
    string campaignPrefix = "bankerTest";

    /**
     * In this test we allocate $1 to a strategy and inject 10 bids at 100 000 micro$ each
     * We subsequently spend 50 000 micro $ on each. We expect therefore that at the
     * end of that we will have $0.50 available in the strategy and $0.5 spent
     */
    ML::set_default_trace_exceptions(false);
    std::map<Id, uint64_t> winNotifications;
    ML::Spinlock lock;

    RouterTester tester("inproc://rrat", "test");
    TestAgent & agent = tester.agent;

    std::shared_ptr<SlaveBanker> banker(new SlaveBanker(campaignPrefix, "banker", std::make_shared<ServiceProxies>(), redis));
    std::shared_ptr<SlaveBudgetController> budgetController
        (new SlaveBudgetController(campaignPrefix, redis));
    tester.router.setBanker(banker);
    tester.router.setBudgetController(budgetController);

    tester.start();
    unsigned expectedWins = 10;
    unsigned expectedNoBudgets = 1;
    uint64_t bidPrice = 100000000; // cpm in micro$ - this will be divided by 1000 by the router
    uint64_t winPrice = 50000; // micro $ as is
    tester.runStep(1, expectedWins, expectedNoBudgets, bidPrice, winPrice);
    // Now wait for all requests to arrive before exiting
    future<bool> completeFuture = agent.getCompleteFuture();
    completeFuture.get();
    cerr
            << "All pending requests have been processed ....shutting down(banker_test)"
            << endl;
    tester.shutdown();
}


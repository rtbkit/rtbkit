/* redis_banker_test.cc
   Sunil Rottoo, 4th April 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.
   
   Test for the banker class.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "jml/arch/format.h"
#include "jml/arch/exception_handler.h"
#include "jml/utils/guard.h"
#include "rtbkit/plugins/bidding_agent/bidding_agent.h"
#include "rtbkit/core/banker/redis_banker.h"
#include "rtbkit/common/auction.h"
#include "rtbkit/core/router/router.h"
#include "jml/utils/environment.h"
#include "jml/arch/timers.h"
#include "dataflow/work_scheduler.h"
#include <future>
#include "soa/service/testing/redis_temporary_server.h"
#include <boost/thread/thread.hpp>


using namespace std;
using namespace ML;
using namespace Datacratic;
using namespace RTBKIT;
using namespace Redis;

const StrategyInfo & getStrategyInfo(const RedisBanker &theBanker,
        const std::string &campaign, const std::string &strategy)
{
    const Campaigns &theCampaigns = theBanker.getCampaigns();
    Campaigns::const_iterator found = theCampaigns.find(campaign);
    BOOST_CHECK(found != theCampaigns.end());

    const Strategies &memStrategies =
            theCampaigns.find(campaign)->second.strategies_;
    const StrategyInfo &stratinfo = memStrategies.find(strategy)->second;
    return stratinfo;
}

BOOST_AUTO_TEST_CASE( test_banker_new_campaign )
{
    auto s = std::make_shared<ServiceProxies>();
    RedisTemporaryServer redis;
    RedisBanker banker("bankerTest", "b", s, redis);
    RedisBudgetController controller("bankerTest", redis);

    auto campaign = banker.getCampaignStatusJson("hello", "");
    cerr << campaign << endl;

    auto strategy = banker.getCampaignStatusJson("hello", "world");
    cerr << strategy << endl;
    
    BOOST_CHECK_EQUAL(banker.authorizeBid("hello", "world", "one", MicroUSD(1)),
                      false);
    
    controller.topupTransferSync("hello", "world", MicroUSD(10));
}

BOOST_AUTO_TEST_CASE( test_transfer_all_remaining )
{
    // Test for ADGR-188
    // When we try to transfer more to a strategy than is available, we
    // should not simply fail, but transfer all that we can

    auto s = std::make_shared<ServiceProxies>();
    RedisTemporaryServer redis;
    RedisBudgetController controller("bankerTest", redis);
    controller.addCampaignSync("testCampaign");
    controller.addStrategySync("testCampaign", "testStrategy");
    controller.setBudgetSync("testCampaign", MicroUSD(1000));
    controller.topupTransferSync("testCampaign", "testStrategy",
                                           MicroUSD(10000));

    RedisBanker banker("bankerTest", "b", s, redis);

    banker.sync();
    Json::Value status = banker.getCampaignStatusJson("testCampaign", "");

    cerr << status << endl;

    BOOST_CHECK_EQUAL(status["available"]["micro-USD"].asInt(), 0);
    BOOST_CHECK_EQUAL(status["strategies"][0]["available"]["micro-USD"].asInt(), 1000);
    BOOST_CHECK_EQUAL(status["strategies"][0]["transferred"]["micro-USD"].asInt(), 1000);
}

#if 1
BOOST_AUTO_TEST_CASE( test_banker_basic )
{
    {
        std::string campaignName = "TestCampaign";
        std::string campaign2 = "TestCampaign2";
        std::string campaign3 = "TestCampaign3";
        std::string campaign4 = "TestCampaign4";
        std::string campaign5 = "TestCampaign5";
        std::string campaign6 = "TestCampaign6";
        std::string campaign7 = "TestCampaign7";
        std::string strategy1 = "strategy1";
        std::string strategy2 = "strategy2";

        string campaignPrefix = "bankerTest";

        RedisTemporaryServer redis;

        Redis::AsyncConnection dbConn(redis);

        auto REDIS_CHECK_CAMPAIGN_EQUAL = [&](const string &key, const string &hash,
                uint64_t value)
        {
            Reply reply = dbConn.exec(HGET(campaignPrefix + ":" + key, key)).reply();
            if (reply.type() != ARRAY)
            {
                return false;
            }
            if (reply.length() == 0)
            {
                return false;
            }
            cerr << "reply is " << reply << endl;
            //uint64_t readValue = boost::lexical_cast<uint64_t>(reply[])
            return true;
        };
        
        unsigned testNum = 1;

        auto s = std::make_shared<ServiceProxies>();
        {
            // add two campaigns - one with no strategies and one with 2 strategies
            // and make sure they load property
            auto runCommand = [&] (Redis::Command command)
                {
                    Reply reply1 = dbConn.exec(command).reply();
                    //cerr << "command = " << command << endl;
                    //cerr << "reply1.type() = " << reply1.type() << endl;
                    //cerr << "reply1.asJson() = " << reply1.asJson() << endl;
                    //cerr << "reply1 = " << reply1 << endl;
                    BOOST_CHECK_EQUAL(reply1.asString(), "OK");
                };

            auto setCampaign = [&] (std::string campaign,
                                    uint64_t available)
                {
                    runCommand(HMSET(campaignPrefix + ":" + campaign,
                                     "available:micro-USD", available));
                };
            
            auto setStrategy = [&] (std::string campaign, std::string strategy,
                                    uint64_t available,
                                    uint64_t spent,
                                    uint64_t transferred)
                {
                    runCommand(HMSET(campaignPrefix + ":" + campaign
                                     + ":" + strategy,
                                     "available:micro-USD", available,
                                     "spent:micro-USD", spent,
                                     "transferred:micro-USD", transferred));
                };

            setCampaign("TestCampaign6", 1000000);

            setCampaign("TestCampaign7", 2000000);
            setStrategy("TestCampaign7", "strategy1", 100000, 100000, 200000);
            setStrategy("TestCampaign7", "strategy2",      0, 200000, 200000);

            setCampaign("TestCampaign4", 1000000);
            setStrategy("TestCampaign4", "strategy1",  90000,  10000, 100000);
        }

        // Check that if we supply the wrong redis port we fail to connect
        BOOST_CHECK_THROW(RedisBanker("xxx", "b", s,
                                      Redis::Address::tcp("dev2", 6377)),
                          ML::Exception);
        // Check that if we supply the wrong redis hostname we fail to connect
        BOOST_CHECK_THROW(RedisBanker("xxx", "b", s,
                                      Redis::Address::tcp("fred", 6379)),
                          ML::Exception);
        
        // Now create a valid banker
        cerr <<"Creating banker and loading all campaigns" << endl;

        RedisBanker theBanker("bankerTest", "b", s, redis);
        RedisBudgetController theController("bankerTest", redis);
        cerr << "done creating banker" << endl;
        {
            Json::Value result = theBanker.dumpAllCampaignsJson();

            BOOST_CHECK_EQUAL(result.isMember(campaign6), true);

            cerr << "All Campaigns: " << result << endl;
            cerr << "\t" << testNum++ << ". Checking that campaign 6 was loaded correctly."
                 << endl;
            theController.addCampaignSync(campaign6);

            // Make sure that we have the correct values
            CampaignInfo campaign = theBanker.getCampaignDebug(campaign6);
            BOOST_CHECK_EQUAL(campaign.available_,MicroUSD(1000000));
            REDIS_CHECK_CAMPAIGN_EQUAL(campaign6, "available_", 1000000);
            BOOST_CHECK(campaign.transferred_.empty());
            // Make sure that there are no strategies
            const Strategies &strats = campaign.strategies_;
            BOOST_CHECK_EQUAL(strats.size(), 0);
        }
        {
            cerr << "\t" << testNum++ << ". Checking that campaign7 was loaded correctly."
                 << endl;
            theController.addCampaignSync(campaign7);
            CampaignInfo campaign = theBanker.getCampaignDebug(campaign7);

            // Make sure that we have the correct values
            BOOST_CHECK_EQUAL(campaign.available_,MicroUSD(2000000));
            BOOST_CHECK_EQUAL(campaign.transferred_,MicroUSD(400000));
            // Make sure that both strategies are in there
            const Strategies &strats = campaign.strategies_;
            BOOST_CHECK_EQUAL(strats.size(), 2);
            Strategies::const_iterator found1 = strats.find(strategy1);
            BOOST_CHECK(found1 != strats.end());
            if(found1 != strats.end())
            {
                BOOST_CHECK_EQUAL(found1->second.available_, MicroUSD(100000));
                BOOST_CHECK_EQUAL(found1->second.spent_, MicroUSD(100000));
                BOOST_CHECK_EQUAL(found1->second.transferred_, MicroUSD(200000));
            }
            Strategies::const_iterator found2 = strats.find(strategy2);
            BOOST_CHECK(found2 != strats.end());
            if(found2 != strats.end())
            {
                BOOST_CHECK_EQUAL(found2->second.available_, MicroUSD(0));
                BOOST_CHECK_EQUAL(found2->second.spent_, MicroUSD(200000));
                BOOST_CHECK_EQUAL(found2->second.transferred_, MicroUSD(200000));
            }
            BOOST_CHECK_EQUAL(campaign.spent_,
                              found1->second.spent_ + found2->second.spent_);
        }
        // Check that we do not accept a campaign that is an empty string
        auto addCampaign1 = [&]()
        {
            theController.addCampaignSync("");
        };
        BOOST_CHECK_THROW(addCampaign1(), BankerException);

        // Now add a valid campaign
        {
            theController.addCampaignSync(campaignName);
            CampaignInfo campaign = theBanker.getCampaignDebug(campaignName);
            cerr << "\t" << testNum++ << ". Added TestCampaign got the result: "
                    << campaign << endl;
            BOOST_CHECK(campaign.available_.empty());
        }
        // Now add a budget to an existing campaign
        {
            // create a $500 campaign
            theController.setBudgetSync(campaignName, USD(500));
            theBanker.sync();
            CampaignInfo campaign = theBanker.getCampaignDebug(campaignName);
            cerr << "\t" << testNum++ << ". Set the budget for : " << campaign
                    << endl;
            BOOST_CHECK_EQUAL(campaign.available_, USD(500));
        }
        // Now add a strategy 1 to a campaign
        {
            theController.addStrategySync(campaignName, strategy1);
            CampaignInfo campaign = theBanker.getCampaignDebug(campaignName);
            cerr << "\t" << testNum++ << ". Added strategy : " << strategy1
                    << " to campaign " << campaign << endl;
            BOOST_CHECK_EQUAL(campaign.available_, USD(500));
            // The strategy isn't there yet as we haven't yet transferred
            // any money into it.
        }
        // Next add a strategy to a campaign that is not known.  This should
        // not fail, but we don't expect to see it there until we transfer
        // some budget to it.
        {
            theController.addStrategySync(campaign2, strategy1);
        }
        // Set the budget for a non-existing campaign we expect the campaign
        // to be added
        {
            // create a $600 campaign
            theController.setBudgetSync(campaign3, USD(600));
            theBanker.sync();
            CampaignInfo campaign = theBanker.getCampaignDebug(campaign3);
            cerr << "\t" << testNum++
                    << ". Set the budget for previously non-existing campaign: "
                    << campaign3 << endl;
            BOOST_CHECK_EQUAL(campaign.available_, USD(600));
        }
        // Add another strategy to campaign1
        {
            theController.addStrategySync(campaignName, strategy2);
            CampaignInfo campaign = theBanker.getCampaignDebug(campaignName);
            cerr << "\t" << testNum++ << ". Added strategy : " << strategy2
                    << " to campaign " << campaign << endl;
            BOOST_CHECK_EQUAL(campaign.available_, USD(500));
        }
        {
            // Top up strategy 1 to $20
            {
                std::promise<CampaignInfo> thePromise;
                auto onTopupDone = [&] (std::exception_ptr exc)
                    {
                        cerr << "called onTopupDone" << endl;
                        if (exc)
                            thePromise.set_exception(exc);
                        else {
                            cerr << "getting debug campaign" << endl;
                            theBanker.sync();
                            thePromise.set_value(theBanker.getCampaignDebug(campaignName));
                        }
                        cerr << "finished onTopupDone" << endl;
                    };

                cerr << "performing topup to strategy" << endl;
                theController.topupTransfer(campaignName, strategy1,
                                                      MicroUSD(20000000),
                                                      onTopupDone);

                CampaignInfo campaign = thePromise.get_future().get();
                cerr << "\t " << testNum++ << ". topped up strategy : " << strategy1
                        << " to campaign " << campaign << endl;
                BOOST_CHECK_EQUAL(campaign.transferred_, USD(20));
                // Make sure that the campaign now has $480
                BOOST_CHECK_EQUAL(campaign.available_, USD(480));
                BOOST_CHECK_GE(campaign.strategies_.size(), 1);
                const Strategies &strategies = campaign.strategies_;
                // Make sure the strategies are there in-memory
                Strategies::const_iterator foundStrat1 = strategies.find(strategy1);
                BOOST_CHECK(foundStrat1 != strategies.end());

                // Make sure that the strategy 1 has $20 available
                BOOST_CHECK_EQUAL(foundStrat1->second.available_, USD(20));
                //..and 20$ transferred
                BOOST_CHECK_EQUAL(foundStrat1->second.transferred_, USD(20));
            }
            // topup strategy2 to $20
            {
                std::promise<CampaignInfo> thePromise;
                auto onTopupDone = [&] (std::exception_ptr exc)
                    {
                        cerr << "onTopupDone callback with exc " << (bool)exc
                        << endl;
                        if (exc)
                            thePromise.set_exception(exc);
                        else {
                            theBanker.sync();
                            thePromise.set_value(theBanker.getCampaignDebug(campaignName));
                        }
                        cerr << "onTopupDone callback finished" << endl;
                    };

                theController.topupTransfer(campaignName, strategy2,
                                                      USD(20), onTopupDone);

                CampaignInfo campaign = thePromise.get_future().get();

                 cerr << "\t " << testNum++ << ". Added strategy : " << strategy2
                         << " to campaign " << campaign << endl;
                 BOOST_CHECK_EQUAL(campaign.transferred_, USD(40));
                 // Make sure that the campaign now has $480
                 BOOST_CHECK_EQUAL(campaign.available_, USD(460));
                 BOOST_CHECK_EQUAL(campaign.strategies_.size(), 2);
                 const Strategies &strategies = campaign.strategies_;
                 // Make sure the strategies are there in-memory
                 Strategies::const_iterator foundStrat1 = strategies.find(strategy1);
                 BOOST_CHECK(foundStrat1 != strategies.end());
                 Strategies::const_iterator foundStrat2 = strategies.find(strategy2);
                 BOOST_CHECK(foundStrat2 != strategies.end());
                 // Make sure that the strategy 2 has $20 available
                 BOOST_CHECK_EQUAL(foundStrat2->second.available_, USD(20));
                 //..and 20$ transferred
                 BOOST_CHECK_EQUAL(foundStrat2->second.transferred_, USD(20));

            }
        }
        cerr << "\t" << testNum++
                << ". Trying to add 100 000 to campaign. Exceeds maximum test"
                << endl;
        // Now try to add 50000$ to the campaign. It should throw an exception since this is
        // the current limit

        BOOST_CHECK_THROW(theController.addBudgetSync(campaignName,USD(100000)),
                          BankerException);
        cerr << "finished adding too much" << endl;

        {
            // Top up strategy 1 to $20
            // Now add $30 to the campaign this should be fine and we should
            // have $510 available
            cerr << "\t" << testNum++
                    << ". Add $30 to the budget for campaign: " << campaignName
                    << endl;
            theController.addBudgetSync(campaignName, USD(30));
            theBanker.sync();
            CampaignInfo campaign = theBanker.getCampaignDebug(campaignName);
            BOOST_CHECK_EQUAL(campaign.available_, USD(490));
            cerr << "The campaign is " << campaign << endl;
        }
        // Now we want a test where we set the budget to an absolute value.
        // We first try to set the budget to a value that is less than the
        // amount transferred. We should get an exception
        cerr << "\t" << testNum++
                << ". Testing that we cannot set the campaign budget to less than transferred amount"
                << endl;
        BOOST_CHECK_THROW(theController.setBudgetSync(campaignName, MicroUSD(15000000)),
                          BankerException);
        {
            // Now we set the budget to $300 which should be okay. We expect the
            // available amount to now be $280 since $20 has already been transferred
            cerr << "\t" << testNum++ << ". Set the budget to $300: "
                    << campaignName << endl;
            theController.setBudgetSync(campaignName, USD(300));
            theBanker.sync();
            CampaignInfo campaign = theBanker.getCampaignDebug(campaignName);
            BOOST_CHECK_EQUAL(campaign.available_, USD(260));
            BOOST_CHECK_EQUAL(campaign.transferred_, USD(40));
            const Strategies &memStrategies = campaign.strategies_;
            const StrategyInfo &stratinfo =
                    memStrategies.find(strategy1)->second;
            BOOST_CHECK_EQUAL(stratinfo.commitments_.size(), 0);
        }
        {
            // Test - Simple Bid
            // Now bid on something for a value of $19 for strategy 1
            {
                cerr << "\t" << testNum++ << ". Bid on something for $19: "
                        << campaignName << endl;


                BOOST_CHECK(theBanker.authorizeBid(campaignName, strategy1, "bid1", MicroUSD(19000000)));
                theBanker.sync();
                CampaignInfo campaign = theBanker.getCampaignDebug(campaignName);

                const Strategies &memStrategies = campaign.strategies_;
                const StrategyInfo &stratinfo =
                        memStrategies.find(strategy1)->second;
                BOOST_CHECK_EQUAL(stratinfo.available_, MicroUSD(1000000));
                BOOST_CHECK_EQUAL(stratinfo.committed_, MicroUSD(19000000));
                BOOST_CHECK_EQUAL(stratinfo.commitments_.size(), 1);
            }
            {
                // Bid something for a value of $15 for strategy2
                cerr << "\t" << testNum++ << ". Bid on something for $15: "
                        << campaignName << " strategy : " << strategy2 << endl;
                theBanker.authorizeBid(campaignName, strategy2, "bid1", MicroUSD(15000000));
                theBanker.sync();
                CampaignInfo campaign = theBanker.getCampaignDebug(campaignName);

                const Strategies &memStrategies = campaign.strategies_;
                const StrategyInfo &stratinfo =
                        memStrategies.find(strategy2)->second;
                BOOST_CHECK_EQUAL(stratinfo.available_, MicroUSD(5000000));
                BOOST_CHECK_EQUAL(stratinfo.committed_, MicroUSD(15000000));
                BOOST_CHECK_EQUAL(stratinfo.commitments_.size(), 1);
                // And win the bid
                cerr << "\t" << testNum++ << ". Win the bid for $15: "
                        << campaignName << " strategy : " << strategy2 << endl;

                theBanker.winBid(campaignName, strategy2, "bid1", MicroUSD(15000000));
                theBanker.sync();
                CampaignInfo wincampaign = theBanker.getCampaignDebug(campaignName);
                // Now make sure we have the right accounting
                const Strategies &winStrategies = wincampaign.strategies_;
                const StrategyInfo &winStratInfo =
                         winStrategies.find(strategy2)->second;
                BOOST_CHECK_EQUAL(winStratInfo.available_, MicroUSD(5000000));
                BOOST_CHECK_EQUAL(winStratInfo.committed_, USD(0));
                BOOST_CHECK_EQUAL(winStratInfo.spent_, MicroUSD(15000000));
                BOOST_CHECK_EQUAL(winStratInfo.commitments_.size(), 0);
                BOOST_CHECK_EQUAL(wincampaign.spent_, MicroUSD(15000000));
            }
        }
        {
            // Test - Cancel
            // cancel a bid we want to be back to the original
            cerr << "\t" << testNum++ << ". Cancel previous bid: "
                    << campaignName << endl;
            theBanker.cancelBid(campaignName, strategy1, "bid1");
            theBanker.sync();
            CampaignInfo campaign = theBanker.getCampaignDebug(campaignName);

            const Strategies &memStrategies = campaign.strategies_;
            const StrategyInfo &stratinfo =
                    memStrategies.find(strategy1)->second;
            BOOST_CHECK_EQUAL(stratinfo.available_, MicroUSD(20000000));
            BOOST_CHECK_EQUAL(stratinfo.committed_, MicroUSD(0));
            BOOST_CHECK_EQUAL(stratinfo.commitments_.size(), 0);
        }
        {
            // Test - win a bid and pay a lower price
            // Did on something for a value of $19
            cerr << "\t" << testNum++ << ". Win a bid but pay a lower price: "
                    << campaignName << endl;
            theBanker.authorizeBid(campaignName, strategy1, "bid1", MicroUSD(19000000));
            theBanker.winBid(campaignName, strategy1, "bid1", MicroUSD(15000000));
            theBanker.sync();
            CampaignInfo wincampaign = theBanker.getCampaignDebug(campaignName);

            const Strategies &memStrategies = wincampaign.strategies_;
            const StrategyInfo &stratinfo =
                    memStrategies.find(strategy1)->second;
            BOOST_CHECK_EQUAL(stratinfo.available_, MicroUSD(5000000));
            BOOST_CHECK_EQUAL(stratinfo.committed_, MicroUSD(0));
            BOOST_CHECK_EQUAL(stratinfo.spent_, MicroUSD(15000000));
            BOOST_CHECK_EQUAL(stratinfo.commitments_.size(), 0);
            BOOST_CHECK_EQUAL(wincampaign.spent_, MicroUSD(30000000));
        }
        Json::Value campaignStatus = theBanker.dumpAllCampaignsJson();
        cerr << "campaign status : " << campaignStatus << endl;
        {
            cerr << "\t" << testNum++
                    << ". submit a bit for a certain amount but pay more than available: "
                    << campaignName << endl;
            // **
            // Test - win a bid but pay more for it than is available. At this point we have $5
            // Bid $4.90
            BOOST_CHECK(theBanker.authorizeBid(campaignName, strategy1, "bid1", MicroUSD(4900000)));

            // We also make sure we cannot submit the same bid twice
            BOOST_CHECK_THROW(
                              theBanker.authorizeBid(campaignName, strategy1, "bid1", MicroUSD(4900000)),
                    ML::Exception);

            // Now win the bid at a higher price
            theBanker.winBid(campaignName, strategy1, "bid1", MicroUSD(5100000));
            CampaignInfo winCampaign = theBanker.getCampaignDebug(campaignName);
            const Strategies &memStrategies = winCampaign.strategies_;
            const StrategyInfo &stratinfo =
                    memStrategies.find(strategy1)->second;
            BOOST_CHECK_EQUAL(stratinfo.available_, MicroUSD(-100000));
            BOOST_CHECK_EQUAL(stratinfo.committed_, MicroUSD(0));
            BOOST_CHECK_EQUAL(stratinfo.commitments_.size(), 0);
            // we should have spent $20.10
            BOOST_CHECK_EQUAL(stratinfo.spent_, USD(20.10));
        }
        {
            cerr << "\t" << testNum++
                    << ". submit a bit with no funds available: "
                    << campaignName << endl;
            // Test. Submit a bid with no funds available. This is synchronous
            // so we don't use a promise
            bool result = theBanker.authorizeBid(campaignName, strategy1, "bid1",
                                                 MicroUSD(4900000));
            BOOST_CHECK_EQUAL(result, false);
            theBanker.sync();
        }
        {
            // simulate the case where we have a campaign already in the database
            // when we add the campaign we expect it to be loaded
            theController.addCampaignSync(campaign4);
            CampaignInfo campaign = theBanker.getCampaignDebug(campaign4);
            cerr << "\t" << testNum++
                    << ". Added TestCampaign4 and got the result: "
                    << campaign << endl;
            BOOST_CHECK_EQUAL(campaign.available_, MicroUSD(1000000));
            BOOST_CHECK_EQUAL(campaign.transferred_, MicroUSD(100000));
        }
        {
            // simulate the case where we have a campaign and its associated strategy
            // already in the database. When we add a strategy to the
            // campaign we expect them both to be loaded.
            theController.addStrategySync(campaign4, strategy1);
            CampaignInfo campaign = theBanker.getCampaignDebug(campaign4);
            cerr << "\t" << testNum++ << ". Added strategy : " << strategy1
                    << " to campaign " << campaign4 << endl;
            BOOST_CHECK_EQUAL(campaign.available_, MicroUSD(1000000));
            BOOST_CHECK_EQUAL(campaign.transferred_, MicroUSD(100000));
            BOOST_CHECK_EQUAL(campaign.spent_, MicroUSD(10000));

            const Strategies &memStrategies = campaign.strategies_;
            const StrategyInfo &stratinfo =
                    memStrategies.find(strategy1)->second;
            BOOST_CHECK_EQUAL(stratinfo.available_, MicroUSD(90000));
            BOOST_CHECK_EQUAL(stratinfo.transferred_, MicroUSD(100000));
            BOOST_CHECK_EQUAL(stratinfo.spent_, MicroUSD(10000));
            BOOST_CHECK_EQUAL(stratinfo.commitments_.size(), 0);
        }
        cerr << "campaign status at the end : " << theBanker.dumpAllCampaignsJson();
    }
}

#endif

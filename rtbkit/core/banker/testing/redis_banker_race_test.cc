/* redis_banker_race_test.cc
   Jeremy Barnes, 2 November 2012
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


BOOST_AUTO_TEST_CASE( test_topup_transfer_race )
{
    auto s = std::make_shared<ServiceProxies>();
    RedisTemporaryServer redis;

    // Create a campaign with a budget
    {
        RedisBudgetController controller("bankerTest", redis);
        controller.addCampaignSync("testCampaign");
        controller.addStrategySync("testCampaign", "testStrategy");
        controller.setBudgetSync("testCampaign", MicroUSD(1000000));
    }

    // Do 1,000 topup transfers of one micro

    int nTopupThreads = 2;
    int nAddBudgetThreads = 2;
    int nBidThreads = 1; // for the moment, we don't support multiple2;
    int numTransfersPerThread = 10000;
    int numAddBudgetsPerThread = 1000;

    auto runTopupThread = [&] ()
        {
            RedisBudgetController controller("bankerTest", redis);

            for (unsigned i = 0;  i < numTransfersPerThread;  ++i) {
                controller.topupTransferSync("testCampaign",
                                                       "testStrategy",
                                                       MicroUSD(1));
            }
        };

    auto runAddBudgetThread = [&] ()
        {
            RedisBudgetController controller("bankerTest", redis);

            for (unsigned i = 0;  i < numAddBudgetsPerThread;  ++i) {
                controller.addBudgetSync("testCampaign", MicroUSD(1));
            }
        };

    uint64_t numBidsCommitted = 0;
    volatile bool finished = false;

    auto runBidThread = [&] (int threadNum)
        {
            RedisBanker banker("bankerTest", "b", s, redis);

            string item = "thread" + to_string(threadNum);

            while (!finished) {
                // Authorize 10
                if (!banker.authorizeBid("testCampaign", "testStrategy", item,
                                         MicroUSD(1))) {
                    banker.sync();
                    continue;
                }

                // Commit 1
                banker.commitBid("testCampaign", "testStrategy", item,
                                 MicroUSD(1));

                ML::atomic_inc(numBidsCommitted);
            }

            banker.sync();
        };

    boost::thread_group topupThreads;
    for (unsigned i = 0;  i < nTopupThreads;  ++i)
        topupThreads.create_thread(runTopupThread);

    for (unsigned i = 0;  i < nAddBudgetThreads;  ++i)
        topupThreads.create_thread(runAddBudgetThread);

    boost::thread_group bidThreads;
    for (unsigned i = 0;  i < nBidThreads;  ++i)
        bidThreads.create_thread(std::bind(runBidThread, i));
    

    topupThreads.join_all();

    finished = true;

    bidThreads.join_all();
    
    RedisBanker banker("bankerTest", "b", s, redis);
    banker.sync();
    Json::Value status = banker.getCampaignStatusJson("testCampaign", "");

    cerr << status << endl;

    uint32_t amountAdded       = nAddBudgetThreads * numAddBudgetsPerThread;
    uint32_t amountTransferred = nTopupThreads * numTransfersPerThread;

    cerr << "numBidsCommitted = "  << numBidsCommitted << endl;
    cerr << "amountTransferred = " << amountTransferred << endl;
    cerr << "amountAdded =       " << amountAdded << endl;



    BOOST_CHECK_EQUAL(status["available"]["micro-USD"].asInt(), 1000000 - amountTransferred + amountAdded);
    BOOST_CHECK_EQUAL(status["strategies"][0]["available"]["micro-USD"].asInt(),
                      amountTransferred - numBidsCommitted);
    BOOST_CHECK_EQUAL(status["strategies"][0]["transferred"]["micro-USD"].asInt(),
                      amountTransferred);
    BOOST_CHECK_EQUAL(status["strategies"][0]["spent"]["micro-USD"].asInt(),
                      numBidsCommitted);
    BOOST_CHECK_EQUAL(status["spent"]["micro-USD"].asInt(), numBidsCommitted);

    //BOOST_CHECK_EQUAL(status["available"].
}

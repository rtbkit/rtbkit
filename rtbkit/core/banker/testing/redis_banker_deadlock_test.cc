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
#include "jml/utils/testing/watchdog.h"
#include "dataflow/work_scheduler.h"
#include <future>
#include "soa/service/testing/redis_temporary_server.h"
#include <boost/thread/thread.hpp>


using namespace std;
using namespace ML;
using namespace Datacratic;
using namespace RTBKIT;
using namespace Redis;


BOOST_AUTO_TEST_CASE( test_banker_deadlock )
{
    auto s = std::make_shared<ServiceProxies>();
    RedisTemporaryServer redis;

    // Create a campaign with a budget and a strategy
    {
        RedisBudgetController controller("bankerTest", redis);
        controller.addCampaignSync("testCampaign");
        controller.addStrategySync("testCampaign", "testStrategy");
        controller.setBudgetSync("testCampaign", MicroUSD(100000000));
        controller.topupTransferSync("testCampaign", "testStrategy",
                                               MicroUSD(90000000));
    }

    // Do 1,000 topup transfers of one micro

    int nSyncThreads = 10;
    int nBidThreads = 1; // for the moment, we don't support multiple;

    RedisBanker banker("bankerTest", "b", s, redis);

    volatile bool finished = false;

    auto runSyncThread = [&] ()
        {
            while (!finished) {
                banker.sync();
            }
        };
    
    uint64_t numBidsCommitted = 0;

    auto runBidThread = [&] (int threadNum)
        {

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

    boost::thread_group threads;
    for (unsigned i = 0;  i < nSyncThreads;  ++i)
        threads.create_thread(runSyncThread);
    for (unsigned i = 0;  i < nBidThreads;  ++i)
        threads.create_thread(std::bind(runBidThread, i));

    // Test for ADGR-213; the expected behaviour is no deadlock and we print
    // one message per second and succeed.  If we deadlock the watchdog will
    // catch us and we will fail.
    ML::Watchdog watchdog(10.0);
    
    for (unsigned i = 0;  i < 3;  ++i) {
        ML::sleep(1.0);
        Json::Value status = banker.getCampaignStatusJson("testCampaign", "");
        cerr << "status " << status << endl;
    }
    finished = true;

    cerr << "finished; waiting to join" << endl;
    threads.join_all();
    cerr << "done" << endl;
}

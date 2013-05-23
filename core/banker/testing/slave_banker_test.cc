/* slave_banker_test.cc
   Jeremy Barnes, 8 November 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <memory>
#include <boost/test/unit_test.hpp>
#include "jml/arch/format.h"
#include "jml/arch/exception_handler.h"
#include "jml/utils/guard.h"
#include "rtbkit/core/banker/master_banker.h"
#include "rtbkit/core/banker/slave_banker.h"
#include "jml/utils/environment.h"
#include "jml/arch/timers.h"
#include "jml/utils/testing/watchdog.h"
#include <future>
#include <boost/thread/thread.hpp>
#include "soa/service/testing/zookeeper_temporary_server.h"

using namespace std;
using namespace ML;
using namespace Datacratic;
using namespace RTBKIT;

#if 1

BOOST_AUTO_TEST_CASE( test_master_slave_banker )
{
    ZooKeeper::TemporaryServer zookeeper;
    zookeeper.start();

    auto proxies = std::make_shared<ServiceProxies>();
    proxies->useZookeeper(ML::format("localhost:%d", zookeeper.getPort()));

    string bankerServiceName = "rtbBanker";

    MasterBanker master(proxies, bankerServiceName);
    master.init(make_shared<NoBankerPersistence>());
    master.monitorProviderClient.inhibit_ = true;
    auto addr = master.bindTcp();

    cerr << "master banker is listening on " << addr.first << ","
         << addr.second << endl;

    auto uri = proxies->getServiceClassInstances(bankerServiceName, "http");
    std::cout << "reachable from " << uri << std::endl;

    master.start();

    proxies->config->dump(cerr);
    
    SlaveBudgetController slave;
    slave.init(proxies->config);
    slave.start();
    slave.addAccountSync({"hello", "world"});

    BOOST_CHECK_THROW(slave.addAccountSync({"$$#$#Q@", "  asdad0321 "}),
                      std::exception);

    slave.setBudgetSync("hello", USD(200));

    cerr << "finished adding account" << endl;

    //slave.shutdown();

    SlaveBanker banker(proxies->zmqContext);
    banker.init(proxies->config, "slave");
    banker.start();
    banker.addSpendAccountSync({"hello", "world"});

    for (unsigned i = 0;  i < 2;  ++i) {
        ML::sleep(1.0);
        cerr << slave.getAccountSummarySync({"hello"}, 3) << endl;
    }

    banker.shutdown();
}
#endif

#if 1
BOOST_AUTO_TEST_CASE( test_initialization_and_spending )
{
    /* We test that the initialization of an account works even when there
       was spending that occurred earlier on the account.
       
       The specific situation is to make sure that we properly track spend
       that occurs *before* a banker can make an initial synchronization with
       the master banker.
    */

    ZooKeeper::TemporaryServer zookeeper;
    zookeeper.start();

    auto proxies = std::make_shared<ServiceProxies>();
    proxies->useZookeeper(ML::format("localhost:%d", zookeeper.getPort()));

    MasterBanker master(proxies);
    master.init(make_shared<NoBankerPersistence>());
    master.monitorProviderClient.inhibit_ = true;
    auto addr = master.bindTcp();
    cerr << "master banker is listening on " << addr.first << ","
         << addr.second << endl;

    master.start();

    proxies->config->dump(cerr);
    
    SlaveBudgetController slave;
    slave.init(proxies->config);
    slave.start();
    slave.addAccountSync({"hello", "world"});
    slave.setBudgetSync("hello", USD(200));
    slave.topupTransferSync({"hello", "world"}, USD(20));

    // Record some spend in an initial slave
    {
        SlaveBanker banker(proxies->zmqContext);
        banker.init(proxies->config, "slave");
        banker.start();
        banker.addSpendAccountSync({"hello", "world"});

        // Spend $1
        banker.forceWinBid({"hello", "world"}, USD(1), LineItems());

        // Synchronize
        banker.syncAllSync();
        
        banker.shutdown();
    }

    // Check that the spend was recorded
    auto summ = slave.getAccountSummarySync({"hello", "world"}, 1 /* depth */);
    cerr << "after initial spend was recorded" << endl;
    cerr << summ << endl;

    CurrencyPool total = USD(1);
    total += Amount(CurrencyCode::CC_IMP, 1);

    BOOST_CHECK_EQUAL(summ.spent, total);

    // Now asynchronously start up and record another dollar of spend
    {
        SlaveBanker banker(proxies->zmqContext);
        banker.init(proxies->config, "slave");
        banker.start();
        banker.addSpendAccountSync({"hello", "world"});

        // Spend $2
        banker.forceWinBid({"hello", "world"}, USD(2), LineItems());
        
        auto st = banker.getAccountStateDebug({"hello","world"});
        cerr << "after forceWinBid" << endl;
        cerr << st << endl;

        // Allow the synchronization to work
        ML::sleep(2.0);

        st = banker.getAccountStateDebug({"hello","world"});
        cerr << "after synchronization" << endl;
        cerr << st << endl;

        // Synchronize
        banker.syncAllSync();
        
        banker.shutdown();
    }

    // Check that the spend was recorded
    cerr << "after second spend recorded" << endl;
    summ = slave.getAccountSummarySync({"hello", "world"}, 1 /* depth */);
    
    cerr << summ << endl;

    total += USD(2);
    total += Amount(CurrencyCode::CC_IMP, 1);

    BOOST_CHECK_EQUAL(summ.spent, total);
}
#endif

#if 1
BOOST_AUTO_TEST_CASE( test_bidding_with_slave )
{
    ZooKeeper::TemporaryServer zookeeper;
    zookeeper.start();

    auto proxies = std::make_shared<ServiceProxies>();
    proxies->useZookeeper(ML::format("localhost:%d", zookeeper.getPort()));

    MasterBanker master(proxies);
    master.init(make_shared<NoBankerPersistence>());
    master.monitorProviderClient.inhibit_ = true;
    auto addr = master.bindTcp();
    cerr << "master banker is listening on " << addr.first << ","
         << addr.second << endl;

    master.start();

    AccountKey campaign("campaign");
    AccountKey strategy("campaign:strategy");

    SlaveBudgetController slave;
    slave.init(proxies->config);
    slave.start();

    // Create a budget for the campaign
    slave.addAccountSync(strategy);
    slave.setBudgetSync(campaign.front(), USD(100));

    int nTopupThreads = 2;
    int nAddBudgetThreads = 2;
    int nBidThreads = 2; 
    int nCommitThreads = 1;
    int numTransfersPerThread = 10000;
    int numAddBudgetsPerThread = 10;

    volatile bool finished = false;

    auto runTopupThread = [&] ()
        {
            SlaveBudgetController slave;
            slave.init(proxies->config);
            slave.start();

            while (!finished) {
                slave.topupTransferSync(strategy, USD(2.00));
                ML::sleep(1.0);
            }
        };

    auto runAddBudgetThread = [&] ()
        {
            SlaveBudgetController slave;
            slave.init(proxies->config);
            slave.start();
            
            for (unsigned i = 0;  i < numAddBudgetsPerThread;  ++i) {
                slave.setBudgetSync(campaign.front(), USD(100 + i * 2));
                
                AccountSummary summary
                    = slave.getAccountSummarySync(campaign, 3);
                cerr << summary << endl;

                ML::sleep(1.0);
            }
        };

    uint64_t numBidsCommitted = 0;

    ML::RingBufferSRMW<Amount> toCommitThread(10000);
    

    auto runBidThread = [&] (int threadNum)
        {
            SlaveBanker slave(proxies->zmqContext);
            slave.init(proxies->config, "bid" + to_string(threadNum));
            slave.start();

            AccountKey account = strategy;

            slave.addSpendAccountSync(account);

            int done = 0;
            for (;  !finished;  ++done) {
                string item = "item";

                // Authorize 10
                if (!slave.authorizeBid(account, item, MicroUSD(1))) {
                    ML::sleep(0.01);
                    continue;
                }

                // In half of the cases, we cancel.  In the other half, we
                // transfer it off to the commit thread

                if (done % 2 == 0) {
                    // Commit 1
                    slave.commitBid(account, item, MicroUSD(1), LineItems());
                    ML::atomic_inc(numBidsCommitted);
                }
                else {
                    Amount amount = slave.detachBid(account, item);
                    toCommitThread.push(amount);
                }
            }

            cerr << "finished slave account with "
                 << done << " bids" << endl;

            slave.syncAllSync();
        };

    auto runCommitThread = [&] (int threadNum)
        {
            SlaveBanker slave(proxies->zmqContext);
            slave.init(proxies->config, "commit" + to_string(threadNum));
            slave.start();

            AccountKey account = strategy;
            slave.addSpendAccountSync(account);

            while (!finished || toCommitThread.couldPop()) {
                Amount amount;
                if (toCommitThread.tryPop(amount, 0.1)) {

                    try {
                        slave.attachBid(account, "item", amount);
                        slave.commitBid(account, "item", MicroUSD(1), LineItems());
                        //slave.commitDetachedBid(account, amount, MicroUSD(1), LineItems());
                    } catch (...) {
                        cerr << "commit detached " << amount << " from "
                             << slave.getAccount(account) << endl;
                        throw;
                    }
                    ML::atomic_inc(numBidsCommitted);
                }
                //slave.syncTo(master);
            }

            //slave.syncTo(master);
            cerr << "done commit thread" << endl;

            slave.syncAllSync();
        };

    boost::thread_group budgetThreads;

    for (unsigned i = 0;  i < nAddBudgetThreads;  ++i)
        budgetThreads.create_thread(runAddBudgetThread);

    boost::thread_group bidThreads;
    for (unsigned i = 0;  i < nBidThreads;  ++i)
        bidThreads.create_thread(std::bind(runBidThread, i));

    for (unsigned i = 0;  i < nTopupThreads;  ++i)
        bidThreads.create_thread(runTopupThread);

    for (unsigned i = 0;  i < nCommitThreads;  ++i)
        bidThreads.create_thread(std::bind(runCommitThread, i));
    

    budgetThreads.join_all();

    finished = true;

    bidThreads.join_all();

    uint32_t amountAdded       = nAddBudgetThreads * numAddBudgetsPerThread;
    uint32_t amountTransferred = nTopupThreads * numTransfersPerThread;

    cerr << "numBidsCommitted = "  << numBidsCommitted << endl;
    cerr << "amountTransferred = " << amountTransferred << endl;
    cerr << "amountAdded =       " << amountAdded << endl;
    
    AccountSummary summary
        = slave.getAccountSummarySync(campaign, 3);
    cerr << summary << endl;

#if 0
    cerr << "campaign" << endl;
    cerr << master.getAccountSummary(campaign) << endl;
    cerr << master.getAccount(campaign) << endl; 

    cerr << "strategy" << endl;
    cerr << master.getAccountSummary(strategy) << endl;
    cerr << master.getAccount(strategy) << endl; 
#endif

#if 0    
    RedisBanker banker("bankerTest", "b", s, redis);
    banker.sync();
    Json::Value status = banker.getCampaignStatusJson("testCampaign", "");

    cerr << status << endl;




    BOOST_CHECK_EQUAL(status["available"]["micro-USD"].asInt(), 1000000 - amountTransferred + amountAdded);
    BOOST_CHECK_EQUAL(status["strategies"][0]["available"]["micro-USD"].asInt(),
                      amountTransferred - numBidsCommitted);
    BOOST_CHECK_EQUAL(status["strategies"][0]["transferred"]["micro-USD"].asInt(),
                      amountTransferred);
    BOOST_CHECK_EQUAL(status["strategies"][0]["spent"]["micro-USD"].asInt(),
                      numBidsCommitted);
    BOOST_CHECK_EQUAL(status["spent"]["micro-USD"].asInt(), numBidsCommitted);

    //BOOST_CHECK_EQUAL(status["available"].
#endif
}
#endif

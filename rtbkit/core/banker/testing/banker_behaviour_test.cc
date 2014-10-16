/* banker_behaviour_test.cc
   Wolfgang Sourdeau, 20 December 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Functional tests of the master and slave bankers
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <unordered_set>

#include "boost/test/unit_test.hpp"
#include "jml/arch/format.h"

#include "soa/service/service_base.h"
#include "soa/service/testing/redis_temporary_server.h"
#include "soa/service/testing/zookeeper_temporary_server.h"

#include "rtbkit/core/banker/master_banker.h"
#include "rtbkit/core/banker/slave_banker.h"

#include "banker_temporary_server.h"

using namespace std;

using namespace Datacratic;
using namespace RTBKIT;
using namespace Redis;

namespace {

void
RedisReplyFillSetOfStrings(const Redis::Reply & reply,
                           unordered_set<string> & aSet) {
    ExcAssert(reply.type() == Redis::ARRAY);
    for (int i = 0; i < reply.length(); i++) {
        aSet.insert(reply[i].asString());
    }
}

}

#if 0
/* make sure that requests from the slave banker are properly deferred when
 * the master banker is not available */
BOOST_AUTO_TEST_CASE( test_banker_deferred )
{
    ZooKeeper::TemporaryServer zookeeper;
    zookeeper.start();

    auto proxies = std::make_shared<ServiceProxies>();
    proxies->useZookeeper(ML::format("localhost:%d", zookeeper.getPort()));

    /* spawn slave */
    SlaveBanker slave(proxies->zmqContext);
    slave.init(proxies->config, "slave");
    slave.start();

    /* invoke addSpendAccount but wait 3 seconds before spawning master */
    int done(false);
    auto onSpendAdded = [&] (std::exception_ptr exc, ShadowAccount &&shadowAccount) {
        cerr << "(slave) onSpendAdded..." << endl;
        done = true;
        ML::futex_wake(done);
    };
    slave.addSpendAccount({"hello", "world"}, CurrencyPool(), onSpendAdded);
    ML::sleep(1);

    /* spawn master */
    RedisTemporaryServer redis;
    BankerTemporaryServer master(redis, ML::format("localhost:%d", zookeeper.getPort()));

    /* wait for the addSpendAccount op to complete */
    while (!done) {
        ML::futex_wait(done, false);
    }

    /* shut down the master banker now to force a write to redis */
    master.shutdown();

    /* check that "hello:world" has been saved */
    auto connection = std::make_shared<AsyncConnection>(redis);
    Redis::Result result = connection->exec(SMEMBERS("banker:accounts"), 5);
    BOOST_CHECK(result.ok());
    const Reply & keysReply = result.reply();
    unordered_set<string> keys;
    RedisReplyFillSetOfStrings(keysReply, keys);
    BOOST_CHECK_EQUAL(keys.count("hello:world"), 1);
}
#endif

/* make sure that the slave banker does create a spend account, even when a
 * "legacyImported" account exists */
BOOST_AUTO_TEST_CASE( test_banker_slave_banker_accounts )
{
    ZooKeeper::TemporaryServer zookeeper;
    zookeeper.start();

    auto proxies = std::make_shared<ServiceProxies>();
    proxies->useZookeeper(ML::format("localhost:%d", zookeeper.getPort()));

    /* spawn master */
    RedisTemporaryServer redis;

    /* create some initial accounts */
    {
        MasterBanker master(proxies, "initBanker");
        auto storage = make_shared<RedisBankerPersistence>(redis);
        master.init(storage);
        master.start();

        master.accounts.createAccount({"top"}, AT_BUDGET);
        master.accounts.setBudget({"top"}, MicroUSD(15729914170));
        master.accounts.setBalance({"top", "sub"}, MicroUSD(11005485799),
                                   AT_BUDGET);
        master.accounts.setBalance({"top", "sub", "legacyImported"},
                                   MicroUSD(10991979974),
                                   AT_SPEND);
        master.accounts.importSpend({"top", "sub", "legacyImported"},
                                    MicroUSD(10991979974));
        /* save the initial accounts state (the async save is guaranteed to
         * be over when "master" is destroyed) */
        master.saveState();
    }

    /* instantiate a slave and add a spend account from it */
    {
        /* spawn a master banker server process */
        BankerTemporaryServer master(redis, ML::format("localhost:%d", zookeeper.getPort()));
        ML::sleep(2);

        /* spawn slave */
        SlaveBanker slave("slaveBanker");
        slave.setApplicationLayer(make_application_layer<ZmqLayer>(proxies));
        slave.start();

        slave.addSpendAccountSync({"top", "sub"});
        
        ML::sleep(2);
    }

    /* ensure that "top:sub:slaveBanker" has been created. If not, an
     * exception will be thrown. */
    {
        MasterBanker master(proxies, "initBanker2");
        auto storage = make_shared<RedisBankerPersistence>(redis);
        master.init(storage);
        master.start();
        auto account
            = master.accounts.getAccount({"top", "sub", "slaveBanker"});
        cerr << "account: " << account;
    }
}

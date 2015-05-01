/* master_banker_test.cc
   Wolfgang Sourdeau, 13 December 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.
   
   Unit tests for the MasterBanker class
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <memory>
#include <boost/test/unit_test.hpp>
#include "rtbkit/common/account_key.h"
#include "jml/arch/timers.h"
#include "jml/utils/exc_assert.h"
#include "soa/types/date.h"
#include "soa/service/service_base.h"
#include "rtbkit/common/currency.h"

#include "rtbkit/core/banker/local_banker.h"

using namespace std;
using namespace ML;
using namespace Datacratic;
using namespace RTBKIT;

BOOST_AUTO_TEST_CASE( test_local_banker )
{
    auto proxies = make_shared<ServiceProxies>();
    LocalBanker rBanker(proxies, ROUTER, "router");
    LocalBanker pBanker(proxies, POST_AUCTION, "pal");
    rBanker.init("http://127.0.0.1:27890");
    pBanker.init("http://127.0.0.1:27890");
    rBanker.start();
    pBanker.start();
    
    AccountKey routerAccounts[100];
    AccountKey palAccounts[100];

    auto start = Date().now();
    for (int i = 0; i < 100; ++i) {
        stringstream ss;
        ss << "test" << i << ":account" << i;
        string key = ss.str();
        AccountKey rkey(key);
        AccountKey pkey(key);
        rBanker.addSpendAccount(rkey, {}, nullptr);
        pBanker.addSpendAccount(pkey, {}, nullptr);

        routerAccounts[i] = rkey;
        palAccounts[i] = pkey;
    }
    auto end = Date().now();
    auto taken = end - start;
    while (pBanker.accounts.accounts.size() < 100 || rBanker.accounts.accounts.size() < 100) {
        ML::sleep(0.01);
        //cout << "p: " << pBanker.accounts.accounts.size()
        //     << " r: " << rBanker.accounts.accounts.size() << endl;
        continue;
    }
    cout << "time taken: " << taken << endl;

    ML::sleep(1.0);

    rBanker.sync();

    Amount bidPrice = MicroUSD(2);
    for (auto key : routerAccounts) {
        bool allowed = rBanker.authorizeBid(key, "", bidPrice);
        cout << key.toString() << " bid: " << allowed << endl;
    }

    rBanker.sync();

    ML::sleep(1.0);

    pBanker.sync();

    Amount winPrice = MicroUSD(2);
    for (auto key : palAccounts) {
        pBanker.winBid(key, "", winPrice);
    }

    pBanker.sync();

    auto acc = pBanker.accounts.accounts[AccountKey("test10:account10:pal")];
    acc.pal->imp = 0;
    pBanker.accounts.accounts[AccountKey("test10:account10:pal")] = acc;

    pBanker.sync();

    ML::sleep(2.0);

    rBanker.shutdown();
    pBanker.shutdown();
}


BOOST_AUTO_TEST_CASE( test_router_accumulate )
{
    auto proxies = make_shared<ServiceProxies>();
    LocalBanker rBanker(proxies, ROUTER, "router");
    // Max spend rate per sec set to 10,000 USD/1M
    rBanker.setSpendRate(MicroUSD(10000));

    string key = "parent:child:router";

    rBanker.accounts.addFromJsonString("{\"name\":\"" + key + "\",\"type\":\"Router\","
            "\"parent\":\"parent:sub\",\"rate\":1000,\"balance\":0}");

    ExcAssertEqual(rBanker.accounts.getBalance(key).value, 0);
    rBanker.accounts.accumulateBalance(key, MicroUSD(1000));
    ExcAssertEqual(rBanker.accounts.getBalance(key).value, 1000);
    rBanker.accounts.accumulateBalance(key, MicroUSD(0));
    ExcAssertEqual(rBanker.accounts.getBalance(key).value, 1000);
    rBanker.accounts.accumulateBalance(key, MicroUSD(9000));
    ExcAssertEqual(rBanker.accounts.getBalance(key).value, 10000);
    // should stay at 10,000 since it's the spend per sec max.
    rBanker.accounts.accumulateBalance(key, MicroUSD(1000));
    ExcAssertEqual(rBanker.accounts.getBalance(key).value, 10000);

}

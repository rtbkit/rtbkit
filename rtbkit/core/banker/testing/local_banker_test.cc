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
#include "soa/types/date.h"
#include "rtbkit/common/currency.h"

#include "rtbkit/core/banker/local_banker.h"

using namespace std;
using namespace ML;
using namespace Datacratic;
using namespace RTBKIT;

BOOST_AUTO_TEST_CASE( test_local_banker )
{
    LocalBanker rBanker(ROUTER);
    LocalBanker pBanker(POST_AUCTION);
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
        AccountKey rkey(key + ":router");
        AccountKey pkey(key + ":pal");
        rBanker.addAccount(rkey);
        pBanker.addAccount(pkey);

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

    rBanker.reauthorize();

    Amount bidPrice = MicroUSD(2);
    for (auto key : routerAccounts) {
        auto allowed = rBanker.bid(key, bidPrice);
        cout << key.toString() << " bid: " << allowed << endl;
    }

    rBanker.reauthorize();

    ML::sleep(1.0);

    pBanker.spendUpdate();

    Amount winPrice = MicroUSD(2);
    for (auto key : palAccounts) {
        auto allowed = pBanker.win(key, winPrice);
        cout << key.toString() << " win: " << allowed << endl;
    }

    pBanker.spendUpdate();

    ML::sleep(2.0);

    rBanker.shutdown();
    pBanker.shutdown();
}

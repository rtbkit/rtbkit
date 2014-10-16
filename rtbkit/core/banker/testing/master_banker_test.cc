/* master_banker_test.cc
   Wolfgang Sourdeau, 13 December 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.
   
   Unit tests for the MasterBanker class
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <memory>
#include <boost/test/unit_test.hpp>

#include "soa/service/service_base.h" // NullEventService, ServiceProxies

#include "rtbkit/core/banker/master_banker.h"

using namespace std;
using namespace Datacratic;
using namespace RTBKIT;

BOOST_AUTO_TEST_CASE( test_master_banker_onstateloaded )
{
    auto serviceProxies = std::make_shared<ServiceProxies>();

    MasterBanker testBanker(serviceProxies);
    testBanker.accounts.createAccount(AccountKey("origAccount"), AT_BUDGET);
    Json::Value initialJson = testBanker.accounts.toJson();

    shared_ptr<Accounts> newAccounts = make_shared<Accounts>();
    newAccounts->createAccount(AccountKey("newAccount:spend"), AT_SPEND);

    /* nothing must have changed with BankerPersistence::DATA_INCONSISTENCY */
    testBanker.onStateLoaded(newAccounts,
                             BankerPersistence::DATA_INCONSISTENCY,
                             "inconsistency error");
    Json::Value newJson = testBanker.accounts.toJson();
    BOOST_CHECK_EQUAL(newJson, initialJson);

    /* nothing must have changed */
    testBanker.onStateLoaded(newAccounts,
                             BankerPersistence::PERSISTENCE_ERROR,
                             "backend error");
    newJson = testBanker.accounts.toJson();
    BOOST_CHECK_EQUAL(newJson, initialJson);

    /* accounts must have been replaced with "newAccounts" this time */
    testBanker.onStateLoaded(newAccounts, BankerPersistence::SUCCESS, "");
    newJson = testBanker.accounts.toJson();
    BOOST_CHECK_EQUAL(newJson, newAccounts->toJson());

    /* an exception should be thrown for unknown status codes */
    BOOST_CHECK_THROW(testBanker.onStateLoaded(newAccounts,
                                               BankerPersistence::PersistenceCallbackStatus(-1),
                                               ""),
                      ML::Exception);
}

BOOST_AUTO_TEST_CASE( test_master_banker_onstatesaved )
{
    auto serviceProxies = std::make_shared<ServiceProxies>();

    MasterBanker testBanker(serviceProxies);
    testBanker.accounts.createAccount(AccountKey("account1"), AT_BUDGET);
    testBanker.accounts.createAccount(AccountKey("account2:spend"), AT_SPEND);
    Json::Value initialJson = testBanker.accounts.toJson();

    /* BankerPersistence::SUCCESS: nothing must have changed */
    testBanker.onStateSaved(BankerPersistence::SUCCESS, "");
    Json::Value newJson = testBanker.accounts.toJson();
    BOOST_CHECK_EQUAL(newJson, initialJson);

    /* BankerPersistence::PERSISTENCE_ERROR: nothing must have changed */
    testBanker.onStateSaved(BankerPersistence::PERSISTENCE_ERROR,
                            "backend error");
    newJson = testBanker.accounts.toJson();
    BOOST_CHECK_EQUAL(newJson, initialJson);

    /* BankerPersistence::DATA_INCONSISTENCY: accounts corresponding to the
     * keys passed in a json array must have been marked as out of sync */

    Accounts newAccounts = testBanker.accounts;
    Json::Value badKeys = Json::parse("[ 'account2:spend' ]");
    testBanker.onStateSaved(BankerPersistence::DATA_INCONSISTENCY,
                            badKeys.toString());
    BOOST_CHECK(testBanker.accounts.isAccountOutOfSync({"account2", "spend"}));

    /* an exception should be thrown for unknown status codes */
    BOOST_CHECK_THROW(testBanker.onStateSaved(BankerPersistence::PersistenceCallbackStatus(-1),
                                              ""),
                      ML::Exception);
}

BOOST_AUTO_TEST_CASE( test_master_banker_http_headers )
{
    auto serviceProxies = std::make_shared<ServiceProxies>();

    MasterBanker testBanker(serviceProxies);
    testBanker.init(std::make_shared<NoBankerPersistence>());
    auto uri = testBanker.bindTcp().second;
    testBanker.start();

    HttpRestProxy proxy(uri);

    auto response = proxy.get("/v1/accounts");
                      
    cerr << response << endl;

    BOOST_CHECK_EQUAL(response.getHeader("Access-Control-Allow-Origin"), "*");
}

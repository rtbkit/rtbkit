/* master_banker_test.cc
   Wolfgang Sourdeau, 13 December 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.
   
   Unit tests for RedisBankerPersistence class
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <jml/arch/futex.h>
#include "soa/service/redis.h"
#include "soa/service/testing/redis_temporary_server.h"

#include "rtbkit/core/banker/account.h"
#include "rtbkit/core/banker/master_banker.h"

using namespace std;

using namespace Datacratic;
using namespace RTBKIT;
using namespace Redis;

BOOST_AUTO_TEST_CASE( test_redis_persistence_loadall )
{
    RedisTemporaryServer redis;
    std::shared_ptr<AsyncConnection> connection
        = std::make_shared<AsyncConnection>(redis);
    RedisBankerPersistence storage(connection);
    int done(false);

    /* test the behaviour of rp with an empty database */
    auto OnLoaded_EmptyRedis = [&] (std::shared_ptr<Accounts> accounts,
                                    BankerPersistence::PersistenceCallbackStatus status,
                                    const string & info) {
        BOOST_CHECK_EQUAL(status, BankerPersistence::SUCCESS);
        BOOST_CHECK_EQUAL(info, "");

        /* "banker:accounts" does not exist */
        vector<AccountKey> acctKeys = accounts->getAccountKeys();
        BOOST_CHECK_EQUAL(acctKeys.size(), 0);
        done = true;
        ML::futex_wake(done);
    };
    storage.loadAll("", OnLoaded_EmptyRedis);
    while (!done) {
        ML::futex_wait(done, false);
    }

    /* wrong value type for banker:accounts */
    connection->exec(SET("banker:accounts", "myvalue"));
    done = false;
    auto OnLoaded_BadBankerAccounts1
        = [&] (std::shared_ptr<Accounts> accounts,
               BankerPersistence::PersistenceCallbackStatus status,
               const string & info) {
        /* this could be a DATA_INCONSISTENCY error, but it is handled
           directly by the backend */
        BOOST_CHECK_EQUAL(status, BankerPersistence::BACKEND_ERROR);
        BOOST_CHECK(info.length() != 0); /* we ignore the actual message */
        done = true;
        ML::futex_wake(done);
    };
    storage.loadAll("", OnLoaded_BadBankerAccounts1);
    while (!done) {
        ML::futex_wait(done, false);
    }
    connection->exec(DEL("banker:accounts"));

    /* nil/void entries in banker:accounts */
    Command sadd(SADD("banker:accounts"));
    sadd.addArg("account1");
    connection->exec(sadd);
    done = false;
    auto OnLoaded_BadBankerAccounts2
        = [&] (std::shared_ptr<Accounts> accounts,
               BankerPersistence::PersistenceCallbackStatus status,
               const string & info) {
        BOOST_CHECK_EQUAL(status, BankerPersistence::DATA_INCONSISTENCY);
        BOOST_CHECK(info.length() != 0); /* we ignore the actual message */
        // cerr << "error = " + info << endl;
        done = true;
        ML::futex_wake(done);
    };
    storage.loadAll("", OnLoaded_BadBankerAccounts2);
    while (!done) {
        ML::futex_wait(done, false);
    }

    /* valid account1 */
    Account account1;
    account1.type = AT_BUDGET;
    account1.budgetIncreases = Amount(MicroUSD(123456));
    account1.budgetDecreases = Amount(MicroUSD(0));
    account1.recycledIn = Amount(MicroUSD(987654));
    account1.allocatedIn = Amount(MicroUSD(4567));
    account1.commitmentsRetired = Amount(MicroUSD(898989));
    account1.adjustmentsIn = Amount(MicroUSD(17171717));
    account1.recycledOut = Amount(MicroUSD(64949494));
    account1.allocatedOut = Amount(MicroUSD(8778777));
    account1.commitmentsMade = Amount(MicroUSD(10101010));
    account1.adjustmentsOut = Amount(MicroUSD(9999999));
    account1.spent = Amount(MicroUSD(10111213));
    account1.balance = ((account1.budgetIncreases + account1.recycledIn
                         + account1.allocatedIn + account1.commitmentsRetired
                         + account1.adjustmentsIn)
                        - (account1.spent + account1.recycledOut
                           + account1.allocatedOut + account1.commitmentsMade
                           + account1.adjustmentsOut));

    Json::Value account1Json(account1.toJson());
    connection->exec(SET("banker-account1", account1Json.toString()));
    connection->exec(SET("banker-account2", account1Json.toString()));
    done = false;
    auto OnLoaded_ValidAccount
        = [&] (std::shared_ptr<Accounts> accounts,
               BankerPersistence::PersistenceCallbackStatus status,
               const string & info) {
        BOOST_CHECK_EQUAL(status, BankerPersistence::SUCCESS);
        /* error message should be empty */
        BOOST_CHECK(info.length() == 0);

        /* only returned account = "account1" */
        std::vector<AccountKey> accountKeys = accounts->getAccountKeys();
        BOOST_CHECK_EQUAL(accountKeys.size(), 1);
        BOOST_CHECK_EQUAL(accountKeys[0].toString(), "account1");

        Account storedAccount = accounts->getAccount(accountKeys[0]);
        Json::Value storedAccountJson = storedAccount.toJson();
        BOOST_CHECK_EQUAL(account1Json, storedAccountJson);
        done = true;
        ML::futex_wake(done);
    };
    storage.loadAll("", OnLoaded_ValidAccount);
    while (!done) {
        ML::futex_wait(done, false);
    }
}

BOOST_AUTO_TEST_CASE( test_redis_persistence_saveall )
{
    RedisTemporaryServer redis;
    std::shared_ptr<AsyncConnection> connection
        = std::make_shared<AsyncConnection>(redis);
    RedisBankerPersistence storage(connection);
    int done(false);

    /* generic callback for all subsequent saveAll invocations */
    BankerPersistence::PersistenceCallbackStatus lastStatus;
    string lastInfo;
    auto OnSavedCallback
        = [&] (BankerPersistence::PersistenceCallbackStatus status,
               const string & info) {
        lastStatus = status;
        lastInfo = info;
        done = true;
        ML::futex_wake(done);
    };

    /* basic account setup */
    Accounts accounts;
    AccountKey parentKey("parent"), childKey("parent:child");
    accounts.createAccount(parentKey, AT_BUDGET);
    accounts.createAccount(childKey, AT_SPEND);
    accounts.setBudget(parentKey, MicroUSD(123456));
    accounts.setBalance(childKey, MicroUSD(1234), AT_NONE);

    /* 1. we save an account that does not exist yet in the storage */
    storage.saveAll(accounts, OnSavedCallback);
    while (!done) {
        ML::futex_wait(done, false);
    }

    /* this operation should succeed */
    BOOST_CHECK_EQUAL(lastStatus, BankerPersistence::SUCCESS);
    BOOST_CHECK(lastInfo.length() == 0);

    /* the new accounts should have been registered in the "banker:accounts"
       set */
    Redis::Result result = connection->exec(SMEMBERS("banker:accounts"), 5);
    BOOST_CHECK(result.ok());
    const Reply & keysReply = result.reply();
    BOOST_CHECK_EQUAL(keysReply.type(), ARRAY);
    BOOST_CHECK_EQUAL(keysReply.length(), 2);
    BOOST_CHECK((keysReply[0].asString() == "parent"
                 && keysReply[1].asString() == "parent:child")
                || (keysReply[0].asString() == "parent:child"
                    && keysReply[1].asString() == "parent"));

    /* make sure that the correct data has been stored for "parent" */
    result = connection->exec(GET("banker-parent"), 5);
    BOOST_CHECK(result.ok());
    const Reply & parentReply = result.reply();
    BOOST_CHECK_EQUAL(parentReply.type(), STRING);
    Json::Value accountJson(accounts.getAccount(parentKey).toJson());
    accountJson["spent-tracking"] = Json::Value(Json::objectValue);
    Json::Value storageJson = Json::parse(parentReply.asString());
    BOOST_CHECK_EQUAL(accountJson, storageJson);

    /* make sure that the correct data has been stored for "parent:child" */
    result = connection->exec(GET("banker-parent:child"), 5);
    BOOST_CHECK(result.ok());
    const Reply & childReply = result.reply();
    BOOST_CHECK_EQUAL(childReply.type(), STRING);
    accountJson = accounts.getAccount(childKey).toJson();
    storageJson = Json::parse(childReply.asString());
    BOOST_CHECK_EQUAL(accountJson, storageJson);

    /* 2. we update an existing account and reperform the same tests */
    accounts.importSpend(childKey, MicroUSD(123));
    done = false;
    storage.saveAll(accounts, OnSavedCallback);
    while (!done) {
        ML::futex_wait(done, false);
    }

    /* this operation should succeed */
    BOOST_CHECK_EQUAL(lastStatus, BankerPersistence::SUCCESS);
    BOOST_CHECK(lastInfo.length() == 0);

    /* the same accounts should be registered in the "banker:accounts" set */
    result = connection->exec(SMEMBERS("banker:accounts"), 5);
    BOOST_CHECK(result.ok());
    const Reply & retryKeysReply = result.reply();
    BOOST_CHECK_EQUAL(retryKeysReply.type(), ARRAY);
    BOOST_CHECK_EQUAL(retryKeysReply.length(), 2);
    BOOST_CHECK((retryKeysReply[0].asString() == "parent"
                 && retryKeysReply[1].asString() == "parent:child")
                || (retryKeysReply[0].asString() == "parent:child"
                    && retryKeysReply[1].asString() == "parent"));

    /* make sure "parent:child" has been properly updated */
    result = connection->exec(GET("banker-parent:child"), 5);
    BOOST_CHECK(result.ok());
    const Reply & updatedChildReply = result.reply();
    BOOST_CHECK_EQUAL(updatedChildReply.type(), STRING);
    accountJson = accounts.getAccount(childKey).toJson();
    storageJson = Json::parse(updatedChildReply.asString());
    BOOST_CHECK_EQUAL(accountJson, storageJson);

    /* 3. we save another instance of the save accounts, like when two bankers
     * are running concurrently, and test the error reporting */

    /* we save the "future" copy */
    Accounts accounts2 = accounts;
    accounts2.importSpend(childKey, MicroUSD(123));
    done = false;
    accountJson = accounts.getAccount(childKey).toJson();
    storage.saveAll(accounts2, OnSavedCallback);
    while (!done) {
        ML::futex_wait(done, false);
    }
    /* this operation should succeed */
    BOOST_CHECK_EQUAL(lastStatus, BankerPersistence::SUCCESS);
    BOOST_CHECK(lastInfo.length() == 0);
    /* make sure "parent:child" has been properly updated with the value from
     * accounts2 */
    result = connection->exec(GET("banker-parent:child"), 5);
    BOOST_CHECK(result.ok());
    const Reply & updatedChild2Reply = result.reply();
    BOOST_CHECK_EQUAL(updatedChild2Reply.type(), STRING);
    accountJson = accounts2.getAccount(childKey).toJson();
    storageJson = Json::parse(updatedChild2Reply.asString());
    BOOST_CHECK_EQUAL(accountJson, storageJson);

    /* we attempt to save a previous state */
    done = false;
    storage.saveAll(accounts, OnSavedCallback);
    while (!done) {
        ML::futex_wait(done, false);
    }
    /* this operation should fail */
    BOOST_CHECK_EQUAL(lastStatus, BankerPersistence::DATA_INCONSISTENCY);
    /* account "parent:child" is reported as out of sync */
    BOOST_CHECK_EQUAL(lastInfo, "[\"parent:child\"]");

    /* 4. we ensure that accounts marked as out of sync are silently
     * ignored */
    Json::Value expectedStorageJson = storageJson;

    accounts2.importSpend(childKey, MicroUSD(12));
    accounts2.markAccountOutOfSync(childKey);

    done = false;
    storage.saveAll(accounts2, OnSavedCallback);
    while (!done) {
        ML::futex_wait(done, false);
    }
    /* this operation should succeed */
    BOOST_CHECK_EQUAL(lastStatus, BankerPersistence::SUCCESS);
    BOOST_CHECK(lastInfo.length() == 0);

    result = connection->exec(GET("banker-parent:child"), 5);
    BOOST_CHECK(result.ok());
    const Reply &outOfSyncChildReply = result.reply();
    BOOST_CHECK_EQUAL(outOfSyncChildReply.type(), STRING);
    storageJson = Json::parse(outOfSyncChildReply.asString());

    /* the last expense of 12 mUSD must not be present in the stored account */
    BOOST_CHECK_EQUAL(expectedStorageJson, storageJson);
}

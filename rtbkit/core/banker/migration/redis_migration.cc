/* redis_migration.cc
   Wolfgang Sourdeau, 17 December 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.
   
   Redis migration class from campaign:strategy schema to the new accounts
   schema
 */

#include <jml/arch/futex.h>
#include "soa/service/redis.h"

#include "rtbkit/core/banker/account.h"
#include "rtbkit/core/banker/master_banker.h"

#include "redis_old_types.h"
#include "redis_utils.h"

#include "redis_migration.h"

using namespace std;
using namespace Redis;

using namespace Datacratic;
using namespace RTBKIT;

typedef vector<string> ListOfStrings;

/* REDISMIGRATION */

namespace {

/* load all keys with "campaigns:" */
ListOfStrings
FetchKeys(AsyncConnection & redis)
{
    ListOfStrings keys;

    /* KEYS campaigns:* */
    Result keysResult = redis.exec(KEYS(CampaignsPrefix + "*"));
    if (!keysResult.ok())
        throw ML::Exception("redis error: " + keysResult.error());

    const Reply & keysReply = keysResult.reply();
    ExcAssert(keysReply.type() == ARRAY);

    int keysCount = keysReply.length();
    for (int i = 0; i < keysCount; i++) {
        keys.push_back(keysReply[i]);
    }

    return keys;
}

/* from all keys loaded above, build and load basic Campaign and Strategy
 * instances */
void
LoadCampaignsAndStrategies(AsyncConnection & redis, ListOfStrings & keys,
                           Campaigns & campaigns, Strategies & strategies,
                           int acceptedDelta)
{
    for (string & fullKey: keys) {
        ListOfStrings parts = ML::split(fullKey, ':');
        string campaignKey = parts[1];
        switch (parts.size()) {
        case 2: {
            Campaign newCampaign(campaignKey);
            newCampaign.load(redis);
            campaigns.insert({campaignKey, newCampaign});
            break;
        }
        case 3: {
            string & strategyKey = parts[2];
            Strategy newStrategy(strategyKey, campaignKey);
            newStrategy.load(redis, acceptedDelta);
            strategies.insert({strategyKey, newStrategy});
            break;
        }
        default:
            cerr << "! element with key '" << fullKey << "' ignored" << endl;
        }
    }
}

/* load full campaigns (including strategies) from redis storage */
void
LoadCampaigns(shared_ptr<AsyncConnection> & redis, Campaigns & campaigns,
              int acceptedDelta)
{
    ListOfStrings keys(FetchKeys(*redis));

    if (keys.size() == 0) {
        cerr << "No campaigns key found." << endl;
        return;
    }

    Strategies strategies;
    int valids(0), total(0);
    LoadCampaignsAndStrategies(*redis, keys, campaigns, strategies,
                               acceptedDelta);
    for (auto & pair: strategies) {
        pair.second.assignToParent(campaigns);
    } 
    for (auto & pair: campaigns) {
        if (pair.second.validateAll(acceptedDelta)) {
            valids++;
        }
        total++;
    }
    cerr << valids << " valid campaigns found on " << total << endl;
}

void
ConvertCampaignsToAccounts(Campaigns & campaigns, Accounts & accounts)
{
    for (auto & cPair: campaigns) {
        const Campaign & campaign = cPair.second;
        AccountKey parentKey(campaign.key_);
        if (campaign.validateAll(0)) {
            /* campaign = parent account (budget),
               strategy = child account (strategy budget),
               strategy-spent = grandchild account (strategy spent) */
            accounts.createBudgetAccount(parentKey);

            /* compute initial budget */
            MicroUSD budget(campaign.available_ + campaign.transferred_);
            accounts.setBudget(parentKey, CurrencyPool(budget));
            // parentAccount.checkInvariants();
            
            for (const Strategy & strategy: campaign.strategies) {
                AccountKey budgetKey = parentKey.childKey(strategy.key_);
                Account budgetAccount = accounts.createBudgetAccount(budgetKey);
                MicroUSD childTransferred(strategy.transferred_);

                accounts.setBalance(budgetKey, CurrencyPool(childTransferred), AT_NONE);

                // cerr << "budgetAccount: " << budgetAccount.toJson().toString()
                //      << endl;
                AccountKey spendKey = budgetKey.childKey("legacyImported");
                Account spendAccount = accounts.createSpendAccount(spendKey);
                MicroUSD childSpent(strategy.spent_);
                accounts.setBalance(spendKey, childSpent, AT_NONE);
                accounts.importSpend(spendKey, childSpent);

                // cerr << "spendAccount: " << spendAccount.toJson().toString()
                //      << endl;
            }
            // cerr << "parentAccount: " << parentAccount.toJson().toString()
            //      << endl;
            // throw ML::Exception("breakpoint");
            cerr << "- conversion of " << parentKey << " completed" << endl;
        }
        else {
            cerr << "! conversion of " << parentKey << " skipped (invalid)" << endl;
        }
    }
}

void StoreAccounts(shared_ptr<Redis::AsyncConnection> & redis,
                   Accounts & accounts) {
    RedisBankerPersistence storage(redis);
    volatile int done(0);
    BankerPersistence::OnSavedCallback callback
        = [&](const BankerPersistence::Result& result,
              const std::string & info) {
        switch (result.status) {
        case BankerPersistence::SUCCESS: {
            cerr << "- accounts successfully saved" << endl;
            break;
        }
        case BankerPersistence::DATA_INCONSISTENCY: {
            Json::Value accountKeys = Json::parse(info);
            ExcAssert(accountKeys.type() == Json::arrayValue);
            for (Json::Value jsonKey: accountKeys) {
                ExcAssert(jsonKey.type() == Json::stringValue);
                string keyStr = jsonKey.asString();
                accounts.markAccountOutOfSync(AccountKey(keyStr));
                cerr << "! account '" << keyStr << "' is out of sync" << endl;
            }
            break;
        }
        case BankerPersistence::PERSISTENCE_ERROR: {
            /* the backend is unavailable */
            cerr << "! a redis error occurred: " + info << endl;
        }
        default: {
            throw ML::Exception("erroneous status code");
        }
        }
        done = 1;
        ML::futex_wake(done);
    };
    storage.saveAll(accounts, callback);
    while (done == 0) {
        ML::futex_wait(done, 0);
    }
}

}

void
RedisMigration::
perform(const Redis::Address & sourceAddress, int acceptedDelta,
        const Redis::Address & targetAddress)
{
    cerr << "* Loading campaigns and strategies..." << endl;
    std::shared_ptr<AsyncConnection> sourceRedis
        = make_shared<AsyncConnection>(sourceAddress);
    sourceRedis->test();
    Campaigns campaigns;
    LoadCampaigns(sourceRedis, campaigns, acceptedDelta);

    cerr << "* Converting to accounts..." << endl;
    Accounts accounts;
    ConvertCampaignsToAccounts(campaigns, accounts);
    accounts.checkInvariants();

    std::vector<AccountKey> allKeys = accounts.getAccountKeys(AccountKey(), 1);

#if 0
    cerr << "* Dumping account summaries..." << endl;

    for (AccountKey & key: allKeys) {
        cerr << "key: " << key << ":" << endl;
        cerr << accounts.getAccountSummary(key) << endl;
    }
#elif 0
/*
key: transat_48_82:
toplevel b:4985552748USD/1M s:4824138480USD/1M i:0 a:4843285829USD/1M j:0
  transat_48_82_btpe_241 b:2048712718USD/1M s:2037451854USD/1M i:0 a:2037451854USD/1M j:0
    legacyImported b:2037451854USD/1M s:2037451854USD/1M i:0 a:0 j:0
  transat_48_82_elastic_price_240 b:803044669USD/1M s:803044585USD/1M i:0 a:803044585USD/1M j:0
    legacyImported b:803044585USD/1M s:803044585USD/1M i:0 a:0 j:0
  transat_48_82_probe2_263 b:351880852USD/1M s:350587957USD/1M i:0 a:350587957USD/1M j:0
    legacyImported b:350587957USD/1M s:350587957USD/1M i:0 a:0 j:0
  transat_48_82_probe_239 b:1639647590USD/1M s:1633054084USD/1M i:0 a:1633054084USD/1M j:0
    legacyImported b:1633054084USD/1M s:1633054084USD/1M i:0 a:0 j:0

 */
    ListOfStrings keys = {"transat_48_82", "transat_48_82_btpe_241",
                          "legacyImported"};
    AccountKey key;
    for (string & keyStr: keys) {
        key = key.childKey(keyStr);
        const Account & account = accounts.getAccount(key);
        cerr << "json dump of " << key.toString() << ":" << endl;
        cerr << account.toJson();
    }
#endif

    cerr << "* Storing converted accounts to Redis..." << endl;
    std::shared_ptr<AsyncConnection> targetRedis
        = make_shared<AsyncConnection>(targetAddress);
    targetRedis->test();
    StoreAccounts(targetRedis, accounts);
    cerr << "* Completed" << endl;
}

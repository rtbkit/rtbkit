/* redis_rollback.cc
   Wolfgang Sourdeau, 7 January 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
   
   Redis migration rollback class
 */

#include <jml/arch/futex.h>
#include "soa/service/redis.h"

#include "rtbkit/core/banker/account.h"
#include "rtbkit/core/banker/master_banker.h"

#include "redis_old_types.h"

#include "redis_rollback.h"

using namespace std;
using namespace Redis;

using namespace Datacratic;
using namespace RTBKIT;

namespace {

long long int
CurrencyPoolToLongLong(const CurrencyPool & pool)
{
    long long int result(0);

    for (const Amount & amount: pool.currencyAmounts) {
        if (amount.currencyCode != CurrencyCode::CC_USD) {
            throw ML::Exception("unhandled currency code: "
                                + Amount::getCurrencyStr(amount.currencyCode));
        }
        result += amount.value;
    }

    return result;
}

void
ConvertAccountsToCampaigns(const Accounts & accounts, Campaigns & campaigns)
{
    /* conversion steps:

       campaign:
         transferred = summary.allocated
         available =   summary.budget - transferred

       strategy:
         transferred = summary.budget
         spent =       summary.spent
         available =   transferred - spent
    */

    std::vector<AccountKey> keys = accounts.getAccountKeys();
    if (keys.size() > 0) {
#if 0 /* code to produce artificial expenses and see how they are converted
         back (requires non-const accounts) */
        for (AccountKey & key: accounts.getAccountKeys()) {
            if (key.size() == 2) {
                accounts.setBalance(key, accounts.getAvailable(key) + MicroUSD(1234), AT_NONE);
                AccountKey childKey = key.childKey("legacyImported");
                cerr << "child key: " << childKey << endl;
                if ((accounts.getAvailable(key) - MicroUSD(123)).isNonNegative()) {
                    accounts.setBalance(childKey, MicroUSD(123), AT_NONE);
                    accounts.addSpend(childKey, MicroUSD(12));
                }
            }
        }
#endif

        cerr << keys.size() << " accounts found (including subaccounts)" << endl;
        for (AccountKey & key: accounts.getAccountKeys()) {
            const string & campaignKey = key[0];
            const AccountSummary & summary = accounts.getAccountSummary(key);
            Campaign & campaign = campaigns[campaignKey];
            switch (key.size()) {
            case 1: { /* campaign */
                campaign.key_ = campaignKey;
                throw ML::Exception("this code is now obsolete due to missing"
                                    " members in AccountSummary");
#if 0
                campaign.transferred_
                    = CurrencyPoolToLongLong(summary.allocated);
                campaign.available_
                    = CurrencyPoolToLongLong(summary.budget) - campaign.transferred_;
#endif
                cerr << "- campaign 'campaigns:" + campaignKey + "' recreated"
                     << endl;
                break;
            }
            case 2: { /* strategy (budget) */
                const string & strategyKey = key[1];
                Strategy strategy(strategyKey, campaignKey);
                strategy.valid_ = true; /* we trust the consistency of
                                           accounts */
                strategy.transferred_ = CurrencyPoolToLongLong(summary.budget);
                strategy.spent_ = CurrencyPoolToLongLong(summary.spent);
                strategy.available_ = strategy.transferred_ - strategy.spent_;
                campaign.strategies.push_back(strategy);
                cerr << ("- strategy 'campaigns:"
                         + campaignKey + ":" + strategyKey
                         + "' recreated")
                     << endl;
                break;
            }
            default:
                break;
            }
        }
    }
    else {
        cerr << "No account to convert." << endl;
    }
}

void
StoreCampaigns(Redis::AsyncConnection & redis,
               Campaigns & campaigns)
{
    for (auto & it: campaigns) {
        const Campaign & campaign = it.second;
        if (campaign.validateAll(0)) {
            campaign.save(redis);
            for (const Strategy & strategy: campaign.strategies) {
                strategy.save(redis);
            }
        }
    }
}

}

namespace RTBKIT {

void
RedisRollback::
perform(const Redis::Address & sourceAddress,
        const Redis::Address & targetAddress)
{
    auto sourceRedis = make_shared<AsyncConnection>(sourceAddress);
    sourceRedis->test();

    std::shared_ptr<Accounts> accounts;
    int done(false);
    auto callback = [&] (std::shared_ptr<Accounts> newAccounts,
                         BankerPersistence::PersistenceCallbackStatus status,
                         const std::string & info) {
        if (status != BankerPersistence::PersistenceCallbackStatus::SUCCESS) {
            throw ML::Exception("error during account loading: " + info);
        }
        accounts = newAccounts;
        accounts->checkInvariants();
        done = true;
        ML::futex_wake(done);
    };
    cerr << "* Loading accounts..." << endl;
    RedisBankerPersistence storage(sourceRedis);
    storage.loadAll("", callback);
    while (!done) {
        ML::futex_wait(done, false);
    }

    cerr << "* Converting to campaigns/strategies..." << endl;
    Campaigns campaigns;
    ConvertAccountsToCampaigns(*accounts, campaigns);

    cerr << "* Storing converted accounts to Redis..." << endl;
    auto targetRedis = make_shared<AsyncConnection>(targetAddress);
    targetRedis->test();
    StoreCampaigns(*targetRedis, campaigns);
    cerr << "* Completed" << endl;
}

} // namespace RTBKIT

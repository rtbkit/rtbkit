/* local_banker.h
   Michael Burkat, 9 February 2015
   Copyright (c) 2015 Datacratic Inc.  All rights reserved.

   Local banker simple spend tracking and budget reauthorization.
*/

#pragma once

#include <string>
#include <mutex>
#include <unordered_set>

#include "banker.h"
#include "soa/service/service_base.h"
#include "soa/service/http_client.h"
#include "soa/service/message_loop.h"
#include "rtbkit/common/currency.h"
#include "rtbkit/common/account_key.h"

#include "go_account.h"

namespace RTBKIT {

struct LocalBanker : public Banker, Datacratic::MessageLoop, Datacratic::ServiceBase {
    
    LocalBanker(std::shared_ptr<Datacratic::ServiceProxies> services,
            GoAccountType type,
            const std::string & accountSuffix);
    
    void init(const std::string & bankerUrl, double timeout = 1.0, int numConnections = 128, bool tcpNoDelay = false);

    void setSpendRate(Amount spendRate);
    void setDebug(bool debugSetting);

    void start();
    void shutdown();

    virtual void
    addSpendAccount(const AccountKey & account,
                    CurrencyPool accountFloat,
                    std::function<void (std::exception_ptr, ShadowAccount&&)> onDone)
    {
        addAccount(account);
    }

    virtual bool
    authorizeBid(const AccountKey & account,
                 const std::string & item,
                 Amount amount)
    {
        return bid(account, amount);
    }

    virtual void
    cancelBid(const AccountKey & account,
              const std::string & item)
    {
    }

    virtual void
    winBid(const AccountKey & account,
           const std::string & item,
           Amount amountPaid,
           const LineItems & lineItems = LineItems())
    {
        win(account, amountPaid);
    }

    virtual void
    attachBid(const AccountKey & account,
              const std::string & item,
              Amount amountAuthorized)
    {
    }

    virtual Amount
    detachBid(const AccountKey & account,
              const std::string & item)
    {
        return MicroUSD(0);
    }

    virtual void
    commitBid(const AccountKey & account,
              const std::string & item,
              Amount amountPaid,
              const LineItems & lineItems)
    {
    }

    virtual void
    forceWinBid(const AccountKey & account,
                Amount amountPaid,
                const LineItems & lineItems)
    {
        win(account, amountPaid);
    }

    virtual void sync();

    virtual MonitorIndicator
    getProviderIndicators() const;

    GoAccounts accounts;

private:
    void addAccount(const AccountKey &account);

    void setRate(const AccountKey &key);

    void spendUpdate();

    void reauthorize();

    void sendBidCounts();

    bool bid(const AccountKey &key, Amount bidPrice);

    bool win(const AccountKey &key, Amount winPrice);

    GoAccountType type;
    std::string accountSuffix;
    std::string accountSuffixNoDot;
    std::shared_ptr<Datacratic::HttpClient> httpClient;
    std::mutex mutex;
    std::unordered_set<AccountKey> uninitializedAccounts;
    Amount spendRate;
    double syncRate;
    double reauthRate;
    double bidCountRate;
    std::atomic<bool> reauthorizeInProgress;
    std::atomic<int> reauthorizeSkipped;
    std::atomic<bool> spendUpdateInProgress;
    std::atomic<int> spendUpdateSkipped;
    std::atomic<bool> bidCountsInProgress;
    std::atomic<int> bidCountsSkipped;
    mutable std::mutex syncMtx;
    Datacratic::Date lastSync;
    Datacratic::Date lastReauth;
    bool debug;

    void addAccountImpl(const AccountKey &account);
    void replaceAccount(const AccountKey &account);
};

} // namespace RTBKIT


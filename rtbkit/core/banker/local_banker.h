/* local_banker.h
   Michael Burkat, 9 February 2015
   Copyright (c) 2015 Datacratic Inc.  All rights reserved.

   Local banker simple spend tracking and budget reauthorization.
*/

#pragma once

#include <string>
#include <mutex>
#include <unordered_set>

#include "soa/service/service_base.h"
#include "soa/service/http_client.h"
#include "soa/service/message_loop.h"
#include "rtbkit/common/currency.h"
#include "rtbkit/common/account_key.h"

#include "go_account.h"

namespace RTBKIT {

struct LocalBanker : public Datacratic::MessageLoop, Datacratic::ServiceBase {
    
    LocalBanker(std::shared_ptr<Datacratic::ServiceProxies> services,
            GoAccountType type,
            const std::string & accountSuffix);
    
    void init(const std::string & bankerUrl, double timeout = 1.0, int numConnections = 128, bool tcpNoDelay = false);
    void start();
    void shutdown();

    void addAccount(const AccountKey &account);

    void spendUpdate();

    void reauthorize();

    bool bid(const AccountKey &key, Amount bidPrice);

    bool win(const AccountKey &key, Amount winPrice);

    GoAccountType type;
    std::string accountSuffix;
    GoAccounts accounts;
    std::shared_ptr<Datacratic::HttpClient> httpClient;
    std::mutex mutex;
    std::unordered_set<AccountKey> uninitializedAccounts;

private:
    void addAccountImpl(const AccountKey &account);
};

} // namespace RTBKIT


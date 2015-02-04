#pragma once

#include <string>
#include <unordered_set>

#include "soa/service/http_client.h"
#include "soa/service/message_loop.h"
#include "rtbkit/common/currency.h"
#include "rtbkit/common/account_key.h"

#include "go_account.h"

namespace RTBKIT {

struct LocalBanker : public Datacratic::MessageLoop {
    
    LocalBanker(GoAccountType type, std::string accountSuffix);
    
    void init(std::string bankerUrl, double timeout = 1.0, int numConnections = 4, bool tcpNoDelay = false);
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
    std::unordered_set<AccountKey> uninitializedAccounts;
};

} // namespace RTBKIT


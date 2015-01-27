#pragma once

#include <string>

#include "soa/service/http_client.h"
#include "soa/service/message_loop.h"
#include "rtbkit/common/currency.h"
#include "rtbkit/common/account_key.h"

#include "go_account.h"

namespace RTBKIT {

struct LocalBanker : public Datacratic::MessageLoop {
    
    LocalBanker(GoAccountType type);
    
    void init(std::string bankerUrl, double timeout = 1.0, int numConnections = 4, bool tcpNoDelay = false);
    void start();
    void shutdown();

    void addAccount(AccountKey &account);

    void spendUpdate();

    void reauthorize();

    bool bid(AccountKey &key, Amount bidPrice);

    bool win(AccountKey &key, Amount winPrice);

    GoAccountType type;
    GoAccounts accounts;
    std::shared_ptr<Datacratic::HttpClient> httpClient;
};

} // namespace RTBKIT


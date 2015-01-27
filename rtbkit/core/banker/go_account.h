
#pragma once

#include <string>
#include <atomic>
#include <mutex>
#include <unordered_map>

#include "soa/jsoncpp/json.h"
#include "rtbkit/common/currency.h"
#include "rtbkit/common/account_key.h"

namespace RTBKIT {

struct GoBaseAccount {
    std::string name;
    std::string parent;

    GoBaseAccount(AccountKey &key);
    GoBaseAccount(Json::Value &jsonAccount);
    virtual void toJson(Json::Value &account);
};

struct GoRouterAccount : public GoBaseAccount {
    std::atomic<int64_t> rate;
    std::atomic<int64_t> balance;

    GoRouterAccount(AccountKey &key);
    GoRouterAccount(Json::Value &json);
    bool bid(Amount bidPrice);
    bool win(Amount winPrice) { return false; }
    void toJson(Json::Value &account);
};

struct GoPostAuctionAccount : public GoBaseAccount {
    std::atomic<int64_t> imp;
    std::atomic<int64_t> spend;

    GoPostAuctionAccount(AccountKey &key);
    GoPostAuctionAccount(Json::Value &jsonAccount);
    bool bid(Amount bidPrice) { return false; }
    bool win(Amount winPrice);
    void toJson(Json::Value &account);
};

enum GoAccountType {
    ROUTER,
    POST_AUCTION
};

struct GoAccount {
    GoAccountType type;
    std::shared_ptr<GoRouterAccount> router;
    std::shared_ptr<GoPostAuctionAccount> pal;

    GoAccount(){}
    GoAccount(AccountKey &key, GoAccountType type);
    GoAccount(Json::Value &jsonAccount);
    bool bid(Amount bidPrice);
    bool win(Amount winPrice);
    Json::Value toJson();
};

struct GoAccounts {
    std::mutex mutex;
    std::unordered_map<AccountKey, GoAccount> accounts;

    GoAccounts();
    void add(AccountKey&, GoAccountType type);
    void addFromJsonString(std::string json);
    void updateBalance(AccountKey &key, int64_t newBalance);
    bool bid(AccountKey &key, Amount bidPrice);
    bool win(AccountKey &key, Amount winPrice);
    GoAccount& get(AccountKey&);
    Json::Value toJson();
};

} // namespace RTBKIT

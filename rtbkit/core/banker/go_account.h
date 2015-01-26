
#pragma once

#include <string>
#include <atomic>
#include <unordered_map>

#include "soa/jsoncpp/json.h"
#include "rtbkit/common/currency.h"
#include "rtbkit/common/account_key.h"

namespace RTBKIT {

struct GoBaseAccount {
    std::string name;
    std::string parent;

    GoBaseAccount(AccountKey &key);
    virtual void toJson(Json::Value &account);
    virtual void fromJson(Json::Value &jsonAccount);
};

struct GoRouterAccount : public GoBaseAccount {
    std::atomic<int64_t> rate;
    std::atomic<int64_t> balance;

    GoRouterAccount(AccountKey &key);
    bool bid(Amount bidPrice);
    bool win(Amount winPrice) { return false; }
    void toJson(Json::Value &account);
    void fromJson(Json::Value &jsonAccount);
};

struct GoPostAuctionAccount : public GoBaseAccount {
    std::atomic<int64_t> imp;
    std::atomic<int64_t> spend;

    GoPostAuctionAccount(AccountKey &key);
    bool bid(Amount bidPrice) { return false; }
    bool win(Amount winPrice);
    void toJson(Json::Value &account);
    void fromJson(Json::Value &jsonAccount);
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
    bool bid(Amount bidPrice);
    bool win(Amount winPrice);
    Json::Value toJson();
    void fromJson(Json::Value &jsonAccount);
};

struct GoAccounts {
    std::unordered_map<AccountKey, GoAccount> accounts;

    GoAccounts();
    void add(AccountKey&, GoAccountType type);
    void addFromJson(std::string json);
    GoAccount& get(AccountKey&);
    Json::Value toJson();
};

} // namespace RTBKIT

/* go_account.h
   Michael Burkat, 9 February 2015
   Copyright (c) 2015 Datacratic Inc.  All rights reserved.

   Go account structures.
*/

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

    GoBaseAccount(const AccountKey &key);
    GoBaseAccount(Json::Value &jsonAccount);
    virtual void toJson(Json::Value &account);
};

struct GoRouterAccount : public GoBaseAccount {
    Amount rate;
    Amount balance;
    Amount maxBalance;
    Amount previousBalance;
    int bidsLastPeriod;

    GoRouterAccount(const AccountKey &key);
    GoRouterAccount(Json::Value &json);
    void setMaxBalance(const Amount & newMaxBalance);
    Amount updateBalance(const Amount & newBalance);
    Amount accumulateBalance(const Amount & newBalance);
    bool bid(Amount bidPrice);
    bool win(Amount winPrice) { return false; }
    void toJson(Json::Value &account);
};

struct GoPostAuctionAccount : public GoBaseAccount {
    std::atomic<int64_t> imp;
    Amount spend;

    GoPostAuctionAccount(const AccountKey &key);
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
    GoAccount(const AccountKey &key, GoAccountType type);
    GoAccount(Json::Value &jsonAccount);
    bool bid(Amount bidPrice);
    bool win(Amount winPrice);
    Json::Value toJson();
};

struct GoAccounts {
    std::mutex mutex;
    std::unordered_map<AccountKey, GoAccount> accounts;

    GoAccounts();
    void setMaxBalance(const AccountKey &key, const Amount & maxBalance);
    bool exists(const AccountKey& key);
    void add(const AccountKey&, GoAccountType type);
    bool addFromJsonString(std::string json);
    bool replaceFromJsonString(std::string json);
    Amount updateBalance(const AccountKey &key, const Amount & newBalance);
    Amount accumulateBalance(const AccountKey &key, const Amount & newBalance);
    Amount getBalance(const AccountKey &key);
    bool bid(const AccountKey &key, Amount bidPrice);
    bool win(const AccountKey &key, Amount winPrice);
    Json::Value toJson();

private:
    Amount MaxBalance;
    GoAccount* get(const AccountKey&);
};

} // namespace RTBKIT

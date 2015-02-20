/* go_account.cc
   Michael Burkat, 9 February 2015
   Copyright (c) 2015 Datacratic Inc.  All rights reserved.

   Go account implementation.
*/

#include "go_account.h"

using namespace std;

namespace RTBKIT {

// Go Account
GoAccount::GoAccount(const AccountKey &key, GoAccountType type)
    :type(type)
{
    switch (type) {
    case ROUTER:
        router = make_shared<GoRouterAccount>(key);
        break;
    case POST_AUCTION:
        pal = make_shared<GoPostAuctionAccount>(key);
        break;
    };
}

GoAccount::GoAccount(Json::Value &json)
{
    AccountKey key(json["name"].asString());
    string accType = json["type"].asString();

    if (accType == "Router") {
        type = ROUTER;
        router = make_shared<GoRouterAccount>(json);
    } else if (accType == "PostAuction") {
        type = POST_AUCTION;
        pal = make_shared<GoPostAuctionAccount>(json);
    } else {
        cout << "Go account type error." << endl;
    }
}

bool
GoAccount::bid(Amount bidPrice)
{
    if (type == ROUTER) return router->bid(bidPrice);
    else throw ML::Exception("GoAccounts::bid: attempt bid on non ROUTER account");
    return false;
}

bool
GoAccount::win(Amount winPrice)
{
    if (type == POST_AUCTION) return pal->win(winPrice);
    else throw ML::Exception("GoAccounts::updateBalance: attempt update on non ROUTER account");
    return false;
}

Json::Value
GoAccount::toJson()
{
    Json::Value account(Json::objectValue);
    switch (type) {
    case ROUTER:
        account["type"] = "Router";
        router->toJson(account);
        break;
    case POST_AUCTION:
        account["type"] = "PostAuction";
        pal->toJson(account);
        break;
    };
    return account;
}

// Router Account
GoRouterAccount::GoRouterAccount(const AccountKey &key)
    : GoBaseAccount(key)
{
    rate = 100;
    balance = 0;
}

GoRouterAccount::GoRouterAccount(Json::Value &json)
    : GoBaseAccount(json)
{
    if (json.isMember("rate")) rate = json["rate"].asInt();
    else rate = 0;

    if (json.isMember("balance")) balance = json["balance"].asInt();
    else balance = 0;
}

bool
GoRouterAccount::bid(Amount bidPrice)
{
    if (balance >= bidPrice.value) {
        balance -= bidPrice.value;
        return true;
    }
    return false;
}

void
GoRouterAccount::toJson(Json::Value &account)
{
    account["rate"] = int64_t(rate);
    account["balance"] = int64_t(balance);
    GoBaseAccount::toJson(account);
}

// Post Auction Account
GoPostAuctionAccount::GoPostAuctionAccount(const AccountKey &key)
    : GoBaseAccount(key)
{
    imp = 0;
    spend = 0;
}

GoPostAuctionAccount::GoPostAuctionAccount(Json::Value &json)
    : GoBaseAccount(json)
{
    if (json.isMember("imp")) imp = json["imp"].asInt();
    else imp = 0;

    if (json.isMember("spend")) spend = json["spend"].asInt();
    else spend = 0;
}

bool
GoPostAuctionAccount::win(Amount winPrice)
{
    spend += winPrice.value;
    imp += 1;
    return true;
}

void
GoPostAuctionAccount::toJson(Json::Value &account)
{
    account["imp"] = int64_t(imp);
    account["spend"] = int64_t(spend);
    GoBaseAccount::toJson(account);
}

//Account Base

GoBaseAccount::GoBaseAccount(const AccountKey &key)
    : name(key.toString()), parent(key.parent().toString())
{
}

GoBaseAccount::GoBaseAccount(Json::Value &json)
{
    if (json.isMember("name")) name = json["name"].asString();
    else name = ""; // TODO: throw error instead?

    if (json.isMember("parent")) parent = json["parent"].asString();
    else parent = "";
}

void
GoBaseAccount::toJson(Json::Value &account)
{
    account["name"] = name;
    account["parent"] = parent;
}

// Accounts

GoAccounts::GoAccounts() : accounts{}
{
}

void
GoAccounts::add(const AccountKey &key, GoAccountType type)
{
    if (exists(key)) return;
    std::lock_guard<std::mutex> guard(this->mutex);
    accounts.insert( pair<AccountKey, GoAccount>(key, GoAccount(key, type)) );
}

void
GoAccounts::addFromJsonString(std::string jsonAccount)
{
    Json::Value json = Json::parse(jsonAccount);
    if (json.isMember("type") && json.isMember("name")) {
        string name = json["name"].asString();
        const AccountKey key(name);
        if (exists(key)) return;

        std::lock_guard<std::mutex> guard(this->mutex);
        GoAccount account(json);
        accounts.insert( pair<AccountKey, GoAccount>(key, account) );
        //cout << "account in map: " << accounts[key].toJson() << endl;
    } else {
        cout << "error: type or name not parsed" << endl;
    }
}

void
GoAccounts::updateBalance(const AccountKey &key, int64_t newBalance)
{
    if (!exists(key)) return;
    std::lock_guard<std::mutex> guard(this->mutex);
    auto account = get(key);
    account->router->balance = newBalance;
}

int64_t
GoAccounts::getBalance(const AccountKey &key)
{
    if (!exists(key)) return 0;
    std::lock_guard<std::mutex> guard(this->mutex);
    auto account = get(key);
    return account->router->balance;
}

bool
GoAccounts::bid(const AccountKey &key, Amount bidPrice)
{
    if (!exists(key)) return false;

    std::lock_guard<std::mutex> guard(this->mutex);
    auto account = get(key);
    if (account->type != ROUTER) {
        throw ML::Exception("GoAccounts::bid: attempt bid on non ROUTER account");
    }

    return account->bid(bidPrice);
}

bool
GoAccounts::win(const AccountKey &key, Amount winPrice)
{
    if (!exists(key)) {
        cout << "account not found, unaccounted win: " << key.toString()
             << " " << winPrice.toString() << endl;
        return false;
    }

    std::lock_guard<std::mutex> guard(this->mutex);
    auto account = get(key);
    if (account->type != POST_AUCTION) {
        throw ML::Exception("GoAccounts::win: attempt win on non POST_AUCTION account");
    }

    return account->win(winPrice);
}

bool
GoAccounts::exists(const AccountKey &key)
{
    std::lock_guard<std::mutex> guard(this->mutex);
    bool exists = accounts.find(key) != accounts.end();
    return exists;
}

GoAccount*
GoAccounts::get(const AccountKey &key)
{
    auto account = accounts.find(key);
    if (account == accounts.end()) {
        throw ML::Exception("GoAccounts::get: account '" + key.toString() + "' not found");
    } else {
        return &account->second;
    }
}

} // namspace RTBKIT

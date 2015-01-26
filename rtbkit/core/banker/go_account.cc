
#include "go_account.h"

using namespace std;

namespace RTBKIT {

// Go Account
GoAccount::GoAccount(AccountKey &key, GoAccountType type)
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

bool
GoAccount::bid(Amount bidPrice)
{
    if (type == ROUTER) return router->bid(bidPrice);
    else return false;
}

bool
GoAccount::win(Amount winPrice)
{
    if (type == POST_AUCTION) return pal->win(winPrice);
    else return false;
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

void
GoAccount::fromJson(Json::Value &jsonAccount)
{
}

// Router Account
GoRouterAccount::GoRouterAccount(AccountKey &key)
    : GoBaseAccount(key)
{
    rate = 100;
    balance = 0;
}

bool
GoRouterAccount::bid(Amount bidPrice)
{
    return false;
}

void
GoRouterAccount::toJson(Json::Value &account)
{
    account["rate"] = int64_t(rate);
    account["balance"] = int64_t(balance);
    GoBaseAccount::toJson(account);
}

void
GoRouterAccount::fromJson(Json::Value &jsonAccount)
{

}

// Post Auction Account
GoPostAuctionAccount::GoPostAuctionAccount(AccountKey &key)
    : GoBaseAccount(key)
{
    imp = 0;
    spend = 0;
}

bool
GoPostAuctionAccount::win(Amount winPrice)
{
    return false;
}

void
GoPostAuctionAccount::toJson(Json::Value &account)
{
    account["imp"] = int64_t(imp);
    account["spend"] = int64_t(spend);
    GoBaseAccount::toJson(account);
}

void
GoPostAuctionAccount::fromJson(Json::Value &jsonAccount)
{


}

//Account Base

GoBaseAccount::GoBaseAccount(AccountKey &key)
    : name(key.toString()), parent(key.parent().toString())
{
}

void
GoBaseAccount::toJson(Json::Value &account)
{
    account["name"] = name;
    account["parent"] = parent;
}

void
GoBaseAccount::fromJson(Json::Value &jsonAccount)
{
}

// Accounts

GoAccounts::GoAccounts() : accounts{}
{
}

void
GoAccounts::add(AccountKey &key, GoAccountType type)
{
    auto account = accounts.find(key);
    if (account == accounts.end()) {
        return;
    }

    accounts[key] = GoAccount(key, type);
}

void
GoAccounts::addFromJson(std::string jsonAccount)
{
        Json::Value json = Json::parse(jsonAccount);
        cout << "j type: " << json.type() << endl;
        cout << "string: " << json.asString() << endl;
        Json::Value jAcc = Json::parse(json.asString());
        if (jAcc.isMember("type")) {
            cout << "type was parsed" << endl;
            cout << "type: " << jAcc["type"].asString() << endl;
        } else {
            cout << "type not parsed" << endl;
        }
        // TODO: CONTINUE HERE
}

GoAccount&
GoAccounts::get(AccountKey &key)
{
    auto account = accounts.find(key);
    if (account == accounts.end()) {
        throw ML::Exception("GoAccounts::get: attepted to get account that does not exist");
    } else {
        return account->second;
    }
}

} // namspace RTBKIT

/* local_banker.cc
   Michael Burkat, 9 February 2015
   Copyright (c) 2015 Datacratic Inc.  All rights reserved.

   Local banker implementation.
*/

#include "local_banker.h"
#include "soa/service/http_header.h"
#include "soa/types/date.h"

using namespace std;
using namespace Datacratic;

namespace RTBKIT {

LocalBanker::LocalBanker(shared_ptr<ServiceProxies> services, GoAccountType type, const string & accountSuffix)
        : ServiceBase("localBanker." + accountSuffix, services),
          type(type),
          accountSuffix(accountSuffix),
          accounts()
{
}

void
LocalBanker::init(const string & bankerUrl,
                  double timeout,
                  int numConnections,
                  bool tcpNoDelay)
{
    httpClient = std::make_shared<HttpClient>(bankerUrl, numConnections);
    httpClient->sendExpect100Continue(false);
    addSource("LocalBanker:HttpClient", httpClient);

    auto reauthorizePeriodic = [&] (uint64_t wakeups) {
        reauthorize();
    };
    auto spendUpdatePeriodic = [&] (uint64_t wakeups) {
        spendUpdate();
    };
    auto initializeAccountsPeriodic = [&] (uint64_t wakeups) {
        unordered_set<AccountKey> tempUninitialized;
        {
            std::lock_guard<std::mutex> guard(this->mutex);
            swap(uninitializedAccounts, tempUninitialized);
            this->recordLevel(accounts.accounts.size(), "accounts");
        }
        for (auto &key : tempUninitialized) {
            addAccountImpl(key);
        }
    };

    if (type == ROUTER)
        addPeriodic("localBanker::reauthorize", 1.0, reauthorizePeriodic);

    if (type == POST_AUCTION)
        addPeriodic("localBanker::spendUpdate", 0.5, spendUpdatePeriodic);

    addPeriodic("uninitializedAccounts", 1.0, initializeAccountsPeriodic);
}

void
LocalBanker::start()
{
    MessageLoop::start();
}

void
LocalBanker::shutdown()
{
    MessageLoop::shutdown();
}

void
LocalBanker::addAccount(const AccountKey &key)
{
    const AccountKey fullKey(key.toString() + ":" + accountSuffix);
    addAccountImpl(fullKey);
}

void
LocalBanker::addAccountImpl(const AccountKey &key)
{
    if (accounts.exists(key)) {
        std::lock_guard<std::mutex> guard(this->mutex);
        if (uninitializedAccounts.find(key) != uninitializedAccounts.end())
            uninitializedAccounts.erase(key);
        return;
    } else {
        std::lock_guard<std::mutex> guard(this->mutex);
        uninitializedAccounts.insert(key);
    }

    this->recordHit("addAccount.attempts");
    const Date sentTime = Date::now();

    auto onResponse = [&, key, sentTime] (const HttpRequest &req,
            HttpClientError error,
            int status,
            string && headers,
            string && body)
    {
        const Date recieveTime = Date::now();
        double latencyMs = recieveTime.secondsSince(sentTime);
        this->recordLevel(latencyMs, "addAccountLatencyMs");

        if (status != 200) {
            cout << "status: " << status << endl
                 << "error:  " << error << endl
                 << "body:   " << body << endl
                 << "url:    " << req.url_ << endl
                 << "cont_str: " << req.content_.str << endl;
            this->recordHit("addAccount.failure");
        } else {
            cout << body << endl;
            std::lock_guard<std::mutex> guard(this->mutex);
            accounts.addFromJsonString(body);
            if (uninitializedAccounts.find(key) != uninitializedAccounts.end())
                uninitializedAccounts.erase(key);
            this->recordHit("addAccount.success");
        }
    };
    auto const &cbs = make_shared<HttpClientSimpleCallbacks>(onResponse);
    Json::Value payload(Json::objectValue);
    payload["accountName"] = key.toString();
    switch (type) {
        case ROUTER:
            payload["accountType"] = "Router";
            break;
        case POST_AUCTION:
            payload["accountType"] = "PostAuction";
            break;
    };
    cout << "calling add go banker account: " << key.toString() << endl;
    httpClient->post("/accounts", cbs, payload, {}, {}, 1);
}

void
LocalBanker::spendUpdate()
{
    const Date sentTime = Date::now();
    this->recordHit("spendUpdate.attempt");

    auto onResponse = [&, sentTime] (const HttpRequest &req,
            HttpClientError error,
            int status,
            string && headers,
            string && body)
    {
        const Date recieveTime = Date::now();
        double latencyMs = recieveTime.secondsSince(sentTime);
        this->recordLevel(latencyMs, "spendUpdateLatencyMs");

        if (status != 200) {
            cout << "status: " << status << endl
                 << "error:  " << error << endl
                 << "body:   " << body << endl;
            this->recordHit("spendUpdate.failure");
        } else {
            this->recordHit("spendUpdate.success");
            //cout << "status: " << status << endl
            //     << "body:   " << body << endl;
        }
    };
    auto const &cbs = make_shared<HttpClientSimpleCallbacks>(onResponse);
    Json::Value payload(Json::arrayValue);
//    int i = 0;
    {
        std::lock_guard<std::mutex> guard(this->mutex);
        for (auto it : accounts.accounts) {
//         cout << "i: " << i++ << "\n"
//              << "name: " << it.first.toString() << "\n"
//              << "info: " << it.second.toJson() << endl;
            payload.append(it.second.toJson());
        }
    }
    httpClient->post("/spendupdate", cbs, payload, {}, {}, 0.5);
}

void
LocalBanker::reauthorize()
{
    const Date sentTime = Date::now();
    this->recordHit("reauthorize.attempt");

    auto onResponse = [&, sentTime] (const HttpRequest &req,
            HttpClientError error,
            int status,
            string && headers,
            string && body)
    {
        const Date recieveTime = Date::now();
        double latencyMs = recieveTime.secondsSince(sentTime);
        this->recordLevel(latencyMs, "reauthorizeLatencyMs");

        if (status != 200) {
            cout << "status: " << status << endl
                 << "error:  " << error << endl
                 << "body:   " << body << endl
                 << "url:    " << req.url_ << endl
                 << "cont_str: " << req.content_.str << endl;
            this->recordHit("reauthorize.failure");
        } else {
            Json::Value jsonAccounts = Json::parse(body);
            for ( auto jsonAccount : jsonAccounts ) {
                auto key = AccountKey(jsonAccount["name"].asString());
                int64_t newBalance = jsonAccount["balance"].asInt();
                //cout << "account: " << key.toString() << "\n"
                //     << "new bal: " << newBalance << endl;
                accounts.updateBalance(key, newBalance);
            }
            this->recordHit("reauthorize.success");
        }
    };

    auto const &cbs = make_shared<HttpClientSimpleCallbacks>(onResponse);
    Json::Value payload(Json::arrayValue);
//    int i = 0;
    {
        std::lock_guard<std::mutex> guard(this->mutex);
        for (auto it : accounts.accounts) {
//         cout << "i: " << i++ << "\n"
//              << "name: " << it.first.toString() << "\n"
//              << "info: " << it.second.toJson() << endl;
            payload.append(it.first.toString());
        }
    }
    httpClient->post("/reauthorize/1", cbs, payload, {}, {}, 2);
}

bool
LocalBanker::bid(const AccountKey &key, Amount bidPrice)
{
    AccountKey fullKey(key.toString() + ":" + accountSuffix);
    bool canBid = accounts.bid(fullKey, bidPrice);
    if (canBid)
        this->recordHit(fullKey.toString() + ".Bid");
    else
        this->recordHit(fullKey.toString() + ".noBid");
    return canBid;
}

bool
LocalBanker::win(const AccountKey &key, Amount winPrice)
{
    AccountKey fullKey(key.toString() + ":" + accountSuffix);
    bool canBid = accounts.win(fullKey, winPrice);
    if (canBid)
        this->recordHit(fullKey.toString() + ".Win");
    else
        this->recordHit(fullKey.toString() + ".noWin");
    return canBid;

}

} // namespace RTBKIT

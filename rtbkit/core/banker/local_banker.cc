/* local_banker.cc
   Michael Burkat, 9 February 2015
   Copyright (c) 2015 Datacratic Inc.  All rights reserved.

   Local banker implementation.
*/

#include <algorithm>
#include "local_banker.h"
#include "soa/service/http_header.h"
#include "soa/types/date.h"

using namespace std;
using namespace Datacratic;

namespace RTBKIT {

LocalBanker::LocalBanker(shared_ptr<ServiceProxies> services, GoAccountType type,
        const string & accountSuffix)
        : ServiceBase(accountSuffix + ".localBanker", services),
          type(type),
          accountSuffix(accountSuffix),
          accountSuffixNoDot(accountSuffix),
          accounts(),
          spendRate(MicroUSD(100000)),
          debug(false)
{
    replace(accountSuffixNoDot.begin(), accountSuffixNoDot.end(), '.', '_');
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
            this->recordCount(accounts.accounts.size(), "accounts");
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
LocalBanker::setSpendRate(Amount newSpendRate)
{
    spendRate = newSpendRate;
}

void
LocalBanker::setDebug(bool debugSetting)
{
    debug = debugSetting;
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
        double latencyMs = recieveTime.secondsSince(sentTime) * 1000;
        this->recordLevel(latencyMs, "addAccountLatencyMs");

        if (status != 200) {
            cout << "addAccount::" << endl
                 << "status: " << status << endl
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
        double latencyMs = recieveTime.secondsSince(sentTime) * 1000;
        this->recordLevel(latencyMs, "spendUpdateLatencyMs");

        if (status != 200) {
            cout << "spendUpdate::" << endl
                 << "status: " << status << endl
                 << "error:  " << error << endl
                 << "body:   " << body << endl;
            this->recordHit("spendUpdate.failure");
        } else {
            this->recordHit("spendUpdate.success");
        }
    };
    auto const &cbs = make_shared<HttpClientSimpleCallbacks>(onResponse);
    Json::Value payload(Json::arrayValue);
    {
        std::lock_guard<std::mutex> guard(this->mutex);
        for (auto it : accounts.accounts) {
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
        double latencyMs = recieveTime.secondsSince(sentTime) * 1000;
        this->recordLevel(latencyMs, "reauthorizeLatencyMs");

        if (status != 200) {
            cout << "reauthorize::" << endl
                 << "status: " << status << endl
                 << "error:  " << error << endl
                 << "body:   " << body << endl
                 << "url:    " << req.url_ << endl
                 << "cont_str: " << req.content_.str << endl;
            this->recordHit("reauthorize.failure");
        } else {
            Json::Value jsonAccounts = Json::parse(body);
            for ( auto jsonAccount : jsonAccounts ) {
                auto key = AccountKey(jsonAccount["name"].asString());
                Amount newBalance(MicroUSD(jsonAccount["balance"].asInt()));

                if (debug) {
                    string gKey = "account." + key.toString() + ":" + accountSuffixNoDot; 
                    recordLevel(accounts.getBalance(key.toString() + ":" + accountSuffix).value,
                            gKey + ".oldBalance");
                    recordLevel(newBalance.value,
                            gKey + ".newBalance");
                }

                accounts.updateBalance(key, newBalance);
                int64_t rate = jsonAccount["rate"].asInt();
                if (rate != spendRate.value) {
                    setRate(key);
                }
            }
            this->recordHit("reauthorize.success");
        }
    };

    auto const &cbs = make_shared<HttpClientSimpleCallbacks>(onResponse);
    Json::Value payload(Json::arrayValue);
    {
        std::lock_guard<std::mutex> guard(this->mutex);
        for (auto it : accounts.accounts) {
            payload.append(it.first.toString());
        }
    }
    httpClient->post("/reauthorize/1", cbs, payload, {}, {}, 1.0);
}

void
LocalBanker::setRate(const AccountKey &key)
{
    const Date sentTime = Date::now();
    this->recordHit("setRate.attempt");

    auto onResponse = [&, sentTime] (const HttpRequest &req,
            HttpClientError error,
            int status,
            string && headers,
            string && body)
    {
        const Date recieveTime = Date::now();
        double latencyMs = recieveTime.secondsSince(sentTime) * 1000;
        this->recordLevel(latencyMs, "setRateLatencyMs");

        if (status != 200) {
            cout << "setRate::" << endl
                 << "status: " << status << endl
                 << "error:  " << error << endl
                 << "body:   " << body << endl;
            this->recordHit("setRate.failure");
        } else {
            this->recordHit("setRate.success");
        }
    };
    auto const &cbs = make_shared<HttpClientSimpleCallbacks>(onResponse);
    Json::Value payload(Json::objectValue);
    payload["USD/1M"] = spendRate.value;
    httpClient->post("/accounts/" + key.toString() + "/rate", cbs, payload, {}, {}, 1.0);
}

bool
LocalBanker::bid(const AccountKey &key, Amount bidPrice)
{
    bool canBid = accounts.bid(key.toString() + ":" + accountSuffix, bidPrice);

    (canBid) ? recordHit("Bid") : recordHit("noBid");

    if (debug) {
        if (canBid)
            recordHit("account." + key.toString() + ":" + accountSuffixNoDot + ".Bid");
        else
            recordHit("account." + key.toString() + ":" + accountSuffixNoDot + ".noBid");
    }
    return canBid;
}

bool
LocalBanker::win(const AccountKey &key, Amount winPrice)
{
    bool winAccounted = accounts.win(key.toString() + ":" + accountSuffix, winPrice);

    (winAccounted) ? recordHit("Win") : recordHit("noWin");

    if (debug) {
        if (winAccounted)
            recordHit("account." + key.toString() + ":" + accountSuffixNoDot + ".Win");
        else
            recordHit("account." + key.toString() + ":" + accountSuffixNoDot + ".noWin");
    }
    return winAccounted;
}

} // namespace RTBKIT

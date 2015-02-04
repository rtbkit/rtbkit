
#include "local_banker.h"
#include "soa/service/http_header.h"
#include "jml/arch/timers.h"

using namespace std;
using namespace Datacratic;

namespace RTBKIT {

LocalBanker::LocalBanker(GoAccountType type, string accountSuffix)
    : type(type), accountSuffix(accountSuffix)
{
}

void
LocalBanker::init(std::string bankerUrl,
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
        for (auto key : uninitializedAccounts) {
            addAccount(key);
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
    auto onResponse = [&] (const HttpRequest &req,
            HttpClientError error,
            int status,
            string && headers,
            string && body)
    {
        if (status != 200) {
            cout << "status: " << status << endl
                 << "error:  " << error << endl
                 << "body:   " << body << endl
                 << "url:    " << req.url_ << endl
                 << "cont_str: " << req.content_.str << endl;
            uninitializedAccounts.insert(key);
        } else {
            cout << "returned account: " << endl;
            cout << body << endl;
            accounts.addFromJsonString(body);
            uninitializedAccounts.erase(key);
        }
    };
    auto const &cbs = make_shared<HttpClientSimpleCallbacks>(onResponse);
    Json::Value payload(Json::objectValue);
    payload["accountName"] = key.toString() + ":" + accountSuffix;
    switch (type) {
        case ROUTER:
            payload["accountType"] = "Router";
            break;
        case POST_AUCTION:
            payload["accountType"] = "PostAuction";
            break;
    };
    httpClient->post("/accounts", cbs, payload, {}, {}, 1);
}

void
LocalBanker::spendUpdate()
{
    auto onResponse = [] (const HttpRequest &req,
            HttpClientError error,
            int status,
            string && headers,
            string && body)
    {
        if (status != 200) {
            cout << "status: " << status << endl
                 << "error:  " << error << endl
                 << "body:   " << body << endl;
        } else {
            cout << "status: " << status << endl
                 << "body:   " << body << endl;
        }
    };
    auto const &cbs = make_shared<HttpClientSimpleCallbacks>(onResponse);
    Json::Value payload(Json::arrayValue);
//    int i = 0;
    for (auto it : accounts.accounts) {
//         cout << "i: " << i++ << "\n"
//              << "name: " << it.first.toString() << "\n"
//              << "info: " << it.second.toJson() << endl;
        payload.append(it.second.toJson());
    }
    httpClient->post("/spendupdate", cbs, payload, {}, {}, 1);
}

void
LocalBanker::reauthorize()
{
    auto onResponse = [&] (const HttpRequest &req,
            HttpClientError error,
            int status,
            string && headers,
            string && body)
    {
        if (status != 200) {
            cout << "status: " << status << endl
                 << "error:  " << error << endl
                 << "body:   " << body << endl
                 << "url:    " << req.url_ << endl
                 << "cont_str: " << req.content_.str << endl;
        } else {
            Json::Value jsonAccounts = Json::parse(body);
            for ( auto jsonAccount : jsonAccounts ) {
                auto key = AccountKey(jsonAccount["name"].asString());
                int64_t newBalance = jsonAccount["balance"].asInt();
                cout << "account: " << key.toString() << "\n"
                     << "new bal: " << newBalance << endl;
                accounts.updateBalance(key, newBalance);
            }
        }
    };

    auto const &cbs = make_shared<HttpClientSimpleCallbacks>(onResponse);
    Json::Value payload(Json::arrayValue);
//    int i = 0;
    for (auto it : accounts.accounts) {
//         cout << "i: " << i++ << "\n"
//              << "name: " << it.first.toString() << "\n"
//              << "info: " << it.second.toJson() << endl;
        payload.append(it.first.toString());
    }
    httpClient->post("/reauthorize/1", cbs, payload, {}, {}, 2);
}

bool
LocalBanker::bid(const AccountKey &key, Amount bidPrice)
{
    return accounts.bid(AccountKey(key.toString() + ":" + accountSuffix), bidPrice);
}

bool
LocalBanker::win(const AccountKey &key, Amount winPrice)
{
    return accounts.win(AccountKey(key.toString() + ":" + accountSuffix), winPrice);
}

} // namespace RTBKIT

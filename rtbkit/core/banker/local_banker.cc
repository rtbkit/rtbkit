
#include "local_banker.h"
#include "soa/service/http_header.h"

using namespace std;
using namespace Datacratic;

namespace RTBKIT {

LocalBanker::LocalBanker(GoAccountType type) : type(type)
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
LocalBanker::addAccount(AccountKey &key)
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
                 << "body:   " << body << endl;
        } else {
            cout << "returned account: " << endl;
            cout << body << endl;
            accounts.addFromJsonString(body);
        }
    };
    auto const &cbs = make_shared<HttpClientSimpleCallbacks>(onResponse);
    Json::Value payload(Json::objectValue);
    payload["name"] = key.toString();
    switch (type) {
        case ROUTER:
            payload["type"] = "Router";
            break;
        case POST_AUCTION:
            payload["type"] = "PostAuction";
            break;
    };
    httpClient->post("/account", cbs, payload, {}, {}, 1);
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
                 << "body:   " << body << endl;
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
    httpClient->post("/reauthorize/1", cbs, payload, {}, {}, 1);
}

bool
LocalBanker::bid(AccountKey &key, Amount bidPrice)
{
    return accounts.bid(key, bidPrice);
}

bool
LocalBanker::win(AccountKey &key, Amount winPrice)
{
    return accounts.win(key, winPrice);
}

} // namespace RTBKIT

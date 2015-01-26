
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
                << "error:  " << error << endl;
        } else {
            cout << "returned account: " << endl;
            cout << body << endl;
            accounts.addFromJson(body);
            // unparse json account and add it to accounts map
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

}
void
LocalBanker::reauthorize()
{

}

bool
LocalBanker::bid(AccountKey &key, Amount bidPrice)
{
    auto account = accounts.get(key);
    return account.bid(bidPrice);
}

bool
LocalBanker::win(AccountKey &key, Amount winPrice)
{
    auto account = accounts.get(key);
    return account.win(winPrice);
}

} // namespace RTBKIT

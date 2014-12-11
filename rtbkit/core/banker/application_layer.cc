/* application_layer.cc
   Mathieu Stefani, 11 June 2014
   Copyright (c) 2042 Datacratic Inc.  All rights reserved.

   application layers implementation.
*/

#include "application_layer.h"
#include <type_traits>

using namespace std;

namespace RTBKIT {

template<typename Result>
std::shared_ptr<HttpClientSimpleCallbacks>
makeCallback(std::string functionName,
             std::function<void (std::exception_ptr, Result &&)> onDone)
{
    //static_assert(std::is_default_constructible<Result>::value,
    //              "Result is not default constructible");

    return std::make_shared<HttpClientSimpleCallbacks>(
        [=](const HttpRequest &req,
            HttpClientError error, int statusCode,
            std::string &&, std::string &&body) {
        JML_TRACE_EXCEPTIONS(false);
        if (!onDone) {
            return;
        }

        if (error != HttpClientError::None) {
            std::ostringstream oss;
            oss << error;
            onDone(
                std::make_exception_ptr(
                    ML::Exception("HttpRequest('%s %s') failed in '%s': %s",
                                  req.verb_.c_str(), req.url_.c_str(), functionName.c_str(),
                                  oss.str().c_str())),
                   Result { });
        }
        else {
            decodeRestResponseJson<Result>(functionName, nullptr, statusCode, body, onDone);
        }
    });
}

void
ApplicationLayer::
topupTransfer(const AccountKey &account,
              AccountType accountType,
              CurrencyPool amount,
              const BudgetController::OnBudgetResult &onResult)
{
    topupTransfer(account.toString(), accountType, amount, onResult);
}

void
HttpLayer::
init(std::string bankerUri, double timeout, int activeConnections /* = 4 */, bool tcpNoDelay /* = false */)
{
    if (bankerUri.empty())
        throw ML::Exception("bankerUri can not be empty");

    if (bankerUri.compare(0, 7, "http://"))
        bankerUri = "http://" + bankerUri;

    this->timeout = timeout;

    httpClient.reset(new HttpClient(bankerUri, activeConnections));
    httpClient->sendExpect100Continue(false);
    httpClient->enableTcpNoDelay(tcpNoDelay);
    addSource("HttpLayer::httpClient", httpClient);
}

void
HttpLayer::
addAccount(const AccountKey &account,
           const BudgetController::OnBudgetResult &onResult)
{
    httpClient->post("/v1/accounts", budgetResultCallback(onResult),
                     { },
                     { { "accountName", account.toString() },
                         { "accountType", "budget" }
                     },
                     { }, /* headers */
                     timeout);
}


void
HttpLayer::
topupTransfer(const std::string &accountStr,
              AccountType accountType,
              CurrencyPool amount,
              const BudgetController::OnBudgetResult &onResult)
{
    httpClient->put("/v1/accounts/" + accountStr + "/balance",
                    budgetResultCallback(onResult),
                    amount.toJson(),
                    { { "accountType", AccountTypeToString(accountType) } },
                    { }, /* headers */
                    timeout);
}

void
HttpLayer::
setBudget(const std::string &topLevelAccount,
          CurrencyPool amount,
          const BudgetController::OnBudgetResult &onResult)
{
    httpClient->put("/v1/accounts/" + topLevelAccount + "/budget",
                    budgetResultCallback(onResult),
                    amount.toJson(),
                    { },
                    { },
                    timeout);
}

void
HttpLayer::
getAccountSummary(
    const AccountKey &account,
    int depth,
    std::function<void (std::exception_ptr, AccountSummary &&)>
    onResult)
{
    httpClient->get("/v1/accounts/" + account.toString() + "/summary",
                    makeCallback<AccountSummary>(
                        "HttpLayer::getAccountSummary",
                        onResult),
                    { { "depth", to_string(depth) } },
                    { } /* headers */,
                    timeout);
}

void
HttpLayer::
getAccount(const AccountKey &account,
           std::function<void (std::exception_ptr, Account &&)> onResult)
{
    httpClient->get("/v1/accounts/" + account.toString(),
                    makeCallback<Account>(
                        "HttpLayer::getAccount",
                        onResult),
                    { },
                    { },
                    timeout);
}

void
HttpLayer::
addSpendAccount(const std::string &shadowStr,
                std::function<void (std::exception_ptr, Account &&)> onDone)
{
    httpClient->post("/v1/accounts",
                     makeCallback<Account>("HttpLayer::addSpendAccount", onDone),
                     { },
                     {
                         { "accountName", shadowStr },
                         { "accountType", "spend" }
                     },
                     { }, /* headers */
                     timeout);
}

void
HttpLayer::
syncAccount(const ShadowAccount &account, const std::string &shadowStr,
            std::function<void (std::exception_ptr, Account &&)> onDone)
{

    httpClient->put("/v1/accounts/" + shadowStr + "/shadow",
                    makeCallback<Account>("HttpLayer::syncAcount", onDone),
                    account.toJson(),
                    { },
                    { },
                    timeout);
}

void
HttpLayer::
request(std::string method, const std::string &resource,
        const RestParams &params, const std::string &content, OnRequestResult onResult)
{
    std::transform(begin(method), end(method), begin(method), [](char c) { return ::tolower(c); });

    auto onDone = std::make_shared<HttpClientSimpleCallbacks>(
        [=](const HttpRequest &req,
            HttpClientError error, int statusCode,
            std::string &&, std::string &&body) {
        if (error != HttpClientError::None) {
            std::ostringstream oss;
            oss << error;
            onResult(std::make_exception_ptr(
                         ML::Exception("HttpRequest('%s %s') failed with return code %s",
                                       req.verb_.c_str(), req.url_.c_str(), oss.str().c_str())),
                     statusCode, "");
        }
        else {
            onResult(nullptr, statusCode, body);
        }
    });

    if (method == "post") {
        httpClient->post(resource, onDone, content, params, { }, timeout);
    }
    else if (method == "put") {
        httpClient->put(resource, onDone, content, params, { }, timeout);
    }
    else if (method == "get") {
        httpClient->get(resource, onDone, params, { }, timeout);
    }
    else {
        throw ML::Exception("Unknown method '%s'", method.c_str());
    }
}

std::shared_ptr<HttpClientSimpleCallbacks>
HttpLayer::
budgetResultCallback(const BudgetController::OnBudgetResult &onResult)
{
    return std::make_shared<HttpClientSimpleCallbacks>(
        [=](const HttpRequest &req,
            HttpClientError error, int statusCode,
            std::string &&, std::string &&body) {
        if (error != HttpClientError::None) {
            std::ostringstream oss;
            oss << error;
            onResult(std::make_exception_ptr(
                         ML::Exception("HttpRequest('%s %s') failed with return code %s",
                                      req.verb_.c_str(), req.url_.c_str(), oss.str().c_str())));
        }
        else {
            onResult(nullptr);
        }
    });
}

void
ZmqLayer::
init(const std::shared_ptr<ServiceProxies> &services,
     const std::string &bankerServiceName)
{
    if (bankerServiceName.empty())
        throw ML::Exception("bankerServiceName can not be empty");

    proxy.reset(new RestProxy(services->zmqContext));
    proxy->initServiceClass(services->config, bankerServiceName, "zeromq", false /* local */);
    addSource("ZmqLayer::proxy", proxy);
}

void
ZmqLayer::
addAccount(const AccountKey &account,
           const BudgetController::OnBudgetResult &onResult)
{
    proxy->push(budgetResultCallback(onResult),
                "POST", "/v1/accounts",
                { {"accountName", account.toString()},
                    { "accountType", "budget" } });
}


void
ZmqLayer::
topupTransfer(const std::string &accountStr,
              AccountType accountType,
              CurrencyPool amount,
              const BudgetController::OnBudgetResult &onResult)
{
    proxy->push(budgetResultCallback(onResult),
                "PUT", "/v1/accounts/" + accountStr + "/balance",
                { { "accountType", AccountTypeToString(accountType) } },
                amount.toJson().toString());
}

void
ZmqLayer::
setBudget(const std::string &topLevelAccount,
          CurrencyPool amount,
          const BudgetController::OnBudgetResult &onResult)
{
    proxy->push(budgetResultCallback(onResult),
                "PUT", "/v1/accounts/" + topLevelAccount + "/budget",
                { /* {"amount", amount.toString()}*/ },
                amount.toJson().toString());
}

void
ZmqLayer::
getAccountSummary(
    const AccountKey &account,
    int depth,
    std::function<void (std::exception_ptr, AccountSummary &&)>
    onResult)
{
    proxy->push(makeRestResponseJsonDecoder<AccountSummary>("ZmqLayer::getAccountSummary", onResult),
                "GET", "/v1/accounts/" + account.toString() + "/summary",
                { {"depth", to_string(depth)} },
                "");
}

void
ZmqLayer::
getAccount(const AccountKey &account,
           std::function<void (std::exception_ptr, Account &&)> onResult)
{
    proxy->push(makeRestResponseJsonDecoder<Account>("ZmqLayer::getAccount", onResult),
                "GET", "/v1/accounts/" + account.toString());
}

void
ZmqLayer::
addSpendAccount(const std::string &shadowStr,
                std::function<void (std::exception_ptr, Account &&)> onDone)
{
    proxy->push(makeRestResponseJsonDecoder<Account>("ZmqLayer::addSpendAccount", onDone),
                "POST",
                "/v1/accounts",
                {
                    { "accountName", shadowStr },
                    { "accountType", "spend" }
                },
                "");
}

void
ZmqLayer::
syncAccount(const ShadowAccount &account, const std::string &shadowStr,
            std::function<void (std::exception_ptr, Account &&)> onDone)
{
    proxy->push(makeRestResponseJsonDecoder<Account>("ZmqLayer::syncAccount", onDone),
                "PUT",
                "/v1/accounts/" + shadowStr + "/shadow",
                {},
                account.toJson().toString());
}

void
ZmqLayer::
request(std::string method, const std::string &resource,
        const RestParams &params, const std::string &content, OnRequestResult onResult)
{
    proxy->push(onResult, method, resource, params, content);
}


RestProxy::OnDone
ZmqLayer::
budgetResultCallback(const BudgetController::OnBudgetResult &onResult)
{
    return [=] (std::exception_ptr ptr, int resultCode, string body) {
        //cerr << "got budget result callback with resultCode "
        //     << resultCode << " body " << body << endl;
        onResult(ptr);
    };
}

} // namespace RTBKIT

/* application_layer.h                                                  -*- C++ -*-
   Mathieu Stefani, 11 June 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

   The ApplicationLayer is an abstraction layer that defines a communication protocol
   for the master banker.

   Currently, the SlaveBanker can talk to the MasterBanker eiter via HTTP through
   the HttpLayer or zeromq via the ZmqLayer

   Note that the ZmqLayer will discover the MasterBanker zmq endpoint via the
   ConfigurationService (most of the time ZooKeeper).
   The HttpLayer does not require any discovery    
*/

#pragma once

#include "banker.h"
#include "soa/service/message_loop.h"
#include "soa/service/rest_proxy.h"
#include "soa/service/http_client.h"

namespace RTBKIT {

/*****************************************************************************/
/* APPLICATION LAYER                                                         */
/*****************************************************************************/

struct ApplicationLayer : public MessageLoop {
    typedef std::function<void (std::exception_ptr, int statusCode, 
                                const std::string &payload)> OnRequestResult;

    /* BUDGET CONTROLLER */
    virtual void addAccount(
                    const AccountKey &account,
                    const BudgetController::OnBudgetResult &onResult) = 0;

    virtual void topupTransfer(
                    const AccountKey &account,
                    AccountType accountType,
                    CurrencyPool amount,
                    const BudgetController::OnBudgetResult &onResult);

    virtual void topupTransfer(
                    const std::string &accountStr,
                    AccountType accountType,
                    CurrencyPool amount,
                    const BudgetController::OnBudgetResult &onResult) = 0;

    virtual void setBudget(const std::string &topLevelAccount,
                           CurrencyPool amount,
                           const BudgetController::OnBudgetResult &onResult) = 0;

    virtual void getAccountSummary(
                           const AccountKey &account,
                           int depth,
                           std::function<void (std::exception_ptr, AccountSummary &&)>
                           onResult) = 0;

    virtual void getAccount(const AccountKey &account,
                            std::function<void (std::exception_ptr, Account &&)> onResult) = 0;

    /* BANKER */
    virtual void addSpendAccount(
                    const std::string &shadowStr,
                    std::function<void (std::exception_ptr, Account&&)> onDone) = 0;

    virtual void syncAccount(
                    const ShadowAccount & account, const std::string &shadowStr,
                    std::function<void (std::exception_ptr,
                                         Account &&)> onDone) = 0;

    /* CUSTOM */
    virtual void request(
                       std::string method, const std::string &resource,
                       const RestParams &params,
                       const std::string &content,
                       OnRequestResult onResult) = 0;
};

/*****************************************************************************/
/* HTTP LAYER                                                                */
/*****************************************************************************/

struct HttpLayer : public ApplicationLayer {

    void init(std::string bankerUri, double timeout = 1.0, int activeConnections = 4, bool tcpNoDelay = false);

    void addAccount(
                    const AccountKey &account,
                    const BudgetController::OnBudgetResult &onResult);

    using ApplicationLayer::topupTransfer;
    void topupTransfer(
                    const std::string &accountStr,
                    AccountType accountType,
                    CurrencyPool amount,
                    const BudgetController::OnBudgetResult &onResult);

    void setBudget(
                   const std::string &topLevelAccount,
                   CurrencyPool amount,
                   const BudgetController::OnBudgetResult &onResult);

    void getAccountSummary(
                   const AccountKey &account,
                   int depth,
                   std::function<void (std::exception_ptr, AccountSummary &&)>
                   onResult);

    void getAccount(
                   const AccountKey &account,
                   std::function<void (std::exception_ptr, Account &&)> onResult);

    void addSpendAccount(const std::string &shadowStr,
                         std::function<void (std::exception_ptr, Account&&)> onDone);

    void syncAccount(const ShadowAccount &account, const std::string &shadowStr,
                     std::function<void (std::exception_ptr,
                                   Account &&)> onDone);

    void request(std::string method, const std::string &resource,
               const RestParams &params,
               const std::string &content,
               OnRequestResult onResult);
private:
    std::shared_ptr<HttpClient> httpClient;
    double timeout;

    static std::shared_ptr<HttpClientSimpleCallbacks>
    budgetResultCallback(const BudgetController::OnBudgetResult & onResult);

};

/*****************************************************************************/
/* ZEROMQ LAYER                                                              */
/*****************************************************************************/

struct ZmqLayer : public ApplicationLayer {
    void init(const std::shared_ptr<ServiceProxies> &services,
              const std::string &bankerServiceName = "rtbBanker");

    void addAccount(
                    const AccountKey &account,
                    const BudgetController::OnBudgetResult &onResult);

    using ApplicationLayer::topupTransfer;
    void topupTransfer(
                    const std::string &accountStr,
                    AccountType accountType,
                    CurrencyPool amount,
                    const BudgetController::OnBudgetResult &onResult);

    void setBudget(
                   const std::string &topLevelAccount,
                   CurrencyPool amount,
                   const BudgetController::OnBudgetResult &onResult);

    void getAccountSummary(
                   const AccountKey &account,
                   int depth,
                   std::function<void (std::exception_ptr, AccountSummary &&)>
                   onResult);

    void getAccount(
                   const AccountKey &account,
                   std::function<void (std::exception_ptr, Account &&)> onResult);

    void addSpendAccount(const std::string &shadowStr,
                         std::function<void (std::exception_ptr, Account&&)> onDone);

    void syncAccount(const ShadowAccount & account, const std::string &shadowStr,
                     std::function<void (std::exception_ptr,
                                   Account &&)> onDone);
    void request(std::string method, const std::string &resource,
               const RestParams &params,
               const std::string &content,
               OnRequestResult onResult);
private:
    std::shared_ptr<RestProxy> proxy;

    static RestProxy::OnDone
    budgetResultCallback(const BudgetController::OnBudgetResult & onResult);
};

template<typename Layer, typename... Args>
std::shared_ptr<Layer> make_application_layer(Args&& ...args)
{
    static_assert(std::is_base_of<ApplicationLayer, Layer>::value,
                  "The Layer must be a subclass of ApplicationLayer");
    auto layer = std::make_shared<Layer>();
    layer->init(std::forward<Args>(args)...);
    return layer;
}

} // namespace RTBKIT

/* slave_banker.cc
   Jeremy Barnes, 8 November 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Slave banker implementation.
*/

#include "slave_banker.h"
#include "soa/service/http_header.h"
#include "jml/utils/vector_utils.h"
#include "soa/service/rest_proxy.h"
#include <type_traits>

using namespace std;

static constexpr int MaximumFailSyncSeconds = 3;

namespace RTBKIT {

template<typename Result>
std::shared_ptr<HttpClientSimpleCallbacks>
makeCallback(std::string functionName,
                  std::function<void (std::exception_ptr, Result &&)> onDone)
{
    //static_assert(std::is_default_constructible<Result>::value,
    //              "Result is not default constructible");

    return std::make_shared<HttpClientSimpleCallbacks>(
        [=](const HttpRequest &,
            HttpClientError error, int statusCode,
            std::string &&, std::string &&body)
    {
        JML_TRACE_EXCEPTIONS(false);
        if (!onDone) {
            return;
        }

        if (error != HttpClientError::NONE) {
            std::ostringstream oss;
            oss << error;
            onDone(std::make_exception_ptr(ML::Exception("HTTP Request failed in '%s': %s",
                                                         functionName.c_str(), oss.str().c_str())),
                   Result { });
        }
        else {
           decodeRestResponseJson<Result>(functionName, nullptr, statusCode, body, onDone);
        }
    });
}


/*****************************************************************************/
/* SLAVE BUDGET CONTROLLER                                                   */
/*****************************************************************************/

SlaveBudgetController::
SlaveBudgetController()
{
}

void
SlaveBudgetController::
addAccount(const AccountKey & account,
           const OnBudgetResult & onResult)
{
    httpClient->post("/v1/accounts", budgetResultCallback(onResult),
                    { },
                    { { "accountName", account.toString() },
                      { "accountType", "budget" }});
}


void
SlaveBudgetController::
topupTransfer(const AccountKey & account,
              CurrencyPool amount,
              const OnBudgetResult & onResult)
{
    httpClient->put("/v1/accounts/" + account.toString() + "/balance",
                    budgetResultCallback(onResult),
                    amount.toJson(),
                    { { "accountType", "budget" } });
}

void
SlaveBudgetController::
setBudget(const std::string & topLevelAccount,
          CurrencyPool amount,
          const OnBudgetResult & onResult)
{
    httpClient->put("/v1/accounts/" + topLevelAccount + "/budget",
                    budgetResultCallback(onResult),
                    amount.toJson());
}

void
SlaveBudgetController::
addBudget(const std::string & topLevelAccount,
          CurrencyPool amount,
          const OnBudgetResult & onResult)
{
    throw ML::Exception("addBudget no good any more");
}

void
SlaveBudgetController::
getAccountList(const AccountKey & account,
               int depth,
               std::function<void (std::exception_ptr,
                                   std::vector<AccountKey> &&)>)
{
    throw ML::Exception("getAccountList not needed anymore");
}

void
SlaveBudgetController::
getAccountSummary(const AccountKey & account,
                  int depth,
                  std::function<void (std::exception_ptr,
                                      AccountSummary &&)> onResult)
{
    httpClient->get("/v1/accounts/" + account.toString() + "/summary",
                    makeCallback<AccountSummary>(
                       "SlaveBudgetController::getAccountSummary",
                        onResult),
                    { { "depth", to_string(depth) } });
}

void
SlaveBudgetController::
getAccount(const AccountKey & accountKey,
           std::function<void (std::exception_ptr,
                               Account &&)> onResult)
{
    httpClient->get("/v1/accounts/" + accountKey.toString(),
                    makeCallback<Account>(
                    "SlaveBudgetController::getAccount",
                    onResult));
}

std::shared_ptr<HttpClientSimpleCallbacks>
SlaveBudgetController::
budgetResultCallback(const OnBudgetResult & onResult)
{
    return std::make_shared<HttpClientSimpleCallbacks>(
        [=](const HttpRequest &,
            HttpClientError error, int statusCode,
            std::string &&, std::string &&body)
    {
        if (error != HttpClientError::NONE) {
            std::ostringstream oss;
            oss << error;
            onResult(std::make_exception_ptr(
                ML::Exception("HTTP Request failed with return code %s", oss.str().c_str())));
        }
        else {
            onResult(nullptr);
        }
    });
}


/*****************************************************************************/
/* SLAVE BANKER                                                              */
/*****************************************************************************/

SlaveBanker::
SlaveBanker(std::shared_ptr<zmq::context_t> context)
    : createdAccounts(128)
{
}

SlaveBanker::
SlaveBanker(std::shared_ptr<zmq::context_t> context,
            std::shared_ptr<ConfigurationService> config,
            const std::string & accountSuffix,
            const std::string & bankerHost)
    : createdAccounts(128)
{
    init(config, accountSuffix, bankerHost);
}

void
SlaveBanker::
init(std::shared_ptr<ConfigurationService> config,
     const std::string & accountSuffix,
     const std::string & bankerHost)
{
    if (accountSuffix.empty()) {
        throw ML::Exception("'accountSuffix' cannot be empty");
    }
    if (bankerHost.empty()) {
        throw ML::Exception("'bankerHost' cannot be empty");
    }

    // When our account manager creates an account, it will call this
    // function.  We can't do anything from it (because the lock could
    // be held), but we *can* push a message asynchronously to be
    // handled later...
    accounts.onNewAccount = [=] (const AccountKey & accountKey)
        {
            //cerr << "((((1)))) new account " << accountKey << endl;
            createdAccounts.push(accountKey);
        };

    // ... here.  Now we know that no lock is held and so we can
    // perform the work we need to synchronize the account with
    // the server.
    createdAccounts.onEvent = [=] (const AccountKey & accountKey)
        {
            //cerr << "((((2)))) new account " << accountKey << endl;

            auto onDone = [=] (std::exception_ptr exc,
                               ShadowAccount && account)
            {
#if 0
                cerr << "((((3)))) new account " << accountKey << endl;

                cerr << "got back shadow account " << account
                << " for " << accountKey << endl;

                cerr << "current status is " << accounts.getAccount(accountKey)
                << endl;
#endif
            };

            addSpendAccount(accountKey, USD(0), onDone);
        };

    // Since we send one HttpRequest per account when syncing, this is a good idea
    // to keep a fairly large queue size in order to avoid deadlocks
    httpClient.reset(new HttpClient(bankerHost, 4 /* numParallel */,
                                                1024 /* queueSize */));
    addSource("SlaveBanker::httpClient", httpClient);

    addSource("SlaveBanker::createdAccounts", createdAccounts);

    this->accountSuffix = accountSuffix;
    
    addPeriodic("SlaveBanker::reportSpend", 1.0,
                std::bind(&SlaveBanker::reportSpend,
                          this,
                          std::placeholders::_1),
                true /* single threaded */);
    addPeriodic("SlaveBanker::reauthorizeBudget", 1.0,
                std::bind(&SlaveBanker::reauthorizeBudget,
                          this,
                          std::placeholders::_1),
                true /* single threaded */);
}

ShadowAccount
SlaveBanker::
syncAccountSync(const AccountKey & account)
{
    BankerSyncResult<ShadowAccount> result;
    syncAccount(account, result);
    return result.get();
}

void
SlaveBanker::
onSyncResult(const AccountKey & accountKey,
               std::function<void (std::exception_ptr,
                                   ShadowAccount &&)> onDone,
               std::exception_ptr exc,
               Account&& masterAccount)
{
    ShadowAccount result;

    try {
        if (exc) {
            onDone(exc, std::move(result));
            return;
        }

        //cerr << "got result from master for " << accountKey
        //     << " which is "
        //     << masterAccount << endl;
        
        result = accounts.syncFromMaster(accountKey, masterAccount);
    } catch (...) {
        onDone(std::current_exception(), std::move(result));
    }

    try {
        onDone(nullptr, std::move(result));
    } catch (...) {
        cerr << "warning: onDone handler threw" << endl;
    }
}

void
SlaveBanker::
onInitializeResult(const AccountKey & accountKey,
                   std::function<void (std::exception_ptr,
                                       ShadowAccount &&)> onDone,
                   std::exception_ptr exc,
                   Account&& masterAccount)
{
    ShadowAccount result;

    try {
        if (exc) {
            onDone(exc, std::move(result));
            return;
        }

        result = accounts.initializeAndMergeState(accountKey, masterAccount);
    } catch (...) {
        onDone(std::current_exception(), std::move(result));
    }
    
    try {
        onDone(nullptr, std::move(result));
    } catch (...) {
        cerr << "warning: onDone handler threw" << endl;
    }
}


void
SlaveBanker::
syncAccount(const AccountKey & accountKey,
            std::function<void (std::exception_ptr,
                                ShadowAccount &&)> onDone)
{
    auto onDone2
        = std::bind(&SlaveBanker::onSyncResult,
                    this,
                    accountKey,
                    onDone,
                    std::placeholders::_1,
                    std::placeholders::_2);

    //cerr << "syncing account " << accountKey << ": "
    //     << accounts.getAccount(accountKey) << endl;

    httpClient->put("/v1/accounts/" + getShadowAccountStr(accountKey) + "/shadow",
                    makeCallback<Account>("SlaverBanker::syncAcount", onDone2),
                    accounts.getAccount(accountKey).toJson());
}

void
SlaveBanker::
syncAllSync()
{
    BankerSyncResult<void> result;
    syncAll(result);
    result.get();
}

void
SlaveBanker::
syncAll(std::function<void (std::exception_ptr)> onDone)
{
    auto allKeys = accounts.getAccountKeys();

    vector<AccountKey> filteredKeys;
    for (auto k: allKeys)
    	if (accounts.isInitialized(k))
    		filteredKeys.push_back(k);

    allKeys.swap(filteredKeys);

    if (allKeys.empty()) {
        // We need some kind of synchronization here because the lastSync
        // member variable will also be read in the context of an other
        // MessageLoop (the MonitorProviderClient). Thus, if we want to avoid
        // data-race here, we grab a lock.
        std::lock_guard<Lock> guard(syncLock);
        lastSync = Date::now();
        if (onDone)
            onDone(nullptr);
        return;
    }

    struct Aggregator {

        Aggregator(SlaveBanker *self, int numTotal,
                   std::function<void (std::exception_ptr)> onDone)
            : itl(new Itl())
        {
            itl->self = self;
            itl->numTotal = numTotal;
            itl->numFinished = 0;
            itl->exc = nullptr;
            itl->onDone = onDone;
        }

        struct Itl {
            SlaveBanker *self;
            int numTotal;
            int numFinished;
            std::exception_ptr exc;
            std::function<void (std::exception_ptr)> onDone;
        };

        std::shared_ptr<Itl> itl;
        
        void operator () (std::exception_ptr exc, ShadowAccount && account)
        {
            if (exc)
                itl->exc = exc;
            int nowDone = __sync_add_and_fetch(&itl->numFinished, 1);
            if (nowDone == itl->numTotal) {
                if (!itl->exc) {
                    std::lock_guard<Lock> guard(itl->self->syncLock);
                    itl->self->lastSync = Date::now();
                }

                if (itl->onDone)
                    itl->onDone(itl->exc);
                else {
                    if (itl->exc)
                        cerr << "warning: async callback aggregator ate "
                             << "exception" << endl;
                }
            }
        }               
    };
    
    Aggregator aggregator(const_cast<SlaveBanker *>(this), allKeys.size(), onDone);

    //cerr << "syncing " << allKeys.size() << " keys" << endl;

    for (auto & key: allKeys) {
        // We take its parent since syncAccount assumes nothing was added
        if (accounts.isInitialized(key))
            syncAccount(key, aggregator);
    }
}

void
SlaveBanker::
addSpendAccount(const AccountKey & accountKey,
                CurrencyPool accountFloat,
                std::function<void (std::exception_ptr, ShadowAccount&&)> onDone)
{
    bool first = accounts.createAccountAtomic(accountKey);
    if(!first) {
        // already done
        if (onDone) {
            auto account = accounts.getAccount(accountKey);
            onDone(nullptr, std::move(account));
        }
    }
    else {
        // TODO: record float
        //accountFloats[accountKey] = accountFloat;

        // Now kick off the initial synchronization step
        auto onDone2 = std::bind(&SlaveBanker::onInitializeResult,
                                 this,
                                 accountKey,
                                 onDone,
                                 std::placeholders::_1,
                                 std::placeholders::_2);

        cerr << "********* calling addSpendAccount for " << accountKey
             << " for SlaveBanker " << accountSuffix << endl;

        httpClient->post("/v1/accounts",
                         makeCallback<Account>("SlaveBanker::addSpendAccount", onDone2),
                         { },
                         {
                            { "accountName", getShadowAccountStr(accountKey) },
                            { "accountType", "spend" }
                         });

    }
}

void
SlaveBanker::
reportSpend(uint64_t numTimeoutsExpired)
{
    if (numTimeoutsExpired > 1) {
        cerr << "warning: slave banker missed " << numTimeoutsExpired
             << " timeouts" << endl;
    }

    if (reportSpendSent != Date())
        cerr << "warning: report spend still in progress" << endl;

    //cerr << "started report spend" << endl;

    auto onDone = [=] (std::exception_ptr exc)
        {
            //cerr << "finished report spend" << endl;
            reportSpendSent = Date();
            if (exc)
                cerr << "reportSpend got exception" << endl;
        };
    
    syncAll(onDone);
}

void
SlaveBanker::
reauthorizeBudget(uint64_t numTimeoutsExpired)
{
    if (numTimeoutsExpired > 1) {
        cerr << "warning: slave banker missed " << numTimeoutsExpired
             << " timeouts" << endl;
    }

    //std::unique_lock<Lock> guard(lock);
    if (reauthorizeBudgetSent != Date()) {
        cerr << "warning: reauthorize budget still in progress" << endl;
    }

    int numDone = 0;

    // For each of our accounts, we report back what has been spent
    // and re-up to our desired float
    auto onAccount = [&] (const AccountKey & key,
                          const ShadowAccount & account)
        {
            Json::Value payload = CurrencyPool(USD(0.10)).toJson();
            ++numDone;

            // Finally, send it out
            httpClient->post("/v1/accounts/" + getShadowAccountStr(key) + "/balance",
                            std::make_shared<HttpClientSimpleCallbacks>(
                            [=](const HttpRequest &, HttpClientError error,
                                int statusCode,
                                std::string &&, std::string &&body)
                            {
                                if (error != HttpClientError::NONE) {
                                    std::ostringstream oss;
                                    oss << error;
                                    onReauthorizeBudgetMessage(
                                        key,
                                        std::make_exception_ptr(ML::Exception(
                                            "HTTP Request failed in 'reauthorizeBudget': %s",
                                             oss.str().c_str())),
                                        statusCode,
                                        body);
                                }
                                else {
                                    onReauthorizeBudgetMessage(key, nullptr, statusCode,
                                                               body);
                                }
                            }),
                            payload,
                            { { "accountType", "spend" } });

        };

    accounts.forEachInitializedAccount(onAccount);

    if (numDone != 0)
        reauthorizeBudgetSent = Date::now();
}

void
SlaveBanker::
onReauthorizeBudgetMessage(const AccountKey & accountKey,
                           std::exception_ptr exc,
                           int responseCode,
                           const std::string & payload)
{
    //cerr << "finished reauthorize budget" << endl;

    if (exc) {
        cerr << "reauthorize budget got exception" << payload << endl;
        cerr << "accountKey = " << accountKey << endl;
        abort();  // for now...
        return;
    }
    else if (responseCode == 200) {
        Account masterAccount = Account::fromJson(Json::parse(payload));
        accounts.syncFromMaster(accountKey, masterAccount);
    }
    reauthorizeBudgetSent = Date();
}

MonitorIndicator
SlaveBanker::
getProviderIndicators() const
{
    Date now = Date::now();

    // See syncAll for the reason of this lock
    std::lock_guard<Lock> guard(syncLock);
    bool syncOk = now < lastSync.plusSeconds(MaximumFailSyncSeconds);

    MonitorIndicator ind;
    ind.serviceName = accountSuffix;
    ind.status = syncOk;
    ind.message = string() + "Sync with MasterBanker: " + (syncOk ? "OK" : "ERROR");

    return ind;
}

} // namespace RTBKIT

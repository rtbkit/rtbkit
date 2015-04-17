/* slave_banker.cc
   Jeremy Barnes, 8 November 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Slave banker implementation.
*/

#include "slave_banker.h"
#include "jml/utils/vector_utils.h"

using namespace std;
using namespace Datacratic;

namespace Default {
    static constexpr int MaximumFailSyncSeconds = 3;

    static constexpr int ExpectedMasterHttpCode = 200;
}

namespace  {
    // @Todo: Might want to shove it in soa
    void logException(std::exception_ptr ptr, std::string message,
                      Logging::Category& category) {
        try {
            std::rethrow_exception(ptr);
        } catch (const ML::Exception& e) {
            LOG(category) << message << std::endl << e.what() << std::endl;
        }
    }

} // namespace
namespace RTBKIT {


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
    applicationLayer->addAccount(account, onResult);
}


void
SlaveBudgetController::
topupTransfer(const AccountKey & account,
              CurrencyPool amount,
              const OnBudgetResult & onResult)
{
    applicationLayer->topupTransfer(account, AT_BUDGET, amount, onResult);
}

void
SlaveBudgetController::
setBudget(const std::string & topLevelAccount,
          CurrencyPool amount,
          const OnBudgetResult & onResult)
{
    applicationLayer->setBudget(topLevelAccount, amount, onResult);
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
    applicationLayer->getAccountSummary(account, depth, onResult);
}

void
SlaveBudgetController::
getAccount(const AccountKey & accountKey,
           std::function<void (std::exception_ptr,
                               Account &&)> onResult)
{
    applicationLayer->getAccount(accountKey, onResult);
}


/*****************************************************************************/
/* SLAVE BANKER                                                              */
/*****************************************************************************/

const CurrencyPool SlaveBanker::DefaultSpendRate = CurrencyPool(USD(0.10));

Logging::Category SlaveBanker::print("SlaveBanker");
Logging::Category SlaveBanker::error("SlaveBanker Error", SlaveBanker::print);
Logging::Category SlaveBanker::trace("SlaveBanker Trace", SlaveBanker::print);

SlaveBanker::SlaveBanker()
    : createdAccounts(128), reauthorizing(false), numReauthorized(0)
{
}

SlaveBanker::
SlaveBanker(
        const std::string & accountSuffix,
        CurrencyPool spendRate,
        double syncRate,
        bool batchedUpdates)
    : createdAccounts(128), reauthorizing(false), numReauthorized(0)
{
    init(accountSuffix, spendRate, syncRate, batchedUpdates);
}

void
SlaveBanker::
init(const std::string & accountSuffix,
        CurrencyPool spendRate,
        double syncRate,
        bool batchedUpdates)
{
    if (accountSuffix.empty()) {
        throw ML::Exception("'accountSuffix' cannot be empty");
    }

    if (spendRate.isZero()) {
        throw ML::Exception("'spendRate' can not be zero");
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

    addSource("SlaveBanker::createdAccounts", createdAccounts);

    this->accountSuffix = accountSuffix;
    this->spendRate = spendRate * syncRate;

    LOG(print) << "Sync Rate: " << syncRate << std::endl;
    LOG(print) << "Spend Rate: " << spendRate.toJson().toString();

    lastSync = lastReauthorize = Date::now();
    
    addPeriodic("SlaveBanker::reportSpend", syncRate,
                std::bind(&SlaveBanker::reportSpend,
                          this,
                          std::placeholders::_1),
                true /* single threaded */);

    auto authorizePtr = batchedUpdates ?
        &SlaveBanker::reauthorizeBudgetBatched :
        &SlaveBanker::reauthorizeBudget;

    addPeriodic("SlaveBanker::reauthorizeBudget", syncRate,
                std::bind(authorizePtr,
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
    applicationLayer->syncAccount(
                          accounts.getAccount(accountKey),
                          getShadowAccountStr(accountKey),
                          onDone2);
}

void
SlaveBanker::
syncAllSync()
{
    BankerSyncResult<void> result;
    syncAll(result);
    result.get();
}

namespace {
    Logging::Category bankerDebug("BankerDebug");
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
        else {
            if (accounts.isStalled(k)) {
                LOG(bankerDebug) << "CRITICAL:" << k << std::endl;

                // let's try again
                accounts.reinitializeStalledAccount(k);
                createdAccounts.push(k);
            }
        }

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

        applicationLayer->addSpendAccount(getShadowAccountStr(accountKey), onDone2);

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
                logException(exc, "Exception when reporting spend", error);
        };
    
    syncAll(onDone);
}

void
SlaveBanker::
reauthorizeBudgetBatched(uint64_t numTimeoutsExpired)
{
    Json::Value body;
    body["amount"] = spendRate.toJson();
    body["accountType"] = "spend";

    Json::Value request;
    auto onAccount = [&](const AccountKey& key, const ShadowAccount& Account) {
        request[getShadowAccountStr(key)] = body;
    };
    accounts.forEachInitializedAndActiveAccount(onAccount);

    std::string payload = request.toStringNoNewLine();

    using std::placeholders::_1;
    using std::placeholders::_2;
    using std::placeholders::_3;
    applicationLayer->request("POST", "/v1/accounts/balance", {}, payload, std::bind(
            &SlaveBanker::onReauthorizeBudgetBatchedResponse, this, _1, _2, _3));
}

void
SlaveBanker::
onReauthorizeBudgetBatchedResponse(
        std::exception_ptr exc, int code, const std::string& payload)
{
    if (exc) {
        logException(exc, "Exception when reauthorizing budget", error);
        return;
    }

    if (code != Default::ExpectedMasterHttpCode) {
        LOG(error) << "Error when reauthorizing budget for account" << std::endl;
        LOG(error) << "Expected HTTP " << Default::ExpectedMasterHttpCode
            << ", got " << code << std::endl;
        return;
    }

    Json::Value response = Json::parse(payload);
    for (const auto& key : response.getMemberNames()) {
        auto account = Account::fromJson(response[key]);
        accounts.syncFromMaster(AccountKey(key).parent(), account);
    }

    lastReauthorize = Date::now();
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
    if (reauthorizing) {
        cerr << "warning: reauthorize budget still in progress" << endl;
        return;
    }

    accountsLeft = 0;

    // For each of our accounts, we report back what has been spent
    // and re-up to our desired float
    auto onAccount = [&] (const AccountKey & key,
                          const ShadowAccount & account)
        {
            Json::Value payload = spendRate.toJson();

            auto onDone = std::bind(&SlaveBanker::onReauthorizeBudgetMessage, this,
                                    key,
                                    std::placeholders::_1, std::placeholders::_2,
                                    std::placeholders::_3);

            accountsLeft++;

            // Finally, send it out
            applicationLayer->request(
                            "POST", "/v1/accounts/" + getShadowAccountStr(key) + "/balance",
                            { { "accountType", "spend" } },
                            payload.toString(),
                            onDone);
        };
    accounts.forEachInitializedAndActiveAccount(onAccount);
    
    if (accountsLeft > 0) {
        reauthorizing = true;
        reauthorizeDate = Date::now();
    }
}

void
SlaveBanker::
onReauthorizeBudgetMessage(const AccountKey & accountKey,
                           std::exception_ptr exc,
                           int responseCode,
                           const std::string & payload)
{
    if (exc) {
        logException(exc,
              ML::format("Exception when reauthorizing budget for account '%s'",
                          accountKey.toString().c_str()),
              error);
    }
    else if (responseCode == Default::ExpectedMasterHttpCode) {
        Account masterAccount = Account::fromJson(Json::parse(payload));
        accounts.syncFromMaster(accountKey, masterAccount);
    }
    else {
        LOG(error) << "Error when reauthorizing budget for account '%s'"
                   << accountKey << std::endl
                   << "Expected HTTP " << Default::ExpectedMasterHttpCode << ", got "
                   << responseCode << std::endl;
    }


    accountsLeft--;
    if (accountsLeft == 0) {
        lastReauthorizeDelay = Date::now() - reauthorizeDate;
        numReauthorized++;
        reauthorizing = false;

        std::lock_guard<Lock> guard(syncLock);
        lastReauthorize = Date::now();
    }
}

void
SlaveBanker::
waitReauthorized()
    const
{
    while (reauthorizing) {
        ML::sleep(0.2);
    }
}

MonitorIndicator
SlaveBanker::
getProviderIndicators() const
{
    Date now = Date::now();

    // See syncAll for the reason of this lock
    std::lock_guard<Lock> guard(syncLock);
    bool syncOk = now < lastSync.plusSeconds(Default::MaximumFailSyncSeconds) &&
                  now < lastReauthorize.plusSeconds(Default::MaximumFailSyncSeconds);

    MonitorIndicator ind;
    ind.serviceName = accountSuffix;
    ind.status = syncOk;
    ind.message = string() + "Sync with MasterBanker: " + (syncOk ? "OK" : "ERROR");

    return ind;
}

/*****************************************************************************/
/* SLAVE BANKER ARGUMENTS                                                    */
/*****************************************************************************/

constexpr bool SlaveBankerArguments::Defaults::UseHttp;
constexpr bool SlaveBankerArguments::Defaults::Batched;
constexpr int SlaveBankerArguments::Defaults::HttpConnections;
constexpr bool SlaveBankerArguments::Defaults::TcpNoDelay;
const std::string SlaveBankerArguments::Defaults::SpendRate{"100000USD/1M"};

SlaveBankerArguments::SlaveBankerArguments()
    : spendRateStr(Defaults::SpendRate)
    , syncRate(Defaults::SyncRate)
    , batched(Defaults::Batched)
    , useHttp(Defaults::UseHttp)
    , httpTimeout(Defaults::HttpTimeout)
    , httpConnections(Defaults::HttpConnections)
    , tcpNoDelay(Defaults::TcpNoDelay)
{
}

Logging::Category SlaveBankerArguments::print("SlaveBankerArguments");
Logging::Category SlaveBankerArguments::error(
        "SlaveBankerArguments Error", SlaveBankerArguments::print);
Logging::Category SlaveBankerArguments::trace(
        "SlaveBankerArguments Trace", SlaveBankerArguments::print);

boost::program_options::options_description
SlaveBankerArguments::makeProgramOptions(std::string title)
{
    namespace po = boost::program_options;

    po::options_description options(std::move(title));
    options.add_options()
        ("spend-rate", po::value<string>(&spendRateStr),
         "Amount of budget in USD to be periodically re-authorized (default 100000USD/1M)")
        ("banker-sync-rate", po::value<double>(&syncRate),
         "frequency at which the slave banker syncs itself with the master banker.")
        ("banker-batched", po::bool_switch(&batched),
         "slave banker now uses batched communication to sync with the master banker.")
        ("use-http-banker", po::bool_switch(&useHttp),
         "Communicate with the MasterBanker over http")
        ("banker-http-timeouts", po::value<double>(&httpTimeout),
         "banker sync request timeout over http.")
        ("http-connections", po::value<int>(&httpConnections)->default_value(Defaults::HttpConnections),
         "Number of active http connections to use when http is enabled")
        ("banker-tcp-nodelay", po::bool_switch(&tcpNoDelay),
          "Enable the TCP_NODELAY option for the http banker interface (use with caution)");

    return options;
}

void
SlaveBankerArguments::
validate() const {
    throw ML::Exception("Unimplemented");
}

std::shared_ptr<SlaveBanker>
SlaveBankerArguments::
makeBanker(std::shared_ptr<ServiceProxies> proxies, const std::string& accountSuffix) const
{
    auto spendRate = CurrencyPool(Amount::parse(spendRateStr));
    auto banker = std::make_shared<SlaveBanker>(accountSuffix, spendRate, syncRate, batched);

    banker->setApplicationLayer(makeApplicationLayer(std::move(proxies)));
    return banker;
}

std::shared_ptr<SlaveBanker>
SlaveBankerArguments::makeBankerDefault(std::shared_ptr<ServiceProxies> proxies) const {
    return makeBanker(std::move(proxies), "");
}

std::shared_ptr<ApplicationLayer>
SlaveBankerArguments::makeApplicationLayer(std::shared_ptr<ServiceProxies> proxies) const
{
    std::shared_ptr<ApplicationLayer> layer;
    if (useHttp) {
        auto bankerUri = proxies->bankerUri;

        ExcCheck(!bankerUri.empty(),
                "the banker-uri must be specified in the bootstrap.json");
        ExcCheck(httpConnections > 0,
                "The number of active http connections must be > 0");

        auto httpTimeout = this->httpTimeout * syncRate;

        std::stringstream ss;
        ss << "using http interface for the MasterBanker" << std::endl;
        ss << "url                = " << bankerUri << std::endl;
        ss << "timeout            = " << httpTimeout << std::endl;
        ss << "active connections = " << httpConnections << std::endl;
        ss << "tcp no delay       = " << tcpNoDelay;
        LOG(print) << ss.str() << std::endl;

        layer = make_application_layer<HttpLayer>(bankerUri, httpTimeout, httpConnections, tcpNoDelay);
    }
    else {
        layer = make_application_layer<ZmqLayer>(proxies);
        LOG(print) << "using zmq interface for the MasterBanker" << std::endl;
    }

    return layer;
}

Amount
SlaveBankerArguments::spendRate() const {
    return Amount::parse(spendRateStr);
}

} // namespace RTBKIT

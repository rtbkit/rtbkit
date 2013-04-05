/* slave_banker.cc
   Jeremy Barnes, 8 November 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Slave banker implementation.
*/

#include "slave_banker.h"
#include "soa/service/http_header.h"
#include "jml/utils/vector_utils.h"

using namespace std;


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
    push(budgetResultCallback(onResult),
         "POST", "/v1/accounts",
         { {"accountName", account.toString()},
           { "accountType", "budget" } });
}


void
SlaveBudgetController::
topupTransfer(const AccountKey & account,
              CurrencyPool amount,
              const OnBudgetResult & onResult)
{
    push(budgetResultCallback(onResult),
         "PUT", "/v1/accounts/" + account.toString() + "/balance",
         { { "accountType", "budget"} },
         amount.toJson().toString());
}

void
SlaveBudgetController::
setBudget(const std::string & topLevelAccount,
          CurrencyPool amount,
          const OnBudgetResult & onResult)
{
    push(budgetResultCallback(onResult),
         "PUT", "/v1/accounts/" + topLevelAccount + "/budget",
         { /* {"amount", amount.toString()}*/ },
         amount.toJson().toString());
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
    push([=] (std::exception_ptr ptr, int resultCode, string body)
         {
             AccountSummary summary;
             if (ptr)
                 onResult(ptr, std::move(summary));
             else if (resultCode < 200 || resultCode >= 300)
                 onResult(std::make_exception_ptr(ML::Exception("getAccountSummary returned code %d: %s", resultCode, body.c_str())),
                          std::move(summary));
             else {
                 try {
                     Json::Value result = Json::parse(body);
                     summary = AccountSummary::fromJson(result);
                     onResult(nullptr, std::move(summary));
                 } catch (...) {
                     onResult(std::current_exception(), std::move(summary));
                 }
             }
         },
         "GET", "/v1/accounts/" + account.toString() + "/summary",
         { {"depth", to_string(depth)} },
         "");
}

void
SlaveBudgetController::
getAccount(const AccountKey & accountKey,
           std::function<void (std::exception_ptr,
                               Account &&)> onResult)
{
    push([=] (std::exception_ptr ptr, int resultCode, string body)
         {
             Account account;
             if (ptr)
                 onResult(ptr, std::move(account));
             else if (resultCode < 200 || resultCode >= 300)
                 onResult(std::make_exception_ptr(ML::Exception("getAccount returned code %d: %s", resultCode, body.c_str())),
                          std::move(account));
             else {
                 try {
                     Json::Value result = Json::parse(body);
                     account = Account::fromJson(result);
                     onResult(nullptr, std::move(account));
                 } catch (...) {
                     onResult(std::current_exception(), std::move(account));
                 }
             }
         },
         "GET", "/v1/accounts/" + accountKey.toString());
}

SlaveBudgetController::OnDone
SlaveBudgetController::
budgetResultCallback(const OnBudgetResult & onResult)
{
    return [=] (std::exception_ptr ptr, int resultCode, string body)
        {
            //cerr << "got budget result callback with resultCode "
            //     << resultCode << " body " << body << endl;
            onResult(ptr);
        };
}


/*****************************************************************************/
/* SLAVE BANKER                                                              */
/*****************************************************************************/

SlaveBanker::
SlaveBanker(std::shared_ptr<zmq::context_t> context)
    : RestProxy(context), createdAccounts(128)
{
}

SlaveBanker::
SlaveBanker(std::shared_ptr<zmq::context_t> context,
            std::shared_ptr<ConfigurationService> config,
            const std::string & accountSuffix,
            const std::string & bankerServiceName)
    : RestProxy(context), createdAccounts(128)
{
    init(config, accountSuffix, bankerServiceName);
}

void
SlaveBanker::
init(std::shared_ptr<ConfigurationService> config,
     const std::string & accountSuffix,
     const std::string & bankerServiceName)
{
    if (accountSuffix.empty()) {
        throw ML::Exception("'accountSuffix' cannot be empty");
    }
    if (bankerServiceName.empty()) {
        throw ML::Exception("'bankerServiceName' cannot be empty");
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
    
    // Connect to the master banker
    RestProxy::initServiceClass(config, bankerServiceName, "zeromq");
    
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

    push(makeRestResponseJsonDecoder<Account>("syncAccount", onDone2),
         "PUT",
         "/v1/accounts/" + getShadowAccountStr(accountKey) + "/shadow",
         {},
         accounts.getAccount(accountKey).toJson().toString());
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
        if (onDone)
            onDone(nullptr);
        return;
    }

    struct Aggregator {

        Aggregator(int numTotal,
                   std::function<void (std::exception_ptr)> onDone)
            : itl(new Itl())
        {
            itl->numTotal = numTotal;
            itl->numFinished = 0;
            itl->exc = nullptr;
            itl->onDone = onDone;
        }

        struct Itl {
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
    
    Aggregator aggregator(allKeys.size(), onDone);

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
		auto onDone2
			= std::bind(&SlaveBanker::onInitializeResult,
						this,
						accountKey,
						onDone,
						std::placeholders::_1,
						std::placeholders::_2);

		cerr << "********* calling addSpendAccount for " << accountKey
			 << " for SlaveBanker " << accountSuffix << endl;

		push(makeRestResponseJsonDecoder<Account>("addSpendAccount", onDone2),
			 "POST",
			 "/v1/accounts",
			 { { "accountName", getShadowAccountStr(accountKey) },
			   { "accountType", "spend" } },
			 "");
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
            RestRequest request;
            request.verb = "POST";
            request.resource
                = "/v1/accounts/"
                + getShadowAccountStr(key)
                + "/balance";
            request.params = { { "accountType", "spend" } };

            Json::Value payload = CurrencyPool(USD(0.10)).toJson();
            request.payload = payload.toString();
            
            //cerr << "sending out request " << request << endl;
            ++numDone;

            // Finally, send it out
            push(request, std::bind(&SlaveBanker::onReauthorizeBudgetMessage,
                                    this,
                                    key,
                                    std::placeholders::_1,
                                    std::placeholders::_2,
                                    std::placeholders::_3));
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

    Account masterAccount = Account::fromJson(Json::parse(payload));
    accounts.syncFromMaster(accountKey, masterAccount);
    reauthorizeBudgetSent = Date();
}

} // namespace RTBKIT

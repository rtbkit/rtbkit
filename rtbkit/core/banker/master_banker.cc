/* master_banker.cc
   Jeremy Barnes, 8 November 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.
   
   Master banker class.
*/

#include <memory>
#include <string>
#include "soa/jsoncpp/value.h"
#include <boost/algorithm/string.hpp>
#include <jml/arch/futex.h>

#include "master_banker.h"
#include "soa/service/rest_request_binding.h"
#include "soa/service/redis.h"


using namespace std;
using namespace ML;
using namespace Redis;


namespace RTBKIT {

using Datacratic::jsonEncode;
using Datacratic::jsonDecode;

double bankerSaveAllPeriod = 10.0;

Logging::Category MasterBanker::print("MasterBanker");
Logging::Category MasterBanker::trace("MasterBanker Trace", MasterBanker::print);
Logging::Category MasterBanker::error("MasterBanker Error", MasterBanker::print);

Logging::Category BankerPersistence::print("BankerPersistence");
Logging::Category BankerPersistence::trace("BankerPersistence Trace", BankerPersistence::print);
Logging::Category BankerPersistence::error("BankerPersistence Error", BankerPersistence::print);


/*****************************************************************************/
/* REDIS BANKER PERSISTENCE                                                  */
/*****************************************************************************/

struct RedisBankerPersistence::Itl {
    shared_ptr<Redis::AsyncConnection> redis;

    int timeout;
};

RedisBankerPersistence::
RedisBankerPersistence(const Redis::Address & redis, int timeout)
{
    itl = make_shared<Itl>();
    itl->redis = make_shared<Redis::AsyncConnection>(redis);
    itl->timeout = timeout;
}

RedisBankerPersistence::
RedisBankerPersistence(shared_ptr<Redis::AsyncConnection> redis, int timeout)
{
    itl = make_shared<Itl>();
    itl->redis = redis;
    itl->timeout = timeout;
}

void
RedisBankerPersistence::
loadAll(const string & topLevelKey, OnLoadedCallback onLoaded)
{
    shared_ptr<Accounts> newAccounts;

    Redis::Result result = itl->redis->exec(SMEMBERS("banker:accounts"), itl->timeout);
    if (!result.ok()) {
        onLoaded(newAccounts, BACKEND_ERROR, result.error());
        return;
    }

    const Reply & keysReply = result.reply();
    if (keysReply.type() != ARRAY) {
        onLoaded(newAccounts, DATA_INCONSISTENCY,
                 "SMEMBERS 'banker:accounts' must return an array");
        return;
    }

    newAccounts = make_shared<Accounts>();
    if (keysReply.length() == 0) {
        onLoaded(newAccounts, SUCCESS, "");
        return;
    }

    Command fetchCommand(MGET);
    vector<string> keys;
    for (int i = 0; i < keysReply.length(); i++) {
        string key(keysReply[i].asString());
        keys.push_back(key);
        fetchCommand.addArg("banker-" + key);
    }

    result = itl->redis->exec(fetchCommand, itl->timeout);
    if (!result.ok()) {
        onLoaded(newAccounts, BACKEND_ERROR, result.error());
        return;
    }

    const Reply & accountsReply = result.reply();
    ExcAssert(accountsReply.type() == ARRAY);
    for (int i = 0; i < accountsReply.length(); i++) {
        if (accountsReply[i].type() == NIL) {
            onLoaded(newAccounts, DATA_INCONSISTENCY,
                     "nil key '" + keys[i]
                     + "' referenced in 'banker:accounts'");
            return;
        }
        Json::Value storageValue = Json::parse(accountsReply[i]);
        newAccounts->restoreAccount(AccountKey(keys[i]), storageValue);
    }

    // newAccounts->checkBudgetConsistency();

    onLoaded(newAccounts, SUCCESS, "");
}

void
RedisBankerPersistence::
saveAll(const Accounts & toSave, OnSavedCallback onSaved)
{
    /* TODO: we need to check the content of the "banker:accounts" set for
     * "extra" account keys */

    // Phase 1: we load all of the keys.  This way we can know what is
    // present and deal with keys that should be zeroed out.  We can also
    // detect if we have a synchronization error and bail out.

    const Date begin = Date::now();
    vector<string> keys;

    Redis::Command fetchCommand(MGET);

    auto latencyBetween = [](const Date& lhs, const Date& rhs) {
        return rhs.secondsSince(lhs) * 1000;
    };

    /* fetch all account keys and values from storage */
    auto onAccount = [&] (const AccountKey & key,
                          const Account & account)
        {
            string keyStr = key.toString();
            keys.push_back(keyStr);
            fetchCommand.addArg("banker-" + keyStr);
        };
    toSave.forEachAccount(onAccount);

    const Date beforePhase1Time = Date::now();
    auto onPhase1Result = [=] (const Redis::Result & result)
        {
            BankerPersistence::Result saveResult;

            const Date afterPhase1Time = Date::now();
            saveResult.recordLatency(
                    "redisPhase1TimeMs", latencyBetween(beforePhase1Time, afterPhase1Time));

            if (!result.ok()) {
                saveResult.status = BACKEND_ERROR;
                saveResult.recordLatency(
                        "totalTimeMs", latencyBetween(begin, Date::now()));
                LOG(error) << "phase1 save operation failed with error '"
                           << result.error() << "'" << std::endl;
                onSaved(saveResult, result.error());
                return;
            }

            vector<Redis::Command> storeCommands;
            storeCommands.push_back(MULTI);

            const Reply & reply = result.reply();
            ExcAssert(reply.type() == ARRAY);

            Json::Value badAccounts(Json::arrayValue);

            /* All accounts known to the banker are fetched.
               We need to check them and restore them (if needed). */
            for (int i = 0; i < reply.length(); i++) {
                const string & key = keys[i];
                bool isParentAccount(key.find(":") == string::npos);
                const Accounts::AccountInfo & bankerAccount
                    = toSave.getAccount(key);
                if (toSave.isAccountOutOfSync(key)) {
                    LOG(trace) << "account '" << key
                               << "' is out of sync and will not be saved" << endl;
                    continue;
                }
                Json::Value bankerValue = bankerAccount.toJson();
                bool saveAccount(false);

                Redis::Result result = reply[i];
                Reply accountReply = result.reply();
                if (accountReply.type() == STRING) {
                    // We have here:
                    // a) an account that we want to write;
                    // b) the current in-database representation of that
                    //    account

                    // We need to do the following:
                    // 1.  Make sure that it's a valid update (eg, that no
                    //     always increasing numbers would go down and that
                    //     the data in the db is correct);
                    // 2.  Find what keys we need to modify to make it
                    //     correct
                    // 3.  Perform the modifications

                    Account storageAccount;
                    Json::Value storageValue = Json::parse(accountReply.asString());
                    storageAccount = storageAccount.fromJson(storageValue);
                    if (bankerAccount.isSameOrPastVersion(storageAccount)) {
                        /* FIXME: the need for updating an account should
                           probably be deduced differently than by comparing
                           JSON content.
                        */
                        saveAccount = (bankerValue != storageValue);

                        if (saveAccount && isParentAccount) {
                            /* set and update the "spent-tracking" output for top
                             * accounts, by integrating the past */
                            if (storageValue.isMember("spent-tracking")) {
                                bankerValue["spent-tracking"]
                                    = storageValue["spent-tracking"];
                            }
                            else {
                                bankerValue["spent-tracking"]
                                    = Json::Value(Json::objectValue);
                            }
                        }
                    }
                    else {
                        /* TODO: the list of inconsistent account should be
                           stored in the db */
                        badAccounts.append(Json::Value(key));
                    }
                }
                else {
                    /* The account does not exist yet in storage, thus we
                       create it. */
                    storeCommands.push_back(SADD("banker:accounts", key));
                    saveAccount = true;
                    if (isParentAccount) {
                        bankerValue["spent-tracking"]
                            = Json::Value(Json::objectValue);
                    }
                }

                if (saveAccount) {
                    if (isParentAccount) {
                        const AccountSummary & summary = toSave.getAccountSummary(key);
                        if (summary.spent != bankerAccount.initialSpent) {
                            // cerr << "adding tracking entry" << endl;
                            string sessionStartStr = toSave.sessionStart.printClassic();
                            bankerValue["spent-tracking"][sessionStartStr]
                                = Json::Value(Json::objectValue);
                            Json::Value & tracking
                                = bankerValue["spent-tracking"][sessionStartStr];
                            CurrencyPool delta(summary.spent
                                               - bankerAccount.initialSpent);
                            tracking["spent"] = delta.toJson();
                            string lastModifiedStr = Date::now().printClassic();
                            tracking["date"] = lastModifiedStr;
                        }
                    }

                    Redis::Command command = SET("banker-" + key,
                                                 boost::trim_copy(bankerValue.toString()));
                    storeCommands.push_back(command);
                }
            }

            if (badAccounts.size() > 0) {
                /* For now we do not save any account when at least one has
                   been detected as inconsistent. */
                saveResult.status = DATA_INCONSISTENCY;
                const Date now = Date::now();
                saveResult.recordLatency(
                        "inPhase1TimeMs", latencyBetween(afterPhase1Time, now));
                saveResult.recordLatency(
                        "totalTimeMs", latencyBetween(begin, now));
                onSaved(saveResult, boost::trim_copy(badAccounts.toString()));
            }
            else if (storeCommands.size() > 1) {
                 storeCommands.push_back(EXEC);
                 
                 const Date beforePhase2Time = Date::now();

                 saveResult.recordLatency(
                        "inPhase1TimeMs", latencyBetween(afterPhase1Time, Date::now()));

                 auto onPhase2Result = [=] (const Redis::Results & results) mutable
                 {
                     const Date afterPhase2Time = Date::now();
                     saveResult.recordLatency(
                             "redisPhase2TimeMs", latencyBetween(beforePhase2Time, afterPhase2Time));

                     saveResult.recordLatency(
                             "totalTimeMs", latencyBetween(begin, Date::now()));

                     if (results.ok()) {
                         saveResult.status = SUCCESS;
                         onSaved(saveResult, "");
                     }
                     else {
                         LOG(error) << "phase2 save operation failed with error '"
                                   << results.error() << "'" << std::endl;
                         saveResult.status = BACKEND_ERROR;
                         onSaved(saveResult, results.error());
                     }
                 };

                 itl->redis->queueMulti(storeCommands, onPhase2Result, itl->timeout);
            }
            else {
                saveResult.status = SUCCESS;
                saveResult.recordLatency(
                        "inPhase1TimeMs", latencyBetween(afterPhase1Time, Date::now()));
                saveResult.recordLatency(
                        "totalTimeMs", latencyBetween(begin, Date::now()));
                onSaved(saveResult, "");
            }
        };

    if (keys.size() == 0) {
        /* no account to save */
        BankerPersistence::Result result;
        result.status = SUCCESS;
        result.recordLatency("totalTimeMs", latencyBetween(begin, Date::now()));
        onSaved(result, "");
        return;
    }

    itl->redis->queue(fetchCommand, onPhase1Result, itl->timeout);
}

/*****************************************************************************/
/* MASTER BANKER                                                             */
/*****************************************************************************/

MasterBanker::
MasterBanker(std::shared_ptr<ServiceProxies> proxies,
             const string & serviceName)
    : ServiceBase(serviceName, proxies),
      RestServiceEndpoint(proxies->zmqContext),
      saving(false)
{
    /* Set the Access-Control-Allow-Origins: * header to allow browser-based
       REST calls directly to the endpoint.
    */
    httpEndpoint.allowAllOrigins();
}

MasterBanker::
~MasterBanker()
{
    saveState();
    while (saving) {
        cerr << "awaiting end of save operation..." << endl;
        ML::futex_wait(saving, true);
    }
    shutdown();
}

void
MasterBanker::
init(const shared_ptr<BankerPersistence> & storage, double saveInterval)
{
    this->storage_ = storage;

    loadStateSync();

    addPeriodic("MasterBanker::saveState", saveInterval,
                bind(&MasterBanker::saveState, this),
                true /* single threaded */);

    registerServiceProvider(serviceName(), { "rtbBanker" });

    getServices()->config->removePath(serviceName());
    RestServiceEndpoint::init(getServices()->config, serviceName());

    onHandleRequest = router.requestHandler();

    router.description = "API for the Datacratic Banker Service";

    router.addHelpRoute("/", "GET");

    RestRequestRouter::OnProcessRequest pingRoute
        = [] (const RestServiceEndpoint::ConnectionId & connection,
              const RestRequest & request,
              const RestRequestParsingContext & context) {
        connection.sendResponse(200, "pong");
        return RestRequestRouter::MR_YES;
    };
    router.addRoute("/ping", "GET", "availability request", pingRoute,
                    Json::Value());

    auto & versionNode = router.addSubRouter("/v1", "version 1 of API");

    addRouteSyncReturn(versionNode,
                       "/summary",
                       {"GET"},
                       "Return the simplified summaries of all existing"
                       " accounts",
                       "",
                       [] (const Json::Value & a) { return a; },
                       &MasterBanker::getAccountsSimpleSummaries,
                       this,
                       RestParamDefault<int>("maxDepth", "maximum depth to traverse", 3));


    auto & accountsNode
        = versionNode.addSubRouter("/accounts",
                                   "Operations on accounts");
    
    addRouteSyncReturn(accountsNode,
                       "",
                       {"POST"},
                       "Add a new account to the banker",
                       "Representation of the added account",
                       [] (const Account & a) { return a.toJson(); },
                       &MasterBanker::onCreateAccount,
                       this,
                       RestParam<AccountKey>("accountName", "account name to create x:y:z"),
                       RestParam<AccountType>("accountType", "account type (spend or budget)"));
    
    addRouteSyncReturn(accountsNode,
                       "",
                       {"GET"},
                       "List accounts that are in the banker",
                       "List of account names matching the given prefix",
                       [] (const vector<AccountKey> & v) { return jsonEncode(v); },
                       &Accounts::getAccountKeys,
                       &accounts,
                       RestParamDefault<AccountKey>
                       ("accountPrefix",
                        "account name to look under (default empty which "
                        "means return all accounts)",
                        AccountKey()),
                       RestParamDefault<int>
                       ("maxDepth", "maximum depth to search (default unlimited)", -1));
    
    auto & account
        = accountsNode.addSubRouter(Rx("/([^/]*)", "/<accountName>"),
                                    "operations on an individual account");
    
    RequestParam<AccountKey> accountKeyParam(-2, "<account>", "account to operate on");

    addRouteSyncReturn(account,
                       "",
                       {"GET"},
                       "Return a representation of the given account",
                       "Representation of the named account",
                       [] (const Account & account) { return account.toJson(); },
                       &Accounts::getAccount,
                       &accounts,
                       accountKeyParam);

    addRouteSyncReturn(account,
                       "/subtree",
                       {"GET"},
                       "Return a representation of the given account and its "
                       "children",
                       "Representation of the given subtree",
                       [] (const Accounts & subtree) { return subtree.toJson(); },
                       &Accounts::getAccounts,
                       &accounts,
                       accountKeyParam,
                       RestParamDefault<int>("depth", "depth of children (default = 0)", 0));

    addRouteSyncReturn(account,
                       "/children",
                       {"GET"},
                       "Return a list of the children of a given account",
                       "Array of names of child accounts",
                       [] (const vector<AccountKey> & keys) { return jsonEncode(keys); },
                       &Accounts::getAccountKeys,
                       &accounts,
                       accountKeyParam,
                       RestParamDefault<int>("depth", "depth of children (default = 0)", 0));

    addRouteSyncReturn(account,
                       "/budget",
                       {"PUT", "POST"},
                       "Set a top level account's budget to match the given "
                       "amount.  ",
                       "Status of the account after the operation",
                       [] (const Account & a) { return a.toJson(); },
                       &MasterBanker::setBudget,
                       this,
                       accountKeyParam,
                       JsonParam<CurrencyPool>("", "amount to set budget to"));

    addRouteSyncReturn(account,
                       "/balance",
                       {"PUT", "POST"},
                       "Transfer budget from the parent such that account's "
                       "balance amount matches the parameter",
                       "Account: Representation of the modified account",
                       [] (const Account & a) { return a.toJson(); },
                       &MasterBanker::setBalance,
                       this,
                       accountKeyParam,
                       JsonParam<CurrencyPool>("", "amount to set balance to"),
                       RestParamDefault<AccountType>("accountType", "type of account for implicit creation (default no creation)", AT_NONE));
    
    addRouteSyncReturn(account,
                       "/adjustment",
                       {"PUT", "POST"},
                       "Perform an adjustment to the account",
                       "Account: Representation of the modified account",
                       [] (const Account & a) { return a.toJson(); },
                       &MasterBanker::addAdjustment,
                       this,
                       accountKeyParam,
                       JsonParam<CurrencyPool>("", "amount to add or substract"));

    addRouteSyncReturn(account,
                       "/summary",
                       "GET",
                       "Return the aggregated summary of the given account",
                       "AccountSummary: aggregation of the given account and its children",
                       [] (const AccountSummary & s) { return s.toJson(); },
                       &Accounts::getAccountSummary,
                       &accounts,
                       accountKeyParam,
                       RestParamDefault<int>("maxDepth", "maximum depth to traverse", 3));

    addRouteSyncReturn(account,
                       "/shadow",
                       {"PUT", "POST"},
                       "Update a spend account's spend and commitments",
                       "Account: Representation of the modified account",
                       [] (const Account & a) { return a.toJson(); },
                       &MasterBanker::syncFromShadow,
                       this,
                       accountKeyParam,
                       JsonParam<ShadowAccount>("",
                                                "Representation of the shadow account"));
    addRouteSyncReturn(account,
                       "/close",
                       {"GET"},
                       "Close an account and all of its child accounts, "
                       "transfers all remaining balances to parent.",
                       "Account: Representation of the modified account",
                       [] (const std::vector<AccountSummary> & a) { 
                            Json::Value result(Json::objectValue);
                            result["account_before_close"] = a[0].toJson();
                            result["account_after_close"] = a[1].toJson();
                            if (a.size() == 4) {
                                result["parent_before_close"] = a[2].toJson();
                                result["parent_after_close"] = a[3].toJson();
                            } else {
                                result["parent_before_close"] = "No Parent Account";
                                result["parent_after_close"] = "No Parent Account";
                            }
                            return result;
                       },
                       &MasterBanker::closeAccount,
                       this,
                       accountKeyParam);

    addRouteSyncReturn(versionNode,
                       "/activeaccounts",
                       {"GET"},
                       "Return a list of all active account names",
                       "Account: list of Account Keys",
                       [] (const std::vector<AccountKey> & accountKeys) {
                            Json::Value result(Json::arrayValue);
                            for (const AccountKey & a : accountKeys)
                                result.append(a.toString());
                            return result;
                       },
                       &MasterBanker::getActiveAccounts,
                       this);


}

void
MasterBanker::
start()
{
    RestServiceEndpoint::start();
}

pair<string, string>
MasterBanker::
bindTcp()
{
    return RestServiceEndpoint::bindTcp(
            getServices()->ports->getRange("banker.zmq"),
            getServices()->ports->getRange("banker.http"));
}

void
MasterBanker::
shutdown()
{
    unregisterServiceProvider(serviceName(), { "rtbBanker" });
    RestServiceEndpoint::shutdown();
}

Json::Value
MasterBanker::
createAccount(const AccountKey & key, AccountType type)
{
    Account account = accounts.createAccount(key, type);
    return account.toJson();

    if (type == AT_BUDGET)
        return account.toJson();
    else {
        ShadowAccount shadow;
        shadow.syncFromMaster(account);
        return shadow.toJson();
    }
}

Json::Value
MasterBanker::
getAccountsSimpleSummaries(int depth)
{
    return accounts.getAccountSummariesJson(true, depth);
}

void
MasterBanker::
onStateSaved(const BankerPersistence::Result& result,
             const string & info)
{
    if (result.status == BankerPersistence::SUCCESS) {
        //cerr << __FUNCTION__
        //     <<  ": banker state saved successfully to backend" << endl;
        lastSavedState = Date::now();
    }
    else if (result.status == BankerPersistence::DATA_INCONSISTENCY) {
        Json::Value accountKeys = Json::parse(info);
        ExcAssert(accountKeys.type() == Json::arrayValue);
        for (Json::Value jsonKey: accountKeys) {
            ExcAssert(jsonKey.type() == Json::stringValue);
            string keyStr = jsonKey.asString();
            accounts.markAccountOutOfSync(AccountKey(keyStr));
            LOG(error) << "account '" << keyStr << "' marked out of sync" << endl;
        }
    }
    else if (result.status == BankerPersistence::BACKEND_ERROR) {
        /* the backend is unavailable */
        LOG(error) << "Failed to save banker state, backend unavailable: "
                   << info << endl;
    }
    else {
        throw ML::Exception("status code is not handled");
    }

    lastSaveStatus = result.status;

    reportLatencies("save state", result.latencies);
    saving = false;
    ML::futex_wake(saving);
}

void
MasterBanker::
saveState()
{
    Guard guard(saveLock);

    if (!storage_ || saving)
        return;

    saving = true;
    storage_->saveAll(accounts, bind(&MasterBanker::onStateSaved, this,
                                          placeholders::_1,
                                          placeholders::_2));
}

void
MasterBanker::
onStateLoaded(shared_ptr<Accounts> newAccounts,
              const BankerPersistence::Result& result,
              const string & info)
{
    if (result.status == BankerPersistence::SUCCESS) {
        newAccounts->ensureInterAccountConsistency();
        accounts = *newAccounts;
        LOG(print) << "successfully loaded accounts" << endl;
    }
    else if (result.status == BankerPersistence::DATA_INCONSISTENCY) {
        /* something is wrong with the backend data types */
        LOG(error) << "Failed to load accounts, DATA_INCONSISTENCY: " << info << endl;
    }
    else if (result.status == BankerPersistence::BACKEND_ERROR) {
        /* the backend is unavailable */
        LOG(error) << "Failed to load accounts, backend unavailable: " << info << endl;
    }
    else {
        throw ML::Exception("status code is not handled");
    }
}

void
MasterBanker::
loadStateSync()
{
    if (!storage_)
        return;

    int done = 0;

    auto onLoaded = [&](shared_ptr<Accounts> accounts,
                        BankerPersistence::PersistenceCallbackStatus status,
                        const string & info) {
        this->onStateLoaded(accounts, status, info);
        done = 1;
        ML::futex_wake(done);
    };

    storage_->loadAll("", onLoaded);

    while (!done) {
        ML::futex_wait(done, 0);
    }
}

void
MasterBanker::
bindFixedHttpAddress(const string & uri)
{
}

const Account
MasterBanker::
setBudget(const AccountKey &key, const CurrencyPool &newBudget)
{
    JML_TRACE_EXCEPTIONS(false);
    if (lastSaveStatus == BankerPersistence::BACKEND_ERROR)
        throw ML::Exception("Error with the backend");

    return accounts.setBudget(key, newBudget);
}

const Account
MasterBanker::
onCreateAccount(const AccountKey &key, AccountType type)
{
    JML_TRACE_EXCEPTIONS(false);
    if (lastSaveStatus == BankerPersistence::BACKEND_ERROR)
        throw ML::Exception("Error with the backend");

    return accounts.createAccount(key, type);
}

const std::vector<AccountSummary>
MasterBanker::
closeAccount(const AccountKey &key)
{
    
    JML_TRACE_EXCEPTIONS(false);
    if (lastSaveStatus == BankerPersistence::BACKEND_ERROR)
        throw ML::Exception("Error with the backend");
 
    AccountKey parentKey = key;
    if (key.size() > 1) {
        parentKey.pop_back();
    }

    AccountSummary before = accounts.getAccountSummary(key);
    AccountSummary beforeParent = accounts.getAccountSummary(parentKey);

    accounts.closeAccount(key);
    
    AccountSummary after = accounts.getAccountSummary(key);
    AccountSummary afterParent = accounts.getAccountSummary(parentKey);

    std::vector<AccountSummary> testClose = {before, after};
    
    if (key.size() > 1) {
        testClose.push_back(beforeParent);
        testClose.push_back(afterParent);
    }  

    return testClose;
}

const std::vector<AccountKey>
MasterBanker::
getActiveAccounts()
{
    vector<AccountKey> activeAccounts;
    auto addActive = [&activeAccounts] (const AccountKey & ak, const Account & a) {
        if (a.status == Account::ACTIVE)
            activeAccounts.push_back(ak);
    };
    accounts.forEachAccount(addActive);
    return activeAccounts;
}

const Account
MasterBanker::
setBalance(const AccountKey &key, CurrencyPool amount, AccountType type)
{
    JML_TRACE_EXCEPTIONS(false);
    if (lastSaveStatus == BankerPersistence::BACKEND_ERROR)
        throw ML::Exception("Error with the backend");

    return accounts.setBalance(key, amount, type);
}

const Account
MasterBanker::
addAdjustment(const AccountKey &key, CurrencyPool amount)
{
    JML_TRACE_EXCEPTIONS(false);
    if (lastSaveStatus == BankerPersistence::BACKEND_ERROR)
        throw ML::Exception("Error with the backend");

    return accounts.addAdjustment(key, amount);
}

const Account
MasterBanker::
syncFromShadow(const AccountKey &key, const ShadowAccount &shadow)
{
    JML_TRACE_EXCEPTIONS(false);
    if (lastSaveStatus == BankerPersistence::BACKEND_ERROR)
        throw ML::Exception("Error with the backend");

    return accounts.syncFromShadow(key, shadow);
}

void
MasterBanker::
reportLatencies(const std::string &category,
                const BankerPersistence::LatencyMap& latencies) const
{
    std::stringstream ss;
    ss << std::endl << "Latency report for " << category << std::endl;
    for (const auto& latency: latencies) {
        ss << "- " << latency.first << ": " << latency.second << std::endl;
    }

    LOG(trace) << ss.str() << std::endl;
}


} // namespace RTBKIT

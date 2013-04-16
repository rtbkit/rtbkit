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

/*****************************************************************************/
/* REDIS BANKER PERSISTENCE                                                  */
/*****************************************************************************/

struct RedisBankerPersistence::Itl {
    shared_ptr<Redis::AsyncConnection> redis;
};

RedisBankerPersistence::
RedisBankerPersistence(const Redis::Address & redis)
{
    itl = make_shared<Itl>();
    itl->redis = make_shared<Redis::AsyncConnection>(redis);
}

RedisBankerPersistence::
RedisBankerPersistence(shared_ptr<Redis::AsyncConnection> redis)
{
    itl = make_shared<Itl>();
    itl->redis = redis;
}

void
RedisBankerPersistence::
loadAll(const string & topLevelKey, OnLoadedCallback onLoaded)
{
    shared_ptr<Accounts> newAccounts;

    Redis::Result result = itl->redis->exec(SMEMBERS("banker:accounts"), 5);
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

    result = itl->redis->exec(fetchCommand, 5);
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

    vector<string> keys;

    Redis::Command fetchCommand(MGET);

    /* fetch all account keys and values from storage */
    auto onAccount = [&] (const AccountKey & key,
                          const Account & account)
        {
            string keyStr = key.toString();
            keys.push_back(keyStr);
            fetchCommand.addArg("banker-" + keyStr);
        };
    toSave.forEachAccount(onAccount);

    auto onPhase1Result = [=] (const Redis::Result & result)
        {
            if (!result.ok()) {
                onSaved(BACKEND_ERROR, result.error());
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
                    cerr << "account '" << key
                         << "' is out of sync and will not be saved" << endl;
                    continue;
                }
                Json::Value bankerValue = bankerAccount.toJson();
                bool saveAccount(false);

                Result result = reply[i];
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
                onSaved(DATA_INCONSISTENCY, boost::trim_copy(badAccounts.toString()));
            }
            else if (storeCommands.size() > 1) {
                 storeCommands.push_back(EXEC);
                 
                 auto onPhase2Result = [=] (const Redis::Results & results)
                 {
                     if (results.ok())
                         onSaved(SUCCESS, "");
                     else
                         onSaved(BACKEND_ERROR, results.error());
                 };

                 itl->redis->queueMulti(storeCommands, onPhase2Result, 5.0);
            }
            else {
                onSaved(SUCCESS, "");
            }
        };

    if (keys.size() == 0) {
        /* no account to save */
        onSaved(SUCCESS, "");
        return;
    }

    itl->redis->queue(fetchCommand, onPhase1Result, 5.0);
}

/*****************************************************************************/
/* MASTER BANKER                                                             */
/*****************************************************************************/

MasterBanker::
MasterBanker(std::shared_ptr<ServiceProxies> proxies,
             const string & serviceName)
    : ServiceBase(serviceName, proxies),
      RestServiceEndpoint(proxies->zmqContext),
      saving(false),
      monitorProviderClient(proxies->zmqContext, *this)
{
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
init(const shared_ptr<BankerPersistence> & storage)
{
    this->storage_ = storage;

    loadStateSync();

    addPeriodic("MasterBanker::saveState", 1.0,
                bind(&MasterBanker::saveState, this),
                true /* single threaded */);

    registerServiceProvider(serviceName(), { "rtbBanker" });

    getServices()->config->removePath(serviceName());
    //registerService();
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
                       &Accounts::getAccountSummariesJson,
                       &accounts,
                       true,
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
                       &Accounts::createAccount,
                       &accounts,
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
                       &Accounts::setBudget,
                       &accounts,
                       accountKeyParam,
                       JsonParam<CurrencyPool>("", "amount to set budget to"));

    addRouteSyncReturn(account,
                       "/balance",
                       {"PUT", "POST"},
                       "Transfer budget from the parent such that account's "
                       "balance amount matches the parameter",
                       "Account: Representation of the modified account",
                       [] (const Account & a) { return a.toJson(); },
                       &Accounts::setBalance,
                       &accounts,
                       accountKeyParam,
                       JsonParam<CurrencyPool>("", "amount to set balance to"),
                       RestParamDefault<AccountType>("accountType", "type of account for implicit creation (default no creation)", AT_NONE));
    
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
                       &Accounts::syncFromShadow,
                       &accounts,
                       accountKeyParam,
                       JsonParam<ShadowAccount>("",
                                                "Representation of the shadow account"));

    monitorProviderClient.init(getServices()->config);
}

void
MasterBanker::
start()
{
    RestServiceEndpoint::start();
    monitorProviderClient.start();
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
    monitorProviderClient.shutdown();
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

void
MasterBanker::
onStateSaved(BankerPersistence::PersistenceCallbackStatus status,
             const string & info)
{
    if (status == BankerPersistence::SUCCESS) {
        //cerr << __FUNCTION__
        //     <<  ": banker state saved successfully to backend" << endl;
        lastSavedState = Date::now();
    }
    else if (status == BankerPersistence::DATA_INCONSISTENCY) {
        Json::Value accountKeys = Json::parse(info);
        ExcAssert(accountKeys.type() == Json::arrayValue);
        for (Json::Value jsonKey: accountKeys) {
            ExcAssert(jsonKey.type() == Json::stringValue);
            string keyStr = jsonKey.asString();
            accounts.markAccountOutOfSync(AccountKey(keyStr));
            cerr << __FUNCTION__
                 << ": account '" << keyStr << "' marked out of sync" << endl;
        }
    }
    else if (status == BankerPersistence::BACKEND_ERROR) {
        /* the backend is unavailable */
        cerr << __FUNCTION__ <<  ": " << info << endl;
    }
    else {
        throw ML::Exception("status code is not handled");
    }

    lastSaveStatus = status;

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
              BankerPersistence::PersistenceCallbackStatus status,
              const string & info)
{
    if (status == BankerPersistence::SUCCESS) {
        newAccounts->ensureInterAccountConsistency();
        accounts = *newAccounts;
        cerr << __FUNCTION__ <<  ": successfully loaded accounts" << endl;
    }
    else if (status == BankerPersistence::DATA_INCONSISTENCY) {
        /* something is wrong with the backend data types */
        cerr << __FUNCTION__ <<  ": " << info << endl;
    }
    else if (status == BankerPersistence::BACKEND_ERROR) {
        /* the backend is unavailable */
        cerr << __FUNCTION__ <<  ": " << info << endl;
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

/** MonitorProvider interface */
string
MasterBanker::
getProviderName()
    const
{
    return serviceName();
}

Json::Value
MasterBanker::
getProviderIndicators()
    const
{
    Json::Value value;

    /* MB health check:
       - no error occurred in last save (implying Redis conn is alive) */
    Date now = Date::now();
    bool status(lastSaveStatus == BankerPersistence::SUCCESS);
    value["status"] = status ? "ok" : "failure";

    return value;
}

} // namespace RTBKIT

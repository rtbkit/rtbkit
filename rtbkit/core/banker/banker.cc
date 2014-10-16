/* banker.cc
   Sunil Rottoo, May 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.
   
   Banker functionality for the RTB router.
*/

#include "banker.h"
#include "jml/arch/timers.h"
#include "jml/arch/format.h"
#include "jml/utils/exc_assert.h"
#include "jml/arch/futex.h"
#include <boost/foreach.hpp>


using namespace std;
using namespace ML;


namespace RTBKIT {


/*****************************************************************************/
/* BUDGET CONTROLLER PROXY                                                   */
/*****************************************************************************/

BudgetController::
BudgetController()
{
    // No more than $100k USD in budget as a default limit
    budgetLimits += USD(100000);
}

BudgetController::
~BudgetController()
{
}

void
BudgetController::
addAccount(const AccountKey & account,
           const OnBudgetResult & onResult)
{
    try {
        addAccountSync(account);
        if (onResult)
            onResult(nullptr);
    } catch (...) {
        if (onResult)
            onResult(std::current_exception());
        else throw;
    }
}

void
BudgetController::
addAccountSync(const AccountKey & account)
{
    BankerSyncResult<void> result;
    addAccount(account, result);
    return result.get();
}

//you must override either this method or the next one to avoid an infinite recursion
void
BudgetController::
topupTransfer(const AccountKey & account,
                        CurrencyPool amount,
                        const OnBudgetResult & onResult)
{
    try {
        topupTransferSync(account, amount);
        if (onResult)
            onResult(nullptr);
    } catch (...) {
        if (onResult)
            onResult(std::current_exception());
        else throw;
    }
}

//you must override either this method or the previous one to avoid an infinite recursion
void
BudgetController::
topupTransferSync(const AccountKey & account,
                            CurrencyPool amount)
{
    BankerSyncResult<void> result;
    topupTransfer(account, amount, result);
    return result.get();
}

void
BudgetController::
setBudget(const std::string & topLevelAccount,
          CurrencyPool amount,
          const OnBudgetResult & onResult)
{
    try {
        setBudgetSync(topLevelAccount, amount);
        if (onResult)
            onResult(nullptr);
    } catch (...) {
        if (onResult)
            onResult(std::current_exception());
        else throw;
    }
}
    
void
BudgetController::
setBudgetSync(const std::string & topLevelAccount,
              CurrencyPool amount)
{
    BankerSyncResult<void> result;
    setBudget(topLevelAccount, amount, result);
    return result.get();
}

void
BudgetController::
addBudget(const std::string & topLevelAccount,
          CurrencyPool amount,
          const OnBudgetResult & onResult)
{
    try {
        addBudgetSync(topLevelAccount, amount);
        if (onResult)
            onResult(nullptr);
    } catch (...) {
        if (onResult)
            onResult(std::current_exception());
        else throw;
    }
}

void
BudgetController::
addBudgetSync(const std::string & topLevelAccount,
              CurrencyPool amount)
{
    BankerSyncResult<void> result;
    addBudget(topLevelAccount, amount, result);
    return result.get();
}


/*****************************************************************************/
/* ACCOUNTANT                                                                */
/*****************************************************************************/

std::vector<AccountKey>
Accountant::
getAccountListSync(const AccountKey & prefix,
                   int depth)
{
    BankerSyncResult<std::vector<AccountKey> > result;
    getAccountList(prefix, depth, result);
    return result.get();
}

void
Accountant::
getAccountList(const AccountKey & prefix,
               int depth,
               std::function<void (std::exception_ptr,
                                   std::vector<AccountKey> &&)> onResult)
{
    std::vector<AccountKey> res;
    std::exception_ptr p;
    try {
        res = getAccountListSync(prefix, depth);
    } catch (...) {
        if (onResult)
            p = std::current_exception();
        else throw;
    }
    if (onResult)
        onResult(p, std::move(res));
}

AccountSummary
Accountant::
getAccountSummarySync(const AccountKey & account, int depth)
{
    BankerSyncResult<AccountSummary> result;
    getAccountSummary(account, depth, result);
    return result.get();
}

void
Accountant::
getAccountSummary(const AccountKey & account,
                 int depth,
                 std::function<void (std::exception_ptr,
                                     AccountSummary &&)> onResult)
{
    AccountSummary res;
    std::exception_ptr p;
    try {
        res = getAccountSummarySync(account, depth);
    } catch (...) {
        if (onResult)
            p = std::current_exception();
        else throw;
    }
    if (onResult)
        onResult(p, std::move(res));
}

Account
Accountant::
getAccountSync(const AccountKey & account)
{
    BankerSyncResult<Account> result;
    getAccount(account, result);
    return result.get();
}

void
Accountant::
getAccount(const AccountKey & account,
           std::function<void (std::exception_ptr,
                               Account &&)> onResult)
{
    Account res;
    std::exception_ptr p;
    try {
        res = getAccountSync(account);
    } catch (...) {
        if (onResult)
            p = std::current_exception();
        else throw;
    }
    if (onResult)
        onResult(p, std::move(res));
}


/*****************************************************************************/
/* BANKER                                                                    */
/*****************************************************************************/

Banker::
~Banker()
{
}

ShadowAccount
Banker::
addSpendAccountSync(const AccountKey & account, CurrencyPool accountFloat)
{
    BankerSyncResult<ShadowAccount> result;
    addSpendAccount(account, accountFloat, result);
    return result.get();
}

void
Banker::
addSpendAccount(const AccountKey & account, CurrencyPool accountFloat,
                std::function<void (std::exception_ptr, ShadowAccount&&)> onDone)
{
    ShadowAccount res;
    std::exception_ptr p;
    try {
        res = addSpendAccountSync(account, accountFloat);
    } catch (...) {
        if (onDone)
            p = std::current_exception();
        else throw;
    }
    if (onDone)
        onDone(p, std::move(res));
}

/*****************************************************************************/
/* BANKER EXCEPTION                                                           */
/*****************************************************************************/

//----------------------------------------------------------------------
string BankerException::errorToString(const BankerError &cond)
{
    if (cond == BankerError::INVALID_CAMPAIGN)
        return "INVALID_CAMPAIGN";
    if (cond == BankerError::CAMPAIGN_NOT_FOUND)
        return "CAMPAIGN_NOT_FOUND";
    else if (cond == BankerError::INVALID_STRATEGY)
        return "INVALID_STRATEGY";
    else if (cond == BankerError::STRATEGY_NOT_FOUND)
        return "STRATEGY_NOT_FOUND";
    else if (cond == BankerError::INSUFFICIENT_FUNDS)
        return "INSUFFICIENT_FUNDS";
    else if (cond == BankerError::LOWER_THAN_TRANSFERRED)
        return "LOWER_THAN_TRANSFERRED";
    else if (cond == BankerError::EXCEEDS_MAX)
        return "EXCEEDS_MAX";
    else if (cond == BankerError::DATABASE_ERROR)
        return "DATABASE_ERROR";
    else if (cond == BankerError::ACCOUNTING_MISMATCH)
        return "ACCOUNTING_MISMATCH";
    else
        return "UNKNOWN_ERROR";
}
//---------------------------------------------------------------------------------
BankerException::BankerException(const std::string &msg,
        const BankerError &error) :
        ML::Exception(msg + BankerException::errorToString(error)), error_(
                error)
{
}

} // namespace RTBKIT

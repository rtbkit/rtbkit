/* account.cc
   Jeremy Barnes, 16 November 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Implementation of the Account class.
*/

#include "account.h"
#include "banker.h"

using namespace std;
using namespace ML;


namespace RTBKIT {


/*****************************************************************************/
/* ACCOUNT TYPE                                                              */
/*****************************************************************************/

AccountType AccountTypeFromString(const std::string & param) {
    return restDecode(param, NULL);
}

const std::string AccountTypeToString(enum AccountType type) {
    switch (type) {
    case AT_NONE:
        return "none";
        break;
    case AT_BUDGET:
        return "budget";
        break;
    case AT_SPEND:
        return "spend";
        break;
    default:
        throw ML::Exception("unknown account type %d", type);
    }
}

/*****************************************************************************/
/* ACCOUNT KEY                                                               */
/*****************************************************************************/


/*****************************************************************************/
/* ACCOUNT                                                                   */
/*****************************************************************************/
void
Account::
setBudget(const CurrencyPool & newBudget)
{
    /* totalBudget = budgetIncreases - budgetDecreases */
    CurrencyPool totalBudget = budgetIncreases - budgetDecreases;
    /* extraBudget = amount to add to members to obtain new total budget */
    CurrencyPool extraBudget = newBudget - totalBudget;

#if 0
    cerr << "totalBudget: " << totalBudget << endl;
    cerr << "balance: " << balance
         << "; extraBudget: " << extraBudget << endl;
#endif
    auto preview = balance + extraBudget;

    if(!preview.isNonNegative()) {
        extraBudget = CurrencyPool() - balance;
    }

    ExcAssert((balance + extraBudget).isNonNegative());

    if (extraBudget.isNonNegative()) {
        budgetIncreases += extraBudget;
    }
    else {
        budgetDecreases -= extraBudget;
    }
    balance += extraBudget;
   
    status = ACTIVE;

    checkInvariants();
}

void
Account::
addAdjustment(const CurrencyPool & newAdjustment)
{
    if (newAdjustment.isNonNegative()) {
        adjustmentsIn += newAdjustment;
    }
    else {
        adjustmentsOut -= newAdjustment;
    }

    balance += newAdjustment;

    checkInvariants();
}

CurrencyPool
Account::
importSpend(const CurrencyPool & spend)
{
    ExcAssert(type == AT_SPEND);
    ExcAssert(spend.isNonNegative());

    checkInvariants();

    /* make sure we have enough balance to spend */
    CurrencyPool newBalance = balance - spend;
    /* FIXME: we might choose a different policy than just crashing here */
    ExcAssert(newBalance.isNonNegative());

    balance = newBalance;
    spent += spend;

    checkInvariants();

    return spend;
}

CurrencyPool
Account::
getNetBudget()
    const
{
    return (budgetIncreases - budgetDecreases
            + allocatedIn - allocatedOut
            + recycledIn - recycledOut
            + adjustmentsIn - adjustmentsOut);
}

std::ostream & operator << (std::ostream & stream, const Account & account)
{
    std::set<CurrencyCode> currencies;

    auto addCurrencies = [&] (const CurrencyPool & c)
        {
            for (auto a: c.currencyAmounts) {
                if (a) currencies.insert(a.currencyCode);
            }
        };

    addCurrencies(account.budgetIncreases);
    addCurrencies(account.budgetDecreases);
    addCurrencies(account.recycledIn);
    addCurrencies(account.recycledOut);
    addCurrencies(account.commitmentsMade);
    addCurrencies(account.commitmentsRetired);
    addCurrencies(account.spent);
    addCurrencies(account.balance);
    addCurrencies(account.adjustmentsIn);
    addCurrencies(account.adjustmentsOut);
    addCurrencies(account.allocatedIn);
    addCurrencies(account.allocatedOut);
    
    for (const auto & li: account.lineItems.entries)
        addCurrencies(li.second);

    stream << endl << "                    ";
    for (auto c: currencies) {
        string s = Amount::getCurrencyStr(c);
        int spaces = 21 - s.size();
        int before = spaces / 2;
        int after = spaces - before;
        stream << ' '
               << string(before, '-') << ' ' << s << ' ' << string(after, '-');
    }
    stream << endl;
    
    stream << "                    ";
    for (unsigned i = 0;  i < currencies.size();  ++i)
        stream << "      credit       debit";
    stream << endl;

    auto printCurrency = [&] (const char * label, 
                              const CurrencyPool & credit,
                              const CurrencyPool & debit)
        {
            if (credit.isZero() && debit.isZero())
                return;
            
            stream << ML::format("%-20s", label);

            auto printAmount = [&] (const Amount & a)
            {
                if (!a.isZero())
                    stream << ML::format("%12lld", (long long)a.value);
                else stream << "            ";
            };
            
            for (auto c: currencies) {
                printAmount(credit.getAvailable(c));
                printAmount(debit.getAvailable(c));
            }
            
            stream << endl;
        };
    
    CurrencyPool z;

    printCurrency("  budget in/out",    account.budgetIncreases,
                                        account.budgetDecreases);
    printCurrency("  none/spent",       CurrencyPool(), account.spent);
    printCurrency("  recycled in/out",  account.recycledIn, account.recycledOut);
    printCurrency("  allocated in/out", account.allocatedIn, account.allocatedOut);
    printCurrency("  commit ret/made",  account.commitmentsRetired, account.commitmentsMade);
    printCurrency("  adj in/out",       account.adjustmentsIn, account.adjustmentsOut);
    stream << "--------------------------------------------" << endl;
    printCurrency("  balance", account.balance, z);
    stream << endl;

    auto printLineItems = [&] (const std::string & name,
                               const LineItems & lineItems)
        {
            if (!lineItems.isZero()) {
                stream << name << endl;
                for (const auto & li: lineItems.entries) {
                    printCurrency(li.first.c_str(), z, li.second);
                }
            }
        };

    printLineItems("Spend Line Items:", account.lineItems);
    printLineItems("Adjustment Line Items:", account.adjustmentLineItems);

    return stream;
}


/*****************************************************************************/
/* SHADOW ACCOUNT                                                            */
/*****************************************************************************/

void
ShadowAccount::
logBidEvents(const Datacratic::EventRecorder & eventRecorder,
             const string & accountKey)
{
    eventRecorder.recordLevel(attachedBids,
                              "banker.accounts." + accountKey + ".attachedBids");
    attachedBids = 0;

    eventRecorder.recordLevel(detachedBids,
                              "banker.accounts." + accountKey + ".detachedBids");
    detachedBids = 0;

    eventRecorder.recordLevel(commitments.size(),
                              "banker.accounts." + accountKey + ".pendingCommitments");

    Date now = Date::now();
    lastExpiredCommitments = 0;
    for (auto & it: commitments) {
        Commitment & commitment = it.second;
        if (now >= commitment.timestamp.plusSeconds(15.0)) {
            lastExpiredCommitments++;
        }
    }
    eventRecorder.recordLevel(lastExpiredCommitments,
                              "banker.accounts." + accountKey + ".expiredCommitments");
}

std::ostream &
operator << (std::ostream & stream, const ShadowAccount & account)
{
    std::set<CurrencyCode> currencies;

    auto addCurrencies = [&] (const CurrencyPool & c)
        {
            for (auto a: c.currencyAmounts) {
                if (a) currencies.insert(a.currencyCode);
            }
        };

    addCurrencies(account.netBudget);
    addCurrencies(account.commitmentsMade);
    addCurrencies(account.commitmentsRetired);
    addCurrencies(account.spent);
    addCurrencies(account.balance);
    
    for (const auto & li: account.lineItems.entries)
        addCurrencies(li.second);

    stream << endl << "                    ";
    for (auto c: currencies) {
        string s = Amount::getCurrencyStr(c);
        int spaces = 21 - s.size();
        int before = spaces / 2;
        int after = spaces - before;
        stream << ' '
               << string(before, '-') << ' ' << s << ' ' << string(after, '-');
    }
    stream << endl;
    
    stream << "                    ";
    for (unsigned i = 0;  i < currencies.size();  ++i)
        stream << "      credit       debit";
    stream << endl;

    auto printCurrency = [&] (const char * label, 
                              const CurrencyPool & credit,
                              const CurrencyPool & debit)
        {
            if (credit.isZero() && debit.isZero())
                return;
            
            stream << ML::format("%-20s", label);

            auto printAmount = [&] (const Amount & a)
            {
                if (!a.isZero())
                    stream << ML::format("%12lld", (long long)a.value);
                else stream << "            ";
            };
            
            for (auto c: currencies) {
                printAmount(credit.getAvailable(c));
                printAmount(debit.getAvailable(c));
            }
            
            stream << endl;
        };
    
    CurrencyPool z;

    printCurrency("  netBudget/spent",     account.netBudget, account.spent);
    printCurrency("  commit ret/made",  account.commitmentsRetired, account.commitmentsMade);
    stream << "--------------------------------------------" << endl;
    printCurrency("  balance", account.balance, z);
    stream << endl;

    auto printLineItems = [&] (const std::string & name,
                               const LineItems & lineItems)
        {
            if (!lineItems.isZero()) {
                stream << name << endl;
                for (const auto & li: lineItems.entries) {
                    printCurrency(li.first.c_str(), z, li.second);
                }
            }
        };

    printLineItems("Spend Line Items:", account.lineItems);

    return stream;
}

/*****************************************************************************/
/* SHADOW ACCOUNTS                                                           */
/*****************************************************************************/
void
ShadowAccounts::
logBidEvents(const Datacratic::EventRecorder & eventRecorder)
{
    Guard guard(lock);

    uint32_t attachedBids(0), detachedBids(0), commitments(0), expired(0);

    for (auto & it: accounts) {
        ShadowAccount & account = it.second;
        attachedBids += account.attachedBids;
        detachedBids += account.detachedBids;
        commitments += account.commitments.size();
        account.logBidEvents(eventRecorder, it.first.toString('.'));
        expired += account.lastExpiredCommitments;
    }

    eventRecorder.recordLevel(attachedBids,
                              "banker.total.attachedBids");
    eventRecorder.recordLevel(detachedBids,
                              "banker.total.detachedBids");
    eventRecorder.recordLevel(commitments,
                              "banker.total.pendingCommitments");
    eventRecorder.recordLevel(expired,
                              "banker.total.expiredCommitments");
}

/*****************************************************************************/
/* ACCOUNTS                                                                  */
/*****************************************************************************/

void
Accounts::
ensureInterAccountConsistency()
{
    Guard guard(lock);

    for (const auto & it: accounts) {
        if (it.first.size() == 1) {
            if (!checkBudgetConsistencyImpl(it.first, -1, 0)) {
                // cerr << "budget of account " << it.first
                //      << " is not consistent\n";
                inconsistentAccounts.insert(it.first);
            }
            CurrencyPool recycledInUp, recycledOutUp, nullPool;
            getRecycledUp(it.first, recycledInUp, recycledOutUp);            
            if (recycledInUp != nullPool) {
                cerr << "upward recycledIn of account " << it.first
                     << " is not null: " << recycledInUp
                     << "\n";
            }
            if (recycledOutUp != nullPool) {
                cerr << "upward recycledOut of account " << it.first
                     << " is not null: " << recycledOutUp
                     << "\n";
            }
        }
    }
}

bool
Accounts::
checkBudgetConsistency(const AccountKey & accountKey, int maxRecursion)
    const
{
    Guard guard(lock);

    ExcAssertEqual(accountKey.size(), 1);

    return checkBudgetConsistencyImpl(accountKey, maxRecursion, 0);
}

bool
Accounts::
checkBudgetConsistencyImpl(const AccountKey & accountKey, int maxRecursion,
                           int level)
    const
{
    const AccountInfo & account = getAccountImpl(accountKey);
    CurrencyPool sumBudgetInc;

    for (const AccountKey & childKey: account.children) {
        const Account & childAccount = getAccountImpl(childKey);
        sumBudgetInc += childAccount.budgetIncreases;
    }

    if (account.allocatedOut != sumBudgetInc) {
        cerr << "budget of account " << accountKey
             << " is not consistent:\n  "
             << account.allocatedOut
             << " != " << sumBudgetInc
             << " (delta = "
             << (account.allocatedOut - sumBudgetInc)
             << ")\n";
        return false;
    }

    if (maxRecursion == -1 || level < maxRecursion) {
        for (const AccountKey & childKey: account.children) {
            if (!checkBudgetConsistencyImpl(childKey,
                                            maxRecursion, level + 1))
                return false;
        }
    }

    return true;
}

void
Accounts::
getRecycledUp(const AccountKey & accountKey,
              CurrencyPool & recycledInUp,
              CurrencyPool & recycledOutUp)
    const
{
    CurrencyPool sumSubRecycledIn, sumSubRecycledOut;
 
    const AccountInfo & account = getAccountImpl(accountKey);
 
    for (const AccountKey & childKey: account.children) {
        CurrencyPool subRecycledIn, subRecycledOut;
        getRecycledUp(childKey, subRecycledIn, subRecycledOut);
        sumSubRecycledIn += subRecycledIn;
        sumSubRecycledOut += subRecycledOut;
    }
 
    recycledInUp = account.recycledIn - sumSubRecycledOut;
    recycledOutUp = account.recycledOut - sumSubRecycledIn;
}
 
} // namespace RTBKIT

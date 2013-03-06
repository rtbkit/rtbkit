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
        throw ML::Exception("unknown account type " + type);
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
    cerr << "available: " << available
         << "; extraBudget: " << extraBudget << endl;
#endif
    ExcAssert((available + extraBudget).isNonNegative());

    if (extraBudget.isNonNegative()) {
        budgetIncreases += extraBudget;
    }
    else {
        budgetDecreases -= extraBudget;
    }
    available += extraBudget;

    checkInvariants();
}

CurrencyPool
Account::
importSpend(const CurrencyPool & spend)
{
    ExcAssert(type == AT_SPEND);
    ExcAssert(spend.isNonNegative());

    checkInvariants();

    /* make sure we have enough available to spend */
    CurrencyPool newAvailable = available - spend;
    /* FIXME: we might choose a different policy than just crashing here */
    ExcAssert(newAvailable.isNonNegative());

    available = newAvailable;
    spent += spend;

    checkInvariants();

    return spend;
}

CurrencyPool
Account::
getBudget()
    const
{
    return (budgetIncreases - budgetDecreases);
            // + allocatedIn - allocatedOut
            // + recycledIn - recycledOut
            // + adjustmentsIn - adjustmentsOut);
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
    addCurrencies(account.available);
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
    printCurrency("  available", account.available, z);
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
    addCurrencies(account.available);
    
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
    printCurrency("  available", account.available, z);
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
                              "banker.total.setachedBids");
    eventRecorder.recordLevel(commitments,
                              "banker.total.pendingCommitments");
    eventRecorder.recordLevel(expired,
                              "banker.total.expiredCommitments");
}

/*****************************************************************************/
/* ACCOUNTSIMPLESUMMARY                                                      */
/*****************************************************************************/

void
AccountSimpleSummary::
dump(ostream & stream, int indent, const std::string & name) const
{
    stream << std::string(indent, ' ')
           << name
           << " b:" << budget
           << " s:" << spent
           << " a:" << available
           << std::endl;
}

Json::Value
AccountSimpleSummary::
toJson() const
{
    Json::Value result(Json::objectValue);
    result["md"]["objectType"] = "AccountSimpleSummary";
    result["md"]["version"] = 1;
    result["budget"] = budget.toJson();
    result["spent"] = spent.toJson();
    result["available"] = available.toJson();
    result["inFlight"] = inFlight.toJson();

    return result;
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
        if (it.first.size() == 1
            && !checkInterAccountConsistencyImpl(it.first)) {
            inconsistentAccounts.insert(it.first);
        }
    }
}

bool
Accounts::
checkInterAccountConsistency(const AccountKey & accountKey)
    const
{
    Guard guard(lock);

    ExcAssertEqual(accountKey.size(), 1);

    return checkInterAccountConsistencyImpl(accountKey);
}

bool
Accounts::
checkInterAccountConsistencyImpl(const AccountKey & accountKey)
    const
{
    bool inconsistent(false);
    const AccountInfo & account = getAccountImpl(accountKey);

    if (account.type == AT_BUDGET && account.children.size() > 0) {
        CurrencyPool sumRecycledIn, sumBudgetInc, sumRecycledOut;

        // cerr << indentStr << "testing " << accountKey.toString() << endl;

        // cerr << "computing sums:" << endl;
        for (const AccountKey & childKey: account.children) {
            const Account & childAccount = getAccountImpl(childKey);
            // cerr << "  account " << childKey.toString()
            //      << " " << &childAccount
            //      << endl;
            // cerr << "    recycledIn: " << childAccount.recycledIn
            //      << "    recycledOut: " << childAccount.recycledOut
            //      << "    budgetIn: " << childAccount.budgetIncreases
            //      << endl;
            sumRecycledIn += childAccount.recycledIn;
            sumBudgetInc += childAccount.budgetIncreases;
            sumRecycledOut += childAccount.recycledOut;
        }
        // cerr << "totals:" << endl
        //      << "  sumRecycledIn: " << sumRecycledIn
        //      << "  sumRecycledOut: " << sumRecycledOut
        //      << "  sumBudgetInc: " << sumBudgetInc
        //      << endl;

        if (account.recycledOut != sumRecycledIn)
        {
            inconsistent = true;
            cerr << "* failure in " << accountKey.toString() << "\n"
                 << "- sum(recycledIn) !="
                 << " recycledOut"
                 << "\n"
                 << sumRecycledIn.toString()
                 << " != "
                 << account.recycledOut.toString()
                 << "\n"
                 << "delta = "
                 << (sumRecycledIn - account.recycledOut).toString()
                 << "\n";
        }

        if (account.recycledIn != sumRecycledOut)
        {
            if (!inconsistent) {
                inconsistent = true;
                cerr << "* failure in " << accountKey.toString() << "\n";
            }
            cerr << "- sum(recycledOut) !="
                 << " recycledIn"
                 << "\n"
                 << sumRecycledOut.toString()
                 << " != "
                 << account.recycledIn.toString()
                 << "\n"
                 << "delta = "
                 << (sumRecycledOut - account.recycledIn).toString()
                 << "\n";
        }

        if (account.allocatedOut != sumBudgetInc)
        {
            if (!inconsistent) {
                inconsistent = true;
                cerr << "* failure in " << accountKey.toString() << "\n";
            }
            cerr << "- sum(budgetIncreases) !="
                 << " recycledOut"
                 << "\n"
                 << sumBudgetInc.toString()
                 << " != "
                 << account.allocatedOut.toString()
                 << "\n"
                 << "delta = "
                 << (sumBudgetInc - account.allocatedOut).toString()
                 << "\n";
        }

        if (inconsistent) {
            const AccountSummary & summary
                = getAccountSummaryImpl(accountKey, 0, 255);
            cerr << summary << "\n";
        }

        for (const AccountKey & childKey: account.children) {
            inconsistent = (!checkInterAccountConsistencyImpl(childKey)
                            || inconsistent);
        }
    }

    return !inconsistent;
}

} // namespace RTBKIT

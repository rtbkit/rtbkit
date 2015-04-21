/* account.h                                                       -*- C++ -*-
   Jeremy Barnes, 16 November 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

*/

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <unordered_set>
#include "rtbkit/common/currency.h"
#include "rtbkit/common/account_key.h"
#include "soa/types/date.h"
#include "jml/utils/string_functions.h"
#include <mutex>
#include <thread>
#include "jml/arch/spinlock.h"

namespace Datacratic {
    struct EventRecorder;
}

namespace RTBKIT {
using namespace Datacratic;

struct Account;

std::ostream & operator << (std::ostream & stream, const Account & account);

struct ShadowAccount;

std::ostream &
operator << (std::ostream & stream, const ShadowAccount & account);


/*****************************************************************************/
/* ACCOUNT TYPE                                                              */
/*****************************************************************************/

enum AccountType {
    AT_NONE,
    AT_BUDGET,      ///< Budgeting account
    AT_SPEND        ///< Spend tracking account; leaf
};

inline AccountType restDecode(const std::string & param, AccountType *)
{
    if (param == "none")
        return AT_NONE;
    else if (param == "budget")
        return AT_BUDGET;
    else if (param == "spend")
        return AT_SPEND;
    else throw ML::Exception("unknown account type " + param);
}

extern AccountType AccountTypeFromString(const std::string & param);
extern const std::string AccountTypeToString(enum AccountType type);

/*****************************************************************************/
/* ACCOUNT                                                                   */
/*****************************************************************************/

/** This is the basic unit in which spend is tracked. */

struct Account {
    Account()
        : type(AT_NONE), status(ACTIVE)
    {
    }

    AccountType type;

    //mutable Date lastAccess;
    //Date lastModification;

    // On the Credit side...
    CurrencyPool budgetIncreases;    ///< Transferred in from parent
    CurrencyPool budgetDecreases;    ///< Transferred out (in parent only)
    CurrencyPool recycledIn;         ///< Unspent transferred back in
    CurrencyPool allocatedIn;
    CurrencyPool commitmentsRetired; ///< Money not spent
    CurrencyPool adjustmentsIn;      ///< Money transferred in by adjustments

    // On the Debit side...
    CurrencyPool recycledOut;
    CurrencyPool allocatedOut;
    CurrencyPool commitmentsMade;
    CurrencyPool adjustmentsOut;  ///< Money transferred out by adjustments
    CurrencyPool spent;           ///< Actually spent

    CurrencyPool balance;       ///< Balance to be spent

    // Extra information tracked, but not used in any calculations
    LineItems lineItems;            ///< Spent line items
    LineItems adjustmentLineItems;  ///< Adjustment line items

    // Invariant: sum(Credit Side) = sum(Debit Side)

    enum Status {CLOSED, ACTIVE};
    Status status;

public:
    bool isSameOrPastVersion(const Account & otherAccount) const
    {
        /* All the amounts in the storage accounts must have a counterpart in
         * the banker accounts and their value must be inferior or equal to
         * the corresponding amounts in the banker. */
        return (budgetIncreases.isSameOrPastVersion(otherAccount.budgetIncreases)
                && budgetDecreases.isSameOrPastVersion(otherAccount.budgetDecreases)
                && recycledIn.isSameOrPastVersion(otherAccount.recycledIn)
                && allocatedIn.isSameOrPastVersion(otherAccount.allocatedIn)
                && commitmentsRetired.isSameOrPastVersion(otherAccount.commitmentsRetired)
                && adjustmentsIn.isSameOrPastVersion(otherAccount.adjustmentsIn)

                && recycledOut.isSameOrPastVersion(otherAccount.recycledOut)
                && allocatedOut.isSameOrPastVersion(otherAccount.allocatedOut)
                && commitmentsMade.isSameOrPastVersion(otherAccount.commitmentsMade)
                && adjustmentsOut.isSameOrPastVersion(otherAccount.adjustmentsOut)
                && spent.isSameOrPastVersion(otherAccount.spent));
    }

    Json::Value toJson() const
    {
        // checkInvariants();

        Json::Value result(Json::objectValue);
        result["md"]["objectType"] = "Account";
        result["md"]["version"] = 1;
        result["type"] = AccountTypeToString(type);
        result["budgetIncreases"] = budgetIncreases.toJson();
        result["budgetDecreases"] = budgetDecreases.toJson();
        result["spent"] = spent.toJson();
        result["recycledIn"] = recycledIn.toJson();
        result["recycledOut"] = recycledOut.toJson();
        result["allocatedIn"] = allocatedIn.toJson();
        result["allocatedOut"] = allocatedOut.toJson();
        result["commitmentsMade"] = commitmentsMade.toJson();
        result["commitmentsRetired"] = commitmentsRetired.toJson();
        result["adjustmentsIn"] = adjustmentsIn.toJson();
        result["adjustmentsOut"] = adjustmentsOut.toJson();
        result["lineItems"] = lineItems.toJson();
        result["adjustmentLineItems"] = adjustmentLineItems.toJson();
        switch (status) {
        case ACTIVE:
            result["status"] = "active";
            break;
        case CLOSED:
            result["status"] = "closed";
            break; 
        default: 
            result["status"] = "active";
        }
        return result;
    }

    static const Account fromJson(const Json::Value & json)
    {
        Account result;
        ExcAssertEqual(json["md"]["objectType"].asString(), "Account");
        ExcAssertEqual(json["md"]["version"].asInt(), 1);

        result.type = AccountTypeFromString(json["type"].asString());
        if (json.isMember("budget")) {
            result.budgetIncreases = CurrencyPool::fromJson(json["budget"]);
            result.budgetDecreases = CurrencyPool::fromJson(json["adjustmentsOut"]);
            result.adjustmentsOut = CurrencyPool();
        }
        else {
            result.budgetIncreases = CurrencyPool::fromJson(json["budgetIncreases"]);
            result.budgetDecreases = CurrencyPool::fromJson(json["budgetDecreases"]);
            result.adjustmentsOut = CurrencyPool::fromJson(json["adjustmentsOut"]);
        }
        
        if (json.isMember("status")) {
            std::string s = json["status"].asString();
            if (s == "active") 
                result.status = ACTIVE;
            else if (s == "closed")
                result.status = CLOSED;
            else
                result.status = ACTIVE;
        } else {
            result.status = ACTIVE;
        }

        result.spent = CurrencyPool::fromJson(json["spent"]);
        result.recycledIn = CurrencyPool::fromJson(json["recycledIn"]);
        result.recycledOut = CurrencyPool::fromJson(json["recycledOut"]);
        result.allocatedIn = CurrencyPool::fromJson(json["allocatedIn"]);
        result.allocatedOut = CurrencyPool::fromJson(json["allocatedOut"]);
        result.commitmentsMade = CurrencyPool::fromJson(json["commitmentsMade"]);
        result.commitmentsRetired = CurrencyPool::fromJson(json["commitmentsRetired"]);

        /* Note: adjustmentsIn is a credit value, ...Out is a debit value */
        result.adjustmentsIn = CurrencyPool::fromJson(json["adjustmentsIn"]);
        result.lineItems = LineItems::fromJson(json["lineItems"]);
        result.adjustmentLineItems = LineItems::fromJson(json["adjustmentLineItems"]);

        result.balance = ((result.budgetIncreases
                             + result.recycledIn
                             + result.commitmentsRetired
                             + result.adjustmentsIn
                             + result.allocatedIn)
                            - (result.budgetDecreases
                               + result.recycledOut
                               + result.commitmentsMade
                               + result.spent
                               + result.adjustmentsOut
                               + result.balance
                               + result.allocatedOut));

        result.checkInvariants();

        return result;
    }
    
    /*************************************************************************/
    /* DERIVED QUANTITIES                                                    */
    /*************************************************************************/

    /** Return the amount which is balance to be recycled. */

    CurrencyPool getRecycledAvail() const
    {
        return (recycledIn - recycledOut).nonNegative();
    }

    /** Returns the budget what was not transferred from or to other accounts.
     */

    CurrencyPool getNetBudget() const;

    /*************************************************************************/
    /* INVARIANTS                                                            */
    /*************************************************************************/
    
    void checkInvariants(const char * whereFrom = "") const
    {
        try {

            // Everything but balance must be positive
            ExcAssert(budgetIncreases.isNonNegative());
            ExcAssert(budgetDecreases.isNonNegative());
            ExcAssert(recycledIn.isNonNegative());
            ExcAssert(recycledOut.isNonNegative());
            ExcAssert(commitmentsRetired.isNonNegative());
            ExcAssert(recycledOut.isNonNegative());
            ExcAssert(commitmentsMade.isNonNegative());
            ExcAssert(spent.isNonNegative());
            ExcAssert(adjustmentsIn.isNonNegative());
            ExcAssert(adjustmentsOut.isNonNegative());
            ExcAssert(allocatedIn.isNonNegative());
            ExcAssert(allocatedOut.isNonNegative());

            // Credit and debit sides must balance out
            CurrencyPool credit = (budgetIncreases + recycledIn + commitmentsRetired
                                   + adjustmentsIn + allocatedIn);
            CurrencyPool debit = (budgetDecreases + recycledOut + commitmentsMade + spent
                                  + adjustmentsOut + balance
                                  + allocatedOut);
            ExcAssertEqual(credit, debit);
        } catch (...) {
            using namespace std;
            cerr << "error on account " << *this << " checking invariants at "
                 << whereFrom << endl;
            throw;
        }
        //...
    }

    /*************************************************************************/
    /* TRANSFER OPERATIONS (ONE SIDED)                                       */
    /*************************************************************************/

    /* These operations need to be paired with a corresponding operation on
       the other side.
    */

    /** Recuperate everything that can safely be removed from the account,
        and return the amount freed.
    */
    CurrencyPool recuperate()
    {
        auto result = balance;
        recycledOut += balance;
        balance.clear();

        checkInvariants();

        return result;
    }
    
    /** Take some budget that had been recuperated from somewhere else and
        add it in.
    */
    void recycle(const CurrencyPool & recuperated)
    {
        ExcAssert(recuperated.isNonNegative());

        recycledIn += recuperated;
        balance += recuperated;

        checkInvariants();
    }

    /** Set the budget to the given amount.  It will adjust the balance
        amount to match the new level.
    */
    void setBudget(const CurrencyPool & newBudget);

    /** Set the balance budget to the given level.  This can either
        transfer money out of or into the account.
        
     */
    CurrencyPool setBalance(Account & parentAccount,
                            const CurrencyPool & newBalance)
    {
        checkInvariants("entry to setBalance");

        auto before = *this;
        auto parentBefore = parentAccount;

        CurrencyPool requiredTotal = newBalance - balance;

        // Some amount needs to be transferred in, and some amount out
        CurrencyPool requiredIn = requiredTotal.nonNegative();
        CurrencyPool requiredOut = requiredIn - requiredTotal;

        ExcAssert(requiredIn.isNonNegative());
        ExcAssert(requiredOut.isNonNegative());

        CurrencyPool toTransfer = parentAccount.balance.limit(requiredIn);

        using namespace std;

        bool debug = false;

        // First take it from the recycled...
        CurrencyPool parentRecycledAvail
            = parentAccount.getRecycledAvail();
        CurrencyPool fromRecycled = parentRecycledAvail.limit(toTransfer);
        CurrencyPool toRecycled = requiredOut;

        if (debug) {
            cerr << "newBalance = " << newBalance << endl;
            cerr << "balance = " << balance << endl;
            cerr << "requiredTotal = " << requiredTotal << endl;
            cerr << "requiredIn = " << requiredIn << endl;
            cerr << "requiredOut = " << requiredOut << endl;
            cerr << "toTransfer = " << toTransfer << endl;
            cerr << "parentRecycledAvail = " << parentRecycledAvail << endl;
            cerr << "fromRecycled = " << fromRecycled << endl;
        }

        // And then as a commitment
        CurrencyPool fromBudget = toTransfer - fromRecycled;

        if (debug) 
            cerr << "fromBudget = " << fromBudget << endl;

        // Take from parent recycled
        parentAccount.recycledOut += fromRecycled;
        parentAccount.balance -= fromRecycled;
        recycledIn += fromRecycled;
        balance += fromRecycled;
        
        // Give back to budget
        parentAccount.allocatedOut += fromBudget;
        parentAccount.balance -= fromBudget;
        budgetIncreases += fromBudget;
        balance += fromBudget;

        // Give to parent recycled
        parentAccount.recycledIn += toRecycled;
        parentAccount.balance += toRecycled;
        recycledOut += toRecycled;
        balance -= toRecycled;

        try {
            checkInvariants("exiting from setBalance");
            parentAccount.checkInvariants("parent check invariants");
        } catch (...) {
            cerr << "before: " << before << endl;
            cerr << "after:  " << *this << endl;

            cerr << "parent before: " << parentBefore << endl;
            cerr << "parent after:  " << parentAccount << endl;
            
            cerr << "newBalance = " << newBalance << endl;
            cerr << "balance = " << balance << endl;
            cerr << "requiredTotal = " << requiredTotal << endl;
            cerr << "requiredIn = " << requiredIn << endl;
            cerr << "requiredOut = " << requiredOut << endl;
            cerr << "toTransfer = " << toTransfer << endl;
            cerr << "parentRecycledAvail = " << parentRecycledAvail << endl;
            cerr << "fromRecycled = " << fromRecycled << endl;

            cerr << "fromBudget = " << fromBudget << endl;
            throw;
        }

        return balance;
    }

    /** Increase or decrease the adjustments made to the account
    */
    void addAdjustment(const CurrencyPool & newAdjustment);

    /** (migration helper) Register an expense on a AT_SPEND account.
     */
    CurrencyPool importSpend(const CurrencyPool & spend);

    void recuperateTo(Account & parentAccount)
    {
        CurrencyPool amount = balance.nonNegative();

        recycledOut += amount;
        balance -= amount;

        parentAccount.recycledIn += amount;
        parentAccount.balance += amount;

        checkInvariants("recuperateTo");
    }
};


/*****************************************************************************/
/* SHADOW ACCOUNT                                                            */
/*****************************************************************************/

/** This is an account that can track spend.  It is a shadow of an account
    that lives in the master banker, and only keeps track of a small amount
    of information.
*/

struct ShadowAccount {
    ShadowAccount()
        : status(Account::ACTIVE), attachedBids(0), detachedBids(0)
        {}

    Account::Status status;

    // credit
    CurrencyPool netBudget;          ///< net of fields not mentioned here
    CurrencyPool commitmentsRetired;

    // debit
    CurrencyPool commitmentsMade;
    CurrencyPool spent;

    CurrencyPool balance;  /// DERIVED; debit - credit

    LineItems lineItems;  ///< Line items for spend

    struct Commitment {
        Commitment(Amount amount, Date timestamp)
            : amount(amount), timestamp(timestamp)
        {
        }

        Amount amount;   ///< Amount the commitment is for
        Date timestamp;  ///< When the commitment was made
    };

    std::unordered_map<std::string, Commitment> commitments; 

    void checkInvariants() const
    {
        try {
            //ExcAssert(netBudget.isNonNegative());
            ExcAssert(commitmentsRetired.isNonNegative());
            ExcAssert(commitmentsMade.isNonNegative());
            ExcAssert(spent.isNonNegative());
            
            CurrencyPool credit = netBudget + commitmentsRetired;
            CurrencyPool debit = commitmentsMade + spent + balance;
            
            ExcAssertEqual(credit, debit);
        } catch (...) {
            using namespace std;
            cerr << "invariants failed:" << endl;
            cerr << *this << endl;
            throw;
        }
    }

    Json::Value toJson() const
    {
        checkInvariants();

        Json::Value result(Json::objectValue);
        result["md"]["objectType"] = "ShadowAccount";
        result["md"]["version"] = 1;
        result["netBudget"] = netBudget.toJson();
        result["commitmentsRetired"] = commitmentsRetired.toJson();
        result["commitmentsMade"] = commitmentsMade.toJson();
        result["spent"] = spent.toJson();
        result["lineItems"] = lineItems.toJson();
        result["balance"] = balance.toJson();

        ShadowAccount reparsed = fromJson(result);
        reparsed.checkInvariants();
        ExcAssertEqual(netBudget, reparsed.netBudget);
        ExcAssertEqual(spent, reparsed.spent);
        ExcAssertEqual(commitmentsRetired, reparsed.commitmentsRetired);
        ExcAssertEqual(commitmentsMade, reparsed.commitmentsMade);
        ExcAssertEqual(lineItems, reparsed.lineItems);

        return result;
    }

    static const ShadowAccount fromJson(const Json::Value & val)
    {
        ShadowAccount result;
        ExcAssertEqual(val["md"]["objectType"].asString(), "ShadowAccount");
        ExcAssertEqual(val["md"]["version"].asInt(), 1);

        result.netBudget = CurrencyPool::fromJson(val["netBudget"]);
        result.commitmentsRetired = CurrencyPool::fromJson(val["commitmentsRetired"]);
        result.commitmentsMade = CurrencyPool::fromJson(val["commitmentsMade"]);
        result.spent = CurrencyPool::fromJson(val["spent"]);
        result.balance = CurrencyPool::fromJson(val["balance"]);
        result.lineItems = LineItems::fromJson(val["lineItems"]);

        result.checkInvariants();

        return result;
    }

    /*************************************************************************/
    /* SPEND TRACKING                                                        */
    /*************************************************************************/

    void forceWinBid(Amount amountPaid,
                     const LineItems & lineItems)
    {
        commitDetachedBid(Amount(), amountPaid, lineItems);
    }

    /// Commit a bid that has been detached from its tracking
    void commitDetachedBid(Amount amountAuthorized,
                           Amount amountPaid,
                           const LineItems & lineItems)
    {
        checkInvariants();
        Amount amountUnspent = amountAuthorized - amountPaid;
        balance += amountUnspent;
        commitmentsRetired += amountAuthorized;
        spent += amountPaid;

        if(amountPaid) {
            // Increase the number of impressions by 1
            // whenever an amount is paid for a bid
            commitEvent(Amount(CurrencyCode::CC_IMP, 1.0));
        }

        this->lineItems += lineItems;
        checkInvariants();
    }

    /// Commit a specific currency (amountToCommit)
    void commitEvent(const Amount & amountToCommit)
    {
        checkInvariants();
        spent += amountToCommit;
        commitmentsRetired += amountToCommit;
        checkInvariants();
    }

    /*************************************************************************/
    /* SPEND AUTHORIZATION                                                   */
    /*************************************************************************/

    bool authorizeBid(const std::string & item,
                      Amount amount)
    {
        checkInvariants();

        if (!balance.hasAvailable(amount))
            return false;  // no budget balance

        attachBid(item, amount);

        balance -= amount;
        commitmentsMade += amount;

        checkInvariants();

        return true;
    }
    
    void commitBid(const std::string & item,
                   Amount amountPaid,
                   const LineItems & lineItems)
    {
        commitDetachedBid(detachBid(item), amountPaid, lineItems);
    }

    void cancelBid(const std::string & item)
    {
        commitDetachedBid(detachBid(item), Amount(), LineItems());
    }
    
    Amount detachBid(const std::string & item)
    {
        checkInvariants();

        auto cit = commitments.find(item);
        if (cit == commitments.end())
            throw ML::Exception("unknown commitment being committed");

        Amount amountAuthorized = cit->second.amount;
        commitments.erase(cit);

        checkInvariants();

        detachedBids++;
        
        return amountAuthorized;
    }

    void attachBid(const std::string & item,
                   Amount amount)
    {
        Date now = Date::now();
        auto c = commitments.insert(make_pair(item, Commitment(amount, now)));
        if (!c.second)
            throw ML::Exception("attempt to re-open commitment");
        attachedBids++;
    }

    /*************************************************************************/
    /* SYNCHRONIZATION                                                       */
    /*************************************************************************/

    const Account syncToMaster(Account & masterAccount) const
    {
        checkInvariants();

        masterAccount.checkInvariants();

        CurrencyPool newCommitmentsMade
            = commitmentsMade - masterAccount.commitmentsMade;
        CurrencyPool newCommitmentsRetired
            = commitmentsRetired - masterAccount.commitmentsRetired;
        CurrencyPool newSpend
            = spent - masterAccount.spent;

        ExcAssert(newCommitmentsMade.isNonNegative());
        ExcAssert(newCommitmentsRetired.isNonNegative());
        ExcAssert(newSpend.isNonNegative());

        masterAccount.commitmentsRetired = commitmentsRetired;
        masterAccount.commitmentsMade = commitmentsMade;
        masterAccount.spent = spent;

        masterAccount.balance
            += (newCommitmentsRetired - newCommitmentsMade - newSpend);

        masterAccount.lineItems = lineItems;

        masterAccount.checkInvariants("syncToMaster");
        checkInvariants();

        return masterAccount;
    }

    void syncFromMaster(const Account & masterAccount)
    {
        checkInvariants();
        masterAccount.checkInvariants();

        // net budget: balance assuming spent, commitments are zero
        netBudget = masterAccount.getNetBudget();
        balance = netBudget + commitmentsRetired
            - commitmentsMade - spent;

        status = masterAccount.status;
        checkInvariants();
    }

    /** This method should be called exactly once the first time that a
        shadow account receives its initial state from the master.

        It will merge any changes that have been made since initialization
        with the initial state from the master, in such a way that the
        state will be the same as if the account had been synchronized before
        any operations had occurred and then all operations had been
        replayed.
    */
    void initializeAndMergeState(const Account & masterAccount)
    {
        // We have to tally up the fields from the master and the current
        // status.

        checkInvariants();
        masterAccount.checkInvariants();

        // net budget: balance assuming spent, commitments are zero
        netBudget = masterAccount.getNetBudget();
        commitmentsMade += masterAccount.commitmentsMade;
        commitmentsRetired += masterAccount.commitmentsRetired;
        spent += masterAccount.spent;
        lineItems += masterAccount.lineItems;

        balance = netBudget + commitmentsRetired - commitmentsMade - spent;

        checkInvariants();
    }

    /* LOGGING */
    uint32_t attachedBids;
    uint32_t detachedBids;
    uint32_t lastExpiredCommitments;

    void logBidEvents(const Datacratic::EventRecorder & eventRecorder,
                      const std::string & accountKey);
};


/*****************************************************************************/
/* ACCOUNT SUMMARY                                                           */
/*****************************************************************************/

/** This is a summary of an account and all of its sub-accounts. */

struct AccountSummary {
    CurrencyPool budget;         ///< Total amount we're allowed to spend
    CurrencyPool inFlight;       ///< Sum of sub-account inFlights (pending commitments)
    CurrencyPool spent;          ///< Sum of sub-account spend
    CurrencyPool adjustments;    ///< Sum of sub-account adjustments
    CurrencyPool adjustedSpent;  ///< Spend minus adjustments
    CurrencyPool effectiveBudget;  ///< budget computed internally
    CurrencyPool available;      ///< Total amount we're allowed to spend

    Account account;

    void addChild(const std::string & name,
                  const AccountSummary & child,
                  bool addInSubaccounts)
    {
        if (addInSubaccounts)
            subAccounts[name] = child;
        effectiveBudget += child.effectiveBudget;
        inFlight += child.inFlight;
        spent += child.spent;
        adjustments += child.adjustments;
    }

    void dump(std::ostream & stream,
              int indent = 0,
              const std::string & name = "toplevel") const
    {
        stream << std::string(indent, ' ')
               << name
               << " b:" << budget
               << " s:" << spent
               << " i:" << inFlight
               << std::endl;
        for (const auto & sa: subAccounts) {
            sa.second.dump(stream, indent + 2, sa.first);
        }
    }

    Json::Value toJson(bool simplified = false) const
    {
        Json::Value result;

        result["md"]["objectType"]
            = simplified ? "AccountSimpleSummary" : "AccountSummary";
        result["md"]["version"] = 1;
        result["budget"] = budget.toJson();
        result["effectiveBudget"] = effectiveBudget.toJson();
        result["spent"] = spent.toJson();
        result["adjustments"] = adjustments.toJson();
        result["adjustedSpent"] = adjustedSpent.toJson();
        result["available"] = available.toJson();
        result["inFlight"] = inFlight.toJson();
        if (!simplified) {
            result["account"] = account.toJson();
            for (const auto & sa: subAccounts) {
                result["subAccounts"][sa.first] = sa.second.toJson();
            }
        }

        return result;
    }

    static AccountSummary fromJson(const Json::Value & val)
    {
        AccountSummary result;

        ExcAssertEqual(val["md"]["objectType"].asString(), "AccountSummary");
        ExcAssertEqual(val["md"]["version"].asInt(), 1);

        result.budget = CurrencyPool::fromJson(val["budget"]);
        result.effectiveBudget = CurrencyPool::fromJson(val["effectiveBudget"]);
        result.inFlight = CurrencyPool::fromJson(val["inFlight"]);
        result.spent = CurrencyPool::fromJson(val["spent"]);
        result.adjustments = CurrencyPool::fromJson(val["adjustments"]);
        result.adjustedSpent = CurrencyPool::fromJson(val["adjustedSpent"]);
        result.available = CurrencyPool::fromJson(val["available"]);

        result.account = Account::fromJson(val["account"]);
        auto & sa = val["subAccounts"];
        for (auto it = sa.begin(), end = sa.end();  it != end;  ++it) {
            result.subAccounts[it.memberName()]
                = AccountSummary::fromJson(*it);
        }

        return result;
    }

    std::map<std::string, AccountSummary> subAccounts;
};

inline std::ostream &
operator << (std::ostream & stream, const AccountSummary & summary)
{
    summary.dump(stream);
    return stream;
}

/*****************************************************************************/
/* ACCOUNTS                                                                  */
/*****************************************************************************/

struct Accounts {
    Accounts()
        : sessionStart(Datacratic::Date::now())
    {
    }

    Datacratic::Date sessionStart;

    struct AccountInfo: public Account {
        std::set<AccountKey> children;

        /* spend tracking across sessions */
        CurrencyPool initialSpent;
    };

    const Account createAccount(const AccountKey & account,
                                AccountType type)
    {
        Guard guard(lock);
        if (account.empty())
            throw ML::Exception("can't create account with empty key");
        return ensureAccount(account, type);
    }

    void restoreAccount(const AccountKey & accountKey,
                        const Json::Value & jsonValue,
                        bool overwrite = false) {
        Guard guard(lock);

        // if (accounts.count(accountKey) != 0 and !overwrite) {
        //     throw ML::Exception("an account already exists with that name");
        // }

        Account validAccount = validAccount.fromJson(jsonValue);
        AccountInfo & newAccount = ensureAccount(accountKey, validAccount.type);
        newAccount.type = AT_SPEND;
        newAccount.type = validAccount.type;
        newAccount.budgetIncreases = validAccount.budgetIncreases;
        newAccount.budgetDecreases = validAccount.budgetDecreases;
        newAccount.spent = validAccount.spent;
        newAccount.recycledIn = validAccount.recycledIn;
        newAccount.recycledOut = validAccount.recycledOut;
        newAccount.allocatedIn = validAccount.allocatedIn;
        newAccount.allocatedOut = validAccount.allocatedOut;
        newAccount.commitmentsMade = validAccount.commitmentsMade;
        newAccount.commitmentsRetired = validAccount.commitmentsRetired;
        newAccount.adjustmentsIn = validAccount.adjustmentsIn;
        newAccount.adjustmentsOut = validAccount.adjustmentsOut;
        newAccount.balance = validAccount.balance;
        newAccount.lineItems = validAccount.lineItems;
        newAccount.adjustmentLineItems = validAccount.adjustmentLineItems;
        newAccount.status = Account::ACTIVE;
    }

    void reactivateAccount(const AccountKey & accountKey)
    {
        Guard guard(lock);
        AccountKey parents = accountKey;
        while (!parents.empty()) {
            getAccountImpl(parents).status = Account::ACTIVE;
            parents.pop_back();
        }
        reactivateAccountChildren(accountKey);
    }

    const Account createBudgetAccount(const AccountKey & account)
    {
        Guard guard(lock);
        if (account.empty())
            throw ML::Exception("can't create account with empty key");
        return ensureAccount(account, AT_BUDGET);
    }

    const Account createSpendAccount(const AccountKey & account)
    {
        Guard guard(lock);
        if (account.size() < 2)
            throw ML::Exception("commitment account must have parent");
        return ensureAccount(account, AT_SPEND);
    }

    const AccountInfo getAccount(const AccountKey & account) const
    {
        Guard guard(lock);
        return getAccountImpl(account);
    }

    std::pair<bool, bool> accountPresentAndActive(const AccountKey & account) const
    {
        Guard guard(lock);
        return accountPresentAndActiveImpl(account);
    }
    
    /** closeAccount behavior is to close all children then close itself,
        always transfering from children to parent. If top most account, 
        then throws an error after closing all children first.
    */
    const Account closeAccount(const AccountKey & account)
    {
        Guard guard(lock);
        return closeAccountImpl(account);
    }

    void checkInvariants() const
    {
        Guard guard(lock);
        for (auto & a: accounts) {
            a.second.checkInvariants();
        }
    }

    Json::Value toJson() const
    {
        Json::Value result(Json::objectValue);

        Guard guard(lock);
        for (auto & a: accounts) {
            result[a.first.toString()] = a.second.toJson();
        }

        return result;
    }

    static Accounts fromJson(const Json::Value & json);

    /*************************************************************************/
    /* BUDGET OPERATIONS                                                     */
    /*************************************************************************/

    /* These operations are assocated with putting money into the system. */

    const Account setBudget(const AccountKey & topLevelAccount,
                            const CurrencyPool & newBudget)
    {
        using namespace std;
        //cerr << "setBudget with newBudget " << newBudget << endl;

        Guard guard(lock);
        if (topLevelAccount.size() != 1)
            throw ML::Exception("can't setBudget except at top level");
        auto & a = ensureAccount(topLevelAccount, AT_BUDGET);
        a.setBudget(newBudget);
        return a;
    }

    /** Sets the balance budget for the given account to the given amount,
        by transferring in from the parent account.

        If typeToCreate is not AT_NONE, then the account will be implicitly
        created if it doesn't exist.
    */
    const Account setBalance(const AccountKey & account,
                             CurrencyPool amount,
                             AccountType typeToCreate)
    {
        Guard guard(lock);

        if (typeToCreate != AT_NONE && !accounts.count(account)) {
            auto & a = ensureAccount(account, typeToCreate);
            a.setBalance(getParentAccount(account), amount);
            return a;
        }
        else {
            auto & a = getAccountImpl(account);

#if 0
            using namespace std;
            if (a.type == AT_BUDGET)
                cerr << Date::now()
                     << " setBalance " << account << " " << " from " << a.balance
                     << " to " << amount << endl;
#endif

            a.setBalance(getParentAccount(account), amount);
            return a;
        }
    }

    const CurrencyPool getBalance(const AccountKey & account) const
    {
        Guard guard(lock);
        auto it = accounts.find(account);
        if (it == accounts.end())
            return CurrencyPool();
        return it->second.balance;
    }

    const Account addAdjustment(const AccountKey & account,
                                CurrencyPool amount)
    {
        Guard guard(lock);

        auto & a = getAccountImpl(account);
        a.addAdjustment(amount);

        return a;
    }



    /*************************************************************************/
    /* TRANSFER OPERATIONS                                                   */
    /*************************************************************************/

    /* These operations are two-sided and involve transferring between a
       parent account and a child account.
    */

    void recuperate(const AccountKey & account)
    {
        Guard guard(lock);
        getAccountImpl(account).recuperateTo(getParentAccount(account));
    }

    AccountSummary getAccountSummary(const AccountKey & account,
                                     int maxDepth = -1) const
    {
        Guard guard(lock);
        return getAccountSummaryImpl(account, 0, maxDepth);
    }

    Json::Value
    getAccountSummariesJson(bool simplified = false, int maxDepth = -1)
        const
    {
        Guard guard(lock);

        Json::Value summaries;

        for (const auto & it: accounts) {
            const AccountKey & key = it.first;
            AccountSummary summary = getAccountSummaryImpl(key, 0, maxDepth);
            summaries[key.toString()] = summary.toJson(simplified);
        }

        return summaries;
    }

    const Account importSpend(const AccountKey & account,
                              const CurrencyPool & amount)
    {
        Guard guard(lock);
        auto & a = getAccountImpl(account);
        a.importSpend(amount);
        return a;
    }
                      
    /*************************************************************************/
    /* HIGH LEVEL OPERATIONS                                                 */
    /*************************************************************************/

    /* These are higher-level opertions that build on top of the others in
       order to make a given condition true.
    */

    /*************************************************************************/
    /* SYNCHRONIZATION OPERATIONS                                            */
    /*************************************************************************/

    const Account syncFromShadow(const AccountKey & account,
                                 const ShadowAccount & shadow)
    {
        Guard guard(lock);

        // In the case that an account was added and the banker crashed
        // before it could be written to persistent storage, we need to
        // create the empty account here.
        if (!accounts.count(account))
            return shadow.syncToMaster(ensureAccount(account, AT_SPEND));
        
        return shadow.syncToMaster(getAccountImpl(account));
    }

    /* "Out of sync" here means that the in-memory version of the relevant
       accounts is obsolete compared to the version stored in the Redis
       backend */
    void markAccountOutOfSync(const AccountKey & account)
    {
        Guard guard(lock);

        outOfSyncAccounts.insert(account);
    }

    bool isAccountOutOfSync(const AccountKey & account) const
    {
        Guard guard(lock);
        
        return (outOfSyncAccounts.count(account) > 0);
    }


    /** interaccount consistency */
    /* "Inconsistent" here means that there is a mismatch between the members
     * used in money transfers for a given Account and the corresponding
     * members in its subaccounts: allocatedOut and budgetIncreases,
     * recycledIn and recycedOut, ...
     */
    void ensureInterAccountConsistency();
    bool isAccountInconsistent(const AccountKey & account) const
    {
        Guard guard(lock);
        
        return (inconsistentAccounts.count(account) > 0);
    }

    /* Returns whether the budgetIncreases of subaccounts are consistent with
       the allocatedOut of the top-account, recursively.
       maxRecusion: -1 = infinity 
    */
    bool checkBudgetConsistency(const AccountKey & accountKey,
                                int maxRecursion = -1) const;

    /* Returns the amounts in recycledIn and recycledOut that were transferred
     * strictly from and to the parent account. */
    void getRecycledUp(const AccountKey & accountKey,
                       CurrencyPool & recycledInUp,
                       CurrencyPool & recycledOutUp) const;

private:
    friend class ShadowAccounts;

    typedef ML::Spinlock Lock;
    typedef std::unique_lock<Lock> Guard;
    mutable Lock lock;

    typedef std::map<AccountKey, AccountInfo> AccountMap;
    AccountMap accounts;

    typedef std::unordered_set<AccountKey> AccountSet;
    AccountSet outOfSyncAccounts;
    AccountSet inconsistentAccounts;

public:
    std::vector<AccountKey>
    getAccountKeys(const AccountKey & prefix = AccountKey(),
                   int maxDepth = -1) const
    {
        Guard guard(lock);

        std::vector<AccountKey> result;

        for (auto it = accounts.lower_bound(prefix), end = accounts.end();
             it != accounts.end() && it->first.hasPrefix(prefix);  ++it) {
            if (maxDepth == -1 || it->first.size() <= maxDepth)
                result.push_back(it->first);
        }
        return result;
    }

    void
    forEachAccount(const std::function<void (const AccountKey &,
                                             const Account &)>
                   & onAccount) const
    {
        Guard guard(lock);
        
        for (auto & a: accounts) {
            onAccount(a.first, a.second);
        }
    }
                        
    size_t size() const
    {
        Guard guard(lock);
        return accounts.size();
    }

    bool empty() const
    {
        Guard guard(lock);
        return accounts.empty();
    }

    /** Return a subtree of the accounts. */
    Accounts getAccounts(const AccountKey & root, int maxDepth = 0)
    {
        Accounts result;
        Guard guard(lock);

        std::function<void (const AccountKey &, int, int)> doAccount
            = [&] (const AccountKey & key, int depth, int maxDepth)
            {
                auto it = accounts.find(key);
                if (it == accounts.end())
                    return;
                result.ensureAccount(it->first, it->second.type) = it->second;

                if (depth >= maxDepth)
                    return;

                for (auto & k: it->second.children)
                    doAccount(k, depth + 1, maxDepth);
            };
              
        doAccount(root, 0, maxDepth);

        return result;
    }

private:

    AccountInfo & ensureAccount(const AccountKey & accountKey,
                                AccountType type)
    {
        ExcAssertGreaterEqual(accountKey.size(), 1);

        auto it = accounts.find(accountKey);
        if (it != accounts.end()) {
            ExcAssertEqual(it->second.type, type);
            return it->second;
        }
        else {
            if (accountKey.size() == 1) {
                ExcAssertEqual(type, AT_BUDGET);
            }
            else {
                AccountInfo & parent
                    = ensureAccount(accountKey.parent(), AT_BUDGET);
                parent.children.insert(accountKey);
            }

            auto & result = accounts[accountKey];
            result.type = type;
            return result;
        }
    }

    AccountInfo & getAccountImpl(const AccountKey & account)
    {
        auto it = accounts.find(account);
        if (it == accounts.end())
            throw ML::Exception("couldn't get account: " + account.toString());
        return it->second;
    }

    std::pair<bool, bool> accountPresentAndActiveImpl(const AccountKey & account) const
    {
        auto it = accounts.find(account);
        if (it == accounts.end())
            return std::make_pair(false, false);
        if (it->second.status == Account::CLOSED)
            return std::make_pair(true, false);
        else
            return std::make_pair(true, true);
    }


    const Account closeAccountImpl(const AccountKey & accountKey)
    {
        AccountInfo & account = getAccountImpl(accountKey);
        if (account.status == Account::CLOSED)
            return account;

        for ( AccountKey child : account.children ) {
            closeAccountImpl(child);
        }

        if (accountKey.size() > 1)
            account.recuperateTo(getParentAccount(accountKey));

        account.status = Account::CLOSED;

        return account;
    }

    void reactivateAccountChildren(const AccountKey & accountKey) {
        if (accountPresentAndActiveImpl(accountKey).first) {
            AccountInfo & account = getAccountImpl(accountKey);
            for (auto child : account.children)
                reactivateAccountChildren(child);

            account.status = Account::ACTIVE;
        }
    }

    const AccountInfo & getAccountImpl(const AccountKey & account) const
    {
        auto it = accounts.find(account);
        if (it == accounts.end())
            throw ML::Exception("couldn't get account: " + account.toString());
        return it->second;
    }

    Account & getParentAccount(const AccountKey & accountKey)
    {
        if (accountKey.size() < 2)
            throw ML::Exception("account has no parent");

        AccountKey parentKey = accountKey;
        parentKey.pop_back();

        Account & result = getAccountImpl(parentKey);
        ExcAssertEqual(result.type, AT_BUDGET);
        return result;
    }

    void forEachChildAccount(const AccountKey & account,
                             std::function<void (const AccountKey & key)> cb) const
    {
        auto & info = getAccountImpl(account);
        for (const AccountKey & ch: info.children)
            cb(ch);
    }

    AccountSummary getAccountSummaryImpl(const AccountKey & account,
                                         int depth, int maxDepth) const
    {
        AccountSummary result;

        const Account & a = getAccountImpl(account);

        result.account = a;
        result.spent = a.spent;
        result.budget = a.budgetIncreases - a.budgetDecreases;
        result.effectiveBudget = a.budgetIncreases - a.budgetDecreases 
                        + a.recycledIn - a.recycledOut
                        + a.allocatedIn - a.allocatedOut;
        result.inFlight = a.commitmentsMade - a.commitmentsRetired;
        result.adjustments = a.adjustmentsIn - a.adjustmentsOut;

        auto doChildAccount = [&] (const AccountKey & key) {
            auto childSummary = getAccountSummaryImpl(key, depth + 1,
                                                      maxDepth);
            result.addChild(key.back(), childSummary,
                            maxDepth == -1 || depth < maxDepth);
        };
        forEachChildAccount(account, doChildAccount);
        
        result.adjustedSpent = result.spent - result.adjustments;

        result.available = (result.effectiveBudget - result.adjustedSpent - result.inFlight);
        
        return result;
    }

    bool checkBudgetConsistencyImpl(const AccountKey & accountKey,
                                    int maxRecursion, int currentLevel) const;
};


/*****************************************************************************/
/* SHADOW ACCOUNTS                                                           */
/*****************************************************************************/

struct ShadowAccounts {
    /** Callback called whenever a new account is created.  This can be
        assigned to in order to add functionality that must be present
        whenever a new account is created.
    */
    std::function<void (AccountKey)> onNewAccount;
    
    const ShadowAccount activateAccount(const AccountKey & account)
    {
        Guard guard(lock);
        return getAccountImpl(account);
    }

    const ShadowAccount syncFromMaster(const AccountKey & account,
                                       const Account & master)
    {
        Guard guard(lock);
        auto & a = getAccountImpl(account);
        ExcAssert(!a.uninitialized);
        a.syncFromMaster(master);
        return a;
    }

    /** Initialize an account by merging with the initial state as
        received from the master banker.
    */
    const ShadowAccount
    initializeAndMergeState(const AccountKey & account,
                            const Account & master)
    {
        Guard guard(lock);
        auto & a = getAccountImpl(account);
        ExcAssert(a.uninitialized);
        a.initializeAndMergeState(master);
        a.uninitialized = false;
        return a;
    }

    void checkInvariants() const
    {
        Guard guard(lock);
        for (auto & a: accounts) {
            a.second.checkInvariants();
        }
    }

    const ShadowAccount getAccount(const AccountKey & accountKey) const
    {
        Guard guard(lock);
        return getAccountImpl(accountKey);
    }

    bool accountExists(const AccountKey & accountKey) const
    {
        Guard guard(lock);
        return accounts.count(accountKey);
    }

    bool createAccountAtomic(const AccountKey & accountKey)
    {
    	Guard guard(lock);

    	AccountEntry & account = getAccountImpl(accountKey, false /* call onCreate */);
    	bool result = account.first;

    	// record that this account creation is requested for the first time
    	account.first = false;
    	return result;

    }

    /*************************************************************************/
    /* SYNCHRONIZATION                                                       */
    /*************************************************************************/

    void syncTo(Accounts & master) const
    {
        Guard guard1(lock);
        Guard guard2(master.lock);

        for (auto & a: accounts)
            a.second.syncToMaster(master.getAccountImpl(a.first));
    }

    void syncFrom(const Accounts & master)
    {
        Guard guard1(lock);
        Guard guard2(master.lock);

        for (auto & a: accounts) {
            a.second.syncFromMaster(master.getAccountImpl(a.first));
            if (master.outOfSyncAccounts.count(a.first) > 0) {
                outOfSyncAccounts.insert(a.first);
            }
        }
    }

    void sync(Accounts & master)
    {
        Guard guard1(lock);
        Guard guard2(master.lock);

        for (auto & a: accounts) {
            a.second.syncToMaster(master.getAccountImpl(a.first));
            a.second.syncFromMaster(master.getAccountImpl(a.first));
        }
    }

    bool isInitialized(const AccountKey & accountKey) const
    {
        Guard guard(lock);
        return !getAccountImpl(accountKey).uninitialized;
    }

    bool isStalled(const AccountKey & accountKey) const
    {
        Guard guard(lock);
        auto & account = getAccountImpl(accountKey);
        return account.uninitialized && account.requested.minutesUntil(Date::now()) >= 1.0;
    }

    void reinitializeStalledAccount(const AccountKey & accountKey)
    {
        ExcAssert(isStalled(accountKey));
        Guard guard(lock);
        auto & account = getAccountImpl(accountKey);
        account.first = true;
        account.requested = Date::now();
    }

    /*************************************************************************/
    /* BID OPERATIONS                                                        */
    /*************************************************************************/

    bool authorizeBid(const AccountKey & accountKey,
                      const std::string & item,
                      Amount amount)
    {
        Guard guard(lock);
        return (outOfSyncAccounts.count(accountKey) == 0
                && getAccountImpl(accountKey).authorizeBid(item, amount));
    }
    
    void commitBid(const AccountKey & accountKey,
                   const std::string & item,
                   Amount amountPaid,
                   const LineItems & lineItems)
    {
        Guard guard(lock);
        return getAccountImpl(accountKey).commitBid(item, amountPaid, lineItems);
    }

    void cancelBid(const AccountKey & accountKey,
                   const std::string & item)
    {
        Guard guard(lock);
        return getAccountImpl(accountKey).cancelBid(item);
    }
    
    void forceWinBid(const AccountKey & accountKey,
                     Amount amountPaid,
                     const LineItems & lineItems)
    {
        Guard guard(lock);
        return getAccountImpl(accountKey).forceWinBid(amountPaid, lineItems);
    }

    /// Commit a bid that has been detached from its tracking
    void commitDetachedBid(const AccountKey & accountKey,
                           Amount amountAuthorized,
                           Amount amountPaid,
                           const LineItems & lineItems)
    {
        Guard guard(lock);
        return getAccountImpl(accountKey)
            .commitDetachedBid(amountAuthorized, amountPaid, lineItems);
    }

    /// Commit a specific currency (amountToCommit)
    void commitEvent(const AccountKey & accountKey, const Amount & amountToCommit)
    {
        Guard guard(lock);
        return getAccountImpl(accountKey).commitEvent(amountToCommit);
    }

    Amount detachBid(const AccountKey & accountKey,
                     const std::string & item)
    {
        Guard guard(lock);
        return getAccountImpl(accountKey).detachBid(item);
    }

    void attachBid(const AccountKey & accountKey,
                   const std::string & item,
                   Amount amountAuthorized)
    {
        Guard guard(lock);
        getAccountImpl(accountKey).attachBid(item, amountAuthorized);
    }

    void logBidEvents(const Datacratic::EventRecorder & eventRecorder);

private:

    struct AccountEntry : public ShadowAccount {
        AccountEntry(bool uninitialized = true, bool first = true)
            : requested(Date::now()), uninitialized(uninitialized), first(first)
        {
        }

        /** This flag marks that the shadow account has been created, but
            it has never had its state read from the master banker.  In this
            case we will need to merge anything that was done to the current
            account with the initial state from the master banker to obtain
            the new state.

            This is the *only* case in which both the master banker and the
            slave banker can both have different ideas of the state of the
            budget of an account.
        */

        Date requested;
        bool uninitialized;
        bool first;
    };

    AccountEntry & getAccountImpl(const AccountKey & account,
                                  bool callOnNewAccount = true)
    {
        auto it = accounts.find(account);
        if (it == accounts.end()) {
            if (callOnNewAccount && onNewAccount)
                onNewAccount(account);
            it = accounts.insert(std::make_pair(account, AccountEntry()))
                .first;
        }
        return it->second;
    }

    const AccountEntry & getAccountImpl(const AccountKey & account) const
    {
        auto it = accounts.find(account);
        if (it == accounts.end())
            throw ML::Exception("getting unknown account " + account.toString());
        return it->second;
    }

    typedef ML::Spinlock Lock;
    typedef std::unique_lock<Lock> Guard;
    mutable Lock lock;

    typedef std::map<AccountKey, AccountEntry> AccountMap;
    AccountMap accounts;

    typedef std::unordered_set<AccountKey> AccountSet;
    AccountSet outOfSyncAccounts;

public:
    std::vector<AccountKey>
    getAccountKeys(const AccountKey & prefix = AccountKey()) const
    {
        Guard guard(lock);

        std::vector<AccountKey> result;

        for (auto it = accounts.lower_bound(prefix), end = accounts.end();
             it != accounts.end() && it->first.hasPrefix(prefix);  ++it) {
            result.push_back(it->first);
        }
        return result;
    }

    void
    forEachAccount(const std::function<void (const AccountKey &,
                                             const ShadowAccount &)> &
                   onAccount) const
    {
        Guard guard(lock);
        
        for (auto & a: accounts) {
            onAccount(a.first, a.second);
        }
    }

    void
    forEachInitializedAndActiveAccount(const std::function<void (const AccountKey &,
                                                        const ShadowAccount &)> & onAccount)
    {
        Guard guard(lock);
        
        for (auto & a: accounts) {
            if (a.second.uninitialized || a.second.status == Account::CLOSED)
                continue;
            onAccount(a.first, a.second);
        }
    }

    size_t size() const
    {
        Guard guard(lock);
        return accounts.size();
    }

    bool empty() const
    {
        Guard guard(lock);
        return accounts.empty();
    }
};

} // namespace RTBKIT


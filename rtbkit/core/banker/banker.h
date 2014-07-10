/* banker.h                                                    -*- C++ -*-
   Sunil Rottoo, May 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.
   
   Class to deal with accounting in the router.
*/

#ifndef __rtb_router__banker_h__
#define __rtb_router__banker_h__

#include <string>
#include <set>
#include <unordered_map>
#include "soa/jsoncpp/json.h"
#include "soa/types/date.h"
#include "jml/arch/spinlock.h"
#include "jml/arch/exception.h"
#include "soa/service/service_base.h"
#include <future>
#include "rtbkit/common/currency.h"
#include "rtbkit/core/monitor/monitor_provider.h"
#include "jml/arch/futex.h"
#include "jml/arch/backtrace.h"
#include "account.h"

namespace Datacratic {
namespace JS {
    class NullBankerJS;
} // namespace JS
} // namespace Datacratic

namespace RTBKIT {

template<typename T>
struct BankerSyncResult:  boost::noncopyable {
    BankerSyncResult()
    {
        done = 0;
        exc = nullptr;
    }

    int done;
    std::exception_ptr exc;
    T result;

    operator std::function<void (std::exception_ptr, T && result)> ()
    {
        return [=] (std::exception_ptr exc, T && result)
            {
                using namespace std;
                if (this->done) {
                    cerr << "already done budget result" << endl;
                    ML::backtrace();
                    abort();
                }
                //cerr << "done result " << this << endl;
                this->exc = exc;
                this->done = 1;
                this->result = std::move(result);
                ML::futex_wake(this->done);
            };
    }

    T && get()
    {
        while (!done)
            ML::futex_wait(done, 0);
        if (exc)
            std::rethrow_exception(exc);
        return std::move(result);
    }
};

template<>
struct BankerSyncResult<void> : boost::noncopyable {
    BankerSyncResult()
    {
        done = 0;
        exc = nullptr;
    }

    int done;
    std::exception_ptr exc;

    operator std::function<void (std::exception_ptr)> ()
    {
        return [=] (std::exception_ptr exc)
            {
                using namespace std;
                if (this->done) {
                    cerr << "already done budget result" << endl;
                    ML::backtrace();
                    abort();
                }
                //cerr << "done result " << this << endl;
                this->exc = exc;
                this->done = 1;
                ML::futex_wake(this->done);
            };
    }

    void get()
    {
        while (!done)
            ML::futex_wait(done, 0);
        if (exc)
            std::rethrow_exception(exc);
    }
};

    


/*****************************************************************************/
/* BUDGET CONTROLLER                                                         */
/*****************************************************************************/

struct BudgetController {

    BudgetController();
    virtual ~BudgetController();


    /*************************************************************************/
    /* BUDGET INTERFACE                                                      */
    /*************************************************************************/

    typedef std::function<void (std::exception_ptr)> OnBudgetResult;

    /** Add an account to the system.  If it already exists, nothing is
        done.

        A newly added account starts with a zero budget.
    */
    virtual void addAccount(const AccountKey & account,
                            const OnBudgetResult & onResult);

    virtual void addAccountSync(const AccountKey & account);

    /**
     * The amount here refers to an absolute amount that the strategy should have. This method
     * transfers the appropriate amount so that the strategy ends up with the specified amount
     * 
     * The default implementation of the following two methods call each other, so you
     * must override at least one of them to avoid an infinite recursion
     * 
     * @param campaign The campaign to which we want to add our strategy
     * @param strategy The name of the strategy
     * @param amount The amount that the strategy should have after the transfer
     *
     * @exception BankerException with error set to INSUFFICIENT_FUNDS if the
     * amount to transfer exceeds the amount available
     *
     */
    virtual void topupTransfer(const AccountKey & account,
                               CurrencyPool amount,
                               const OnBudgetResult & onResult);
    
    virtual void topupTransferSync(const AccountKey & account,
                                   CurrencyPool amount);
    
    /**
     * Set the budget for a given campaign to the specified amount
     * @param campaign The campaign whose budget we want to set
     * @param amount The amount(in micro$) that we want to set the budget to
     *
     * If the campaign does not exist it will be created
     *
     * @exception BankerException with error to set LOWER_THAN_TRANSFERRED if the amount
     * specified is lower than the amount already spent in the campaign
     * @exception BankerException with error to set MAX_EXCEEDED if the amount
     * specified is more than the maximum currently allowed
     *
     */
    virtual void setBudget(const std::string & topLevelAccount,
                           CurrencyPool amount,
                           const OnBudgetResult & onResult);
    
    virtual void setBudgetSync(const std::string & topLevelAccount,
                               CurrencyPool amount);

    /**
     * Add to the budget for an existing campaign. If the campaign does not
     * exist it is created with an initial amount set to the amount specified
     * @param campaign The campaign whose budget we want to add to
     * @param amount The amount(in micro$) that we want to add to the budget
     *
     * If the campaign does not exist it will be created
     *
     * @exception BankerException with error to set LOWER_THAN_TRANSFERRED if the amount
     * specified is lower than the amount already spent in the campaign
     * @exception BankerException with error to set MAX_EXCEEDED if the amount
     * specified is more than the maximum currently allowed
     *
     */
    virtual void addBudget(const std::string & topLevelAccount,
                           CurrencyPool amount,
                           const OnBudgetResult & onResult);

    virtual void addBudgetSync(const std::string & topLevelAccount,
                               CurrencyPool amount);
    
    CurrencyPool budgetLimits;
};


/*****************************************************************************/
/* ACCOUNTANT                                                                */
/*****************************************************************************/

/** The accountant is responsible for keeping the books, especially by
    keeping track of what has been spent.
*/

struct Accountant {
    virtual void sync() {}

    virtual std::vector<AccountKey>
    getAccountListSync(const AccountKey & prefix,
                       int depth);

    virtual void
    getAccountList(const AccountKey & prefix,
                   int depth,
                   std::function<void (std::exception_ptr,
                                       std::vector<AccountKey> &&)> onResult);

    virtual AccountSummary
    getAccountSummarySync(const AccountKey & account,
                          int depth);

    virtual void
    getAccountSummary(const AccountKey & account,
                     int depth,
                     std::function<void (std::exception_ptr,
                                         AccountSummary &&)> onResult);

    virtual Account
    getAccountSync(const AccountKey & account);

    virtual void getAccount(const AccountKey & account,
                            std::function<void (std::exception_ptr,
                                                Account &&)> onResult);
};

/*****************************************************************************/
/* BANKER                                                                    */
/*****************************************************************************/

/** Abstract base class for a banker.  Defines the interface. */

class Banker : public MonitorProvider {
public:
    Banker()
    {
    }

    virtual ~Banker();


    typedef std::function<void (std::exception_ptr)> OnBudgetResult;


    /*************************************************************************/
    /* REGISTRATION INTERFACE                                                */
    /*************************************************************************/

    virtual ShadowAccount
    addSpendAccountSync(const AccountKey & account,
                        CurrencyPool accountFloat = CurrencyPool());

    virtual void
    addSpendAccount(const AccountKey & account,
                    CurrencyPool accountFloat,
                    std::function<void (std::exception_ptr, ShadowAccount&&)> onDone);


    /*************************************************************************/
    /* SPEND INTERFACE                                                       */
    /*************************************************************************/

    /*
     * authorize the spending for a bid for the specified amount. If there are
     * not enough funds we return false.
     * Note that the amount is subtracted from the pool of available funds.
     *
     * @param campaign The campaign that this applies to
     * @param strategy The strategy that is bidding on this item
     * @param item Identifier for the bid request
     * @param amount the amount we want to bid
     *
     * Note that there is no asynchronous interface.  Authorization must be
     * extremely fast and synchronous.
     */
    virtual bool authorizeBid(const AccountKey & account,
                              const std::string & item,
                              Amount amount) = 0;

    /*
     * Cancel the bid that was previously authorized. If we fail to find the bid
     * we return false.Otherwise we return the bid amount to the available pool
     * @param campaign The campaign that this applies to
     * @param strategy The strategy that is bidding on this item
     * @param item Identifier for the bid request
     *
     */
    virtual void cancelBid(const AccountKey & account,
                           const std::string & item)
    {
        return commitBid(account, item, Amount(), LineItems());
    }

    virtual void winBid(const AccountKey & account,
                        const std::string & item,
                        Amount amountPaid,
                        const LineItems & lineItems = LineItems())
    {
        return commitBid(account, item, amountPaid, lineItems);
    }
    
    virtual void attachBid(const AccountKey & account,
                           const std::string & item,
                           Amount amountAuthorized) = 0;

    virtual Amount detachBid(const AccountKey & account,
                             const std::string & item) = 0;

    /** Commit a bid.  This is used internally to both cancel and win bids.
        Asynchonous and returns no value.
    */
    virtual void commitBid(const AccountKey & account,
                           const std::string & item,
                           Amount amountPaid,
                           const LineItems & lineItems) = 0;

    /*
     * This is for the case when bids come in late. In this case because of the
     * fact that all bids are cancelled if they time out (tryCancelBid) we will
     * unconditionally reduce the amount available for the strategy.
     */
    virtual void forceWinBid(const AccountKey & account,
                             Amount amountPaid,
                             const LineItems & lineItems) = 0;

    /*************************************************************************/
    /* INFORMATION                                                           */
    /*************************************************************************/

    /** Synchronize all state with underlying storage. */
    virtual void sync() {}


    /*************************************************************************/
    /* LOGGING                                                               */
    /*************************************************************************/
    virtual void logBidEvents(const Datacratic::EventRecorder & eventRecorder)
    {
    }

    /*************************************************************************/
    /* MONITOR PROVIDER                                                      */
    /*************************************************************************/
    virtual std::string getProviderClass() const
    {
        return "rtbBanker";
    }

    virtual MonitorIndicator getProviderIndicators() const = 0;

protected:
    // For the supplied campaign make sure that the numbers match
    virtual void sanityCheck(const std::string &campaign) const
    {
    }
};

enum class BankerError {
    INVALID_CAMPAIGN,
    CAMPAIGN_NOT_FOUND,
    STRATEGY_NOT_FOUND,
    INVALID_STRATEGY,
    INSUFFICIENT_FUNDS,
    LOWER_THAN_TRANSFERRED,
    EXCEEDS_MAX,
    TOO_LOW,
    ACCOUNTING_MISMATCH,
    DATABASE_ERROR
};

/*****************************************************************************/
/* BANKER EXCEPTION                                                          */
/*****************************************************************************/

struct BankerException: public ML::Exception {

    // List of error conditions. Please update the errorToString method if adding new elements
    BankerException(const std::string &msg, const BankerError &error);

    static std::string errorToString(const BankerError &error);

    BankerError error_;
};


} // namespace RTBKIT

#endif /* __rtb_router__banker_h__ */


/* null_banker.h                                                   -*- C++ -*-
   Jeremy Barens, 11 October 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Banker class that either authorizes all bids or rejects all bids.
*/

#ifndef __banker__null_banker_h__
#define __banker__null_banker_h__

#include "banker.h"

namespace RTBKIT {

/*****************************************************************************/
/* NULL BUDGET CONTROLLER                                                    */
/*****************************************************************************/

/** A budget controller that does exactly nothing. */

struct NullBudgetController: public BudgetController {

    virtual void addAccountSync(const AccountKey & account);
    
    virtual void topupTransferSync(const AccountKey & account,
                                             CurrencyPool amount);
    
    virtual void setBudgetSync(const std::string & topLevelAccount,
                               CurrencyPool amount);
    
    virtual void addBudgetSync(const std::string & topLevelAccount,
                               CurrencyPool amount);
    
};


/*****************************************************************************/
/* NULL BANKER                                                               */
/*****************************************************************************/

/** A null banker that automatically either authorizes or rejects every
    request.
*/

class NullBanker: public Banker {
public:
    NullBanker(bool authorize = false, const std::string & servicName = "");

    virtual bool authorizeBid(const AccountKey & account,
                              const std::string & item,
                              Amount amount);

    /** Commit a bid.  This is used internally to both cancel and win bids.
        Asynchonous and returns no value.
    */
    virtual void commitBid(const AccountKey & account,
                           const std::string & item,
                           Amount amountPaid,
                           const LineItems & lineItems);

    virtual void forceWinBid(const AccountKey & account,
                             Amount amountPaid,
                             const LineItems & lineItems);
    
    virtual void attachBid(const AccountKey & account,
                           const std::string & item,
                           Amount amountAuthorized)
    {
    }

    virtual Amount detachBid(const AccountKey & account,
                             const std::string & item)
    {
        return Amount();
    }

    virtual ShadowAccount
    addSpendAccountSync(const AccountKey & account,
                        CurrencyPool accountFloat = CurrencyPool())
    {
        return ShadowAccount();
    }


    virtual MonitorIndicator
    getProviderIndicators() const;

#if 0
    std::string getCampaignStatusStr(const std::string & campaign,
                                     const std::string &strategy) const;
    Json::Value getCampaignStatusJson(const std::string & campaign,
                                      const std::string &strategy) const;
    
    Json::Value dumpAllCampaignsJson() const;
#endif    

#if 0
    void forEachOutstanding(const ForEachCallback & cb, Date latestDate);
    
    virtual OutstandingStats
    totalOutstandingStats(const std::string & campaign = "*",
                          const std::string & strategy = "*") const;
#endif

protected:
    virtual void sanityCheck(const std::string &campaign) const {};

    friend class Datacratic::JS::NullBankerJS;
    bool authorize_;
    std::string serviceName_;
};


struct NullAccountant: public Accountant {
    virtual AccountSummary
    getAccountSummarySync(const AccountKey & account,
                         int depth)
    {
        return AccountSummary();
    }

    virtual std::vector<AccountKey>
    getAccountListSync(const AccountKey & account,
                       int depth)
    {
        return std::vector<AccountKey>();
    }
};

} // namespace RTBKIT

#endif /* __banker__null_banker_h__ */

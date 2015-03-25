/* split_banker.h
   Michael Burkat, 24 March 2015
   Copyright (c) 2015 Datacratic.  All rights reserved.
   
   Split banker class, to divide campaigns between the
   master banker and local banker.
*/

#pragma once

#include <string>
#include <unordered_set>
#include "banker.h"
#include "local_banker.h"
#include "rtbkit/core/monitor/monitor_provider.h"

namespace RTBKIT {

struct SplitBanker : public Banker {

    std::shared_ptr<Banker> masterBanker;
    std::shared_ptr<Banker> localBanker;
    std::unordered_set<std::string> localCampaigns;

    SplitBanker(std::shared_ptr<Banker> master,
                std::shared_ptr<Banker> local,
                std::unordered_set<std::string> campaigns)
        : masterBanker(master),
          localBanker(local),
          localCampaigns(campaigns)
    {
    }

    bool isLocal(const AccountKey & account)
    {
        return localCampaigns.find(account[0]) != localCampaigns.end();
    }

    virtual void addSpendAccount(const AccountKey & account,
                    CurrencyPool accountFloat,
                    std::function<void (std::exception_ptr, ShadowAccount&&)> onDone)
    {
        if (isLocal(account)) {
            localBanker->addSpendAccount(account, Amount(), onDone);
        } else {
            masterBanker->addSpendAccount(account, Amount(), onDone);
        }
    }

    virtual bool authorizeBid(const AccountKey & account,
                              const std::string & item,
                              Amount amount)
    {
        if (isLocal(account)) {
            return localBanker->authorizeBid(account, item, amount);
        } else {
            return masterBanker->authorizeBid(account, item, amount);
        }
    }

    virtual void cancelBid(const AccountKey & account,
                           const std::string & item)
    {
        if (!isLocal(account)) {
            masterBanker->commitBid(account, item, Amount(), LineItems());
        }
    }

    virtual void winBid(const AccountKey & account,
                        const std::string & item,
                        Amount amountPaid,
                        const LineItems & lineItems = LineItems())
    {
        if (isLocal(account)) {
            localBanker->winBid(account, item, amountPaid, lineItems);
        } else {
            masterBanker->winBid(account, item, amountPaid, lineItems);
        }
    }

    virtual void attachBid(const AccountKey & account,
                           const std::string & item,
                           Amount amountAuthorized)
    {
        if (!isLocal(account)) {
           masterBanker->attachBid(account, item, amountAuthorized);
        }
    }

    virtual Amount detachBid(const AccountKey & account,
                             const std::string & item)
    {
        if (isLocal(account)) {
            return MicroUSD(0);
        } else {
            return masterBanker->detachBid(account, item);
        }
    }

    virtual void commitBid(const AccountKey & account,
                           const std::string & item,
                           Amount amountPaid,
                           const LineItems & lineItems)
    {
        if (!isLocal(account)) {
            masterBanker->commitBid(account, item, amountPaid, lineItems);
        }
    }

    virtual void forceWinBid(const AccountKey & account,
                             Amount amountPaid,
                             const LineItems & lineItems)
    {
        if (isLocal(account)) {
            localBanker->forceWinBid(account, amountPaid, lineItems);
        } else {
            masterBanker->forceWinBid(account, amountPaid, lineItems);
        }
    }

    virtual MonitorIndicator getProviderIndicators() const
    {
        return masterBanker->getProviderIndicators();
    }
};

} // namespace RTBKIT

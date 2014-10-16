/* router_stack.h                                                  -*- C++ -*-
   Jeremy Barnes, 21 November 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   The core router stack.
*/

#pragma once


#include "router.h"
#include "rtbkit/core/monitor/monitor_endpoint.h"
#include "rtbkit/core/banker/master_banker.h"
#include "rtbkit/core/banker/slave_banker.h"
#include "rtbkit/core/post_auction/post_auction_service.h"
#include "rtbkit/core/agent_configuration/agent_configuration_service.h"
#include "soa/service/testing/redis_temporary_server.h"

namespace RTBKIT {


/*****************************************************************************/
/* ROUTER STACK                                                              */
/*****************************************************************************/

/** The Router Stack is a Core Router, a Post Auction Loop and a Banker stuck
    together.  This is mostly used for where we need an integrated component
    (simulations, etc); normally they would be run separately.

    \todo There's a lot of commonalities here between the
    rtbkit_integration_test's Components class.
*/

struct RouterStack: public ServiceBase {
    
    RouterStack(std::shared_ptr<ServiceProxies> services,
                const std::string & serviceName = "routerStack",
                double secondsUntilLossAssumed = 2.0);

    void init();
    
    /** Start the router running in a separate thread.  The given lambda
        will be called when the thread is stopped. */
    virtual void
    start(boost::function<void ()> onStop = boost::function<void ()>());

    /** Sleep until the router is idle (there are no more auctions and
        all connections are idle).
    */
    virtual void sleepUntilIdle();
    
    virtual void shutdown();

    virtual size_t numNonIdle() const;

    void submitAuction(const std::shared_ptr<Auction> & auction,
                       const Id & adSpotId,
                       const Auction::Response & response);

    void notifyFinishedAuction(const Id & auctionId)
    {
        router.notifyFinishedAuction(auctionId);
    }

    int numAuctionsInProgress()
    {
        return router.numAuctionsInProgress();
    }

    Json::Value getStats() const
    {
        return router.getStats();
    }

    /** Inject a WIN into the router.  Thread safe and asynchronous. */
    void injectWin(const Id & auctionId,
                   const Id & adSpotId,
                   Amount winPrice,
                   Date timestamp,
                   const JsonHolder & winMeta,
                   const UserIds & uids,
                   const AccountKey & account,
                   Date bidTimestamp)
    {
        postAuctionLoop.injectWin(auctionId, adSpotId, winPrice,
                                  timestamp, winMeta, uids,
                                  account, bidTimestamp);
    }

    /** Inject a LOSS into the router.  Thread safe and asynchronous.
        Note that this method ONLY is useful for simulations; otherwise
        losses are implicit.
    */
    void injectLoss(const Id & auctionId,
                    const Id & adSpotId,
                    Date timestamp,
                    const JsonHolder & lossMeta,
                    const AccountKey & account,
                    Date bidTimestamp)
    {
        postAuctionLoop.injectLoss(auctionId, adSpotId, timestamp,
                                   lossMeta, account, bidTimestamp);
    }

    /** Inject an IMPRESSION into the router, to be passed on to the
        campaign that bid on it.
        
        If the spot ID is empty, then the click will be sent to all campaigns
        that had a win on the auction.
    */
    void injectCampaignEvent(const std::string & label,
                             const Id & auctionId,
                             const Id & spotId,
                             Date timestamp,
                             const JsonHolder & eventMeta,
                             const UserIds & uids)
    {
        postAuctionLoop.injectCampaignEvent(label, auctionId, spotId,
                                            timestamp, eventMeta, uids);
    }

    /** Place where the state persistence should be put. */
    void initStatePersistence(const std::string & path)
    {
        postAuctionLoop.initStatePersistence(path);
    }

    /** Add budget to the given account.  Returns the new accounting info
        for the account.
    */
    void addBudget(const AccountKey & account, CurrencyPool amount);
    
    /** Set the budget for the given account.  Returns the new accounting info
        for the account.
    */
    void setBudget(const AccountKey & account, CurrencyPool amount);
    
    void topupTransfer(const AccountKey & account, CurrencyPool amount);

    Router router;

    Redis::RedisTemporaryServer redis;
    MasterBanker masterBanker;
    SlaveBudgetController budgetController;

    PostAuctionService postAuctionLoop;
    AgentConfigurationService config;

    MonitorEndpoint monitor;

    bool initialized;
};



} // namespace RTBKIT



/* post_auction_proxy.h                                            -*- C++ -*-
   Jeremy Barnes, 18 December 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Base class to connect to an ad server.  We also have an http ad server
   connector that builds on top of this.
*/

#pragma once

#include "soa/service/zmq_endpoint.h"
#include "soa/types/id.h"
#include "rtbkit/common/currency.h"
#include "rtbkit/common/json_holder.h"
#include "rtbkit/common/bid_request.h"
#include "rtbkit/common/account_key.h"


namespace RTBKIT {


/*****************************************************************************/
/* POST AUCTION PROXY                                                        */
/*****************************************************************************/

/** Proxy class that formats messages to be sent to the post auction service.
*/

struct PostAuctionProxy {

    PostAuctionProxy(std::shared_ptr<zmq::context_t> context);

    virtual ~PostAuctionProxy();

    void init(std::shared_ptr<ConfigurationService> config);

    virtual void start();

    virtual void shutdown();

    /*************************************************************************/
    /* METHODS TO SEND MESSAGES ON                                           */
    /*************************************************************************/

    /** Inject a WIN into the post auction loop.  Thread safe and
        asynchronous. */
    void injectWin(const Id & auctionId,
                   const Id & adSpotId,
                   Amount winPrice,
                   Date timestamp,
                   const JsonHolder & winMeta,
                   const UserIds & ids,
                   const AccountKey & account,
                   Date bidTimestamp);

    /** Inject a LOSS into the router.  Thread safe and asynchronous.
        Note that this method ONLY is useful for simulations; otherwise
        losses are implicit.
    */
    void injectLoss(const Id & auctionId,
                    const Id & adSpotId,
                    Date timestamp,
                    const JsonHolder & lossMeta,
                    const AccountKey & account,
                    Date bidTimestamp);

    /** Inject an IMPRESSION into the router, to be passed on to the
        agent that bid on it.
        
        If the spot ID is empty, then the click will be sent to all
        agents that had a win on the auction.
    */
    void injectCampaignEvent(const std::string & label,
                             const Id & auctionId,
                             const Id & adSpotId,
                             Date timestamp,
                             const JsonHolder & impressionMeta,
                             const UserIds & ids);

private:
    // Connection to the post auction loops
    ZmqNamedProxy toPostAuctionService;

    // later... when we have multiple services
    //ZmqMultipleNamedClientBusProxy toPostAuctionServices;
};


} // namespace RTBKIT


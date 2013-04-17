/* adserver_connector.h                                            -*- C++ -*-
   Jeremy Barnes, 18 December 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Base class to connect to an ad server.  We also have an http ad server
   connector that builds on top of this.
*/

#pragma once

#include "soa/service/service_base.h"
#include "soa/service/zmq_endpoint.h"
#include "soa/types/id.h"
#include "rtbkit/common/currency.h"
#include "rtbkit/common/json_holder.h"
#include "rtbkit/common/bid_request.h"
#include "rtbkit/common/account_key.h"


namespace RTBKIT {


/*****************************************************************************/
/* ADSERVER CONNECTOR                                                        */
/*****************************************************************************/

struct AdServerConnector : public Datacratic::ServiceBase {
    AdServerConnector(const std::string & serviceName,
                      const std::shared_ptr<Datacratic::ServiceProxies> & proxy);
    virtual ~AdServerConnector();

    void init(std::shared_ptr<ConfigurationService> config);
    virtual void shutdown();

    virtual void start();

    void recordUptime() const;

    /*************************************************************************/
    /* METHODS TO SEND MESSAGES ON                                           */
    /*************************************************************************/

    /** Publish a WIN into the post auction loop.  Thread safe and
        asynchronous. */
    void publishWin(const Id & auctionId,
                    const Id & adSpotId,
                    Amount winPrice,
                    Date timestamp,
                    const JsonHolder & winMeta,
                    const UserIds & ids,
                    const AccountKey & account,
                    Date bidTimestamp);

    /** Publish a LOSS into the post auction loop. Thread safe and
        asynchronous. Note that this method ONLY is useful for simulations;
        otherwise losses are implicit.
    */
    void publishLoss(const Id & auctionId,
                     const Id & adSpotId,
                     Date timestamp,
                     const JsonHolder & lossMeta,
                     const AccountKey & account,
                     Date bidTimestamp);

    /** Publish a campaign event into the post auction loop, to be passed on
        to the agent that bid on it.
        
        If the spot ID is empty, then the click will be sent to all
        agents that had a win on the auction.
    */
    void publishCampaignEvent(const std::string & label,
                              const Id & auctionId,
                              const Id & adSpotId,
                              Date timestamp,
                              const JsonHolder & impressionMeta,
                              const UserIds & ids);

private:
    Date startTime_;

    // Connection to the post auction loops
    ZmqNamedProxy toPostAuctionService_;

    // later... when we have multiple services
    //ZmqMultipleNamedClientBusProxy toPostAuctionServices;
};

} // namespace RTBKIT

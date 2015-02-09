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
#include "rtbkit/common/post_auction_proxy.h"


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
    void publishWin(const Id & bidRequestId,
                    const Id & impId,
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
    void publishLoss(const Id & bidRequestId,
                     const Id & impId,
                     Date timestamp,
                     const JsonHolder & lossMeta,
                     const AccountKey & account,
                     Date bidTimestamp);

    /** Publish a campaign-based event into the post auction loop, to be
        passed on to the agent that bid on it.
        
        If the imp ID is empty, then the click will be sent to all
        agents that had a win on the auction.
    */
    void publishCampaignEvent(const std::string & label,
                              const Id & bidRequestId,
                              const Id & impId,
                              Date timestamp,
                              const JsonHolder & eventMeta,
                              const UserIds & ids);

    /** Publish an user-based event into the post auction loop.

        (TBD: currently only mark the event in Carbon)
    */
    void publishUserEvent(const std::string & label,
                          const Id & userId,
                          Date timestamp,
                          const JsonHolder & eventMeta,
                          const UserIds & ids);

    Date startTime_;

    static std::unique_ptr<AdServerConnector> create(std::string const & serviceName, std::shared_ptr<ServiceProxies> const & proxies,
                                                     Json::Value const & json);

    typedef std::function<AdServerConnector * (std::string const & serviceName,
					       std::shared_ptr<ServiceProxies> const & proxies,
                                               Json::Value const & json)> Factory;

    /** plugin interface needs to be able to request the root name of the plugin library */
    static const std::string libNameSufix() {return "adserver";};

    // FIXME: this is being kept just for compatibility reasons.
    // we don't want to break compatibility now, although this interface does not make
    // sense any longer  
    // so any use of it should be considered deprecated
    static void registerFactory(std::string const & name, Factory callback)
    {
      PluginInterface<AdServerConnector>::registerPlugin(name, callback);
    }


private:
    // Connection to the post auction loops
    PostAuctionProxy toPostAuctionService_;

    // later... when we have multiple services
    //ZmqMultipleNamedClientBusProxy toPostAuctionServices;
};

} // namespace RTBKIT

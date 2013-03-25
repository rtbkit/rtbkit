/* exchange_connector.h                                            -*- C++ -*-
   Jeremy Barnes, 13 December 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Exchange connector.
*/

#pragma once

#include "soa/service/service_base.h"
#include "rtbkit/common/auction.h"
#include "jml/utils/unnamed_bool.h"

namespace RTBKIT {

class Router;
class AgentConfig;
class Creative;


/*****************************************************************************/
/* EXCHANGE CONNECTOR                                                        */
/*****************************************************************************/

/** Base class to connect to exchanges.  This class is owned by a router.

    This provides:
    1.  Callbacks that can be used to inject an auction and a win into the
        router;
    2.  Interfaces for the router to control the exchange connector, such as
        cut off or throttle bids.
*/

struct ExchangeConnector: public ServiceBase {

    ExchangeConnector(const std::string & name,
                      ServiceBase & parent);

    ExchangeConnector(const std::string & name,
                      std::shared_ptr<ServiceProxies>
                          = std::shared_ptr<ServiceProxies>());

    virtual ~ExchangeConnector();

    /** Set the router used by the exchange connector. */
    void setRouter(Router * router);


    /*************************************************************************/
    /* METHODS CALLED BY THE ROUTER TO CONTROL THE EXCHANGE CONNECTOR        */
    /*************************************************************************/

    /** Configure the exchange connector.  The JSON provided is entirely
        interpreted by the exchange connector itself.
    */
    virtual void configure(const Json::Value & parameters) = 0;

    /** Start the exchange connector running */
    virtual void start();

    /** Shutdown the exchange connector ready to be destroyed. */
    virtual void shutdown();

    /** Set the time until which the exchange is enabled.  Normally this will
        be pushed forward a few seconds periodically so that everything will
        shut down if there is nothing controlling the exchange connector.
    */
    virtual void enableUntil(Date date) = 0;

    /** Set which percentage of bid requests will be accepted by the
        exchange connector.
    */
    virtual void setAcceptBidRequestProbability(double prob) = 0;

    /** Return the name of the exchange, as it would be written as an
        identifier.
    */
    virtual std::string exchangeName() const = 0;


    /*************************************************************************/
    /* EXCHANGE COMPATIBILITY                                                */
    /*************************************************************************/

    /* This functionality is used by the router to determine which campaigns
       may bid on inventory from the campaign, and which creatives are
       eligible to be shown to fill impressions for the campaign.

       This is where exchange-specific logic as to required information
       in the creative can be implemented, and allows feedback as to why
       a given campaign or creative is not working on an exchange for
       debugging purposes.
       
       Please note that these methods are called infrequently at campaign
       configuration time, and apply to *all* bid requests for each
       campaign.  Filtering of individual bid requests is done via
       the tags and filters mechanism.
    */

    /** Structure used to tell whether or not an exchange is compatible
        with a creative or campaign.
    */
    struct ExchangeCompatibility {
        ExchangeCompatibility()
            : isCompatible(false)
        {
        }

        JML_IMPLEMENT_OPERATOR_BOOL(isCompatible);

        /** Update to indicate that the exchange is compatible. */
        void setCompatible() { isCompatible = true;  reasons.clear(); }

        void setIncompatible() { isCompatible = false;  reasons.clear(); }

        /** Update to indicate that the exchange is incompatible for
            the given reason.
        */
        void setIncompatible(const std::string & reason, bool includeReasons)
        {
            isCompatible = false;
            if (includeReasons)
                reasons.push_back(reason);
        }

        /** Update to indicate that the exchange is incompatible for
            the given reasons.
        */
        void setIncompatible(ML::compact_vector<std::string, 1> && reasons,
                             bool includeReasons)
        {
            isCompatible = false;
            if (includeReasons)
                this->reasons = std::move(reasons);
        }

        bool isCompatible;   ///< Is it compatible?
        ML::compact_vector<std::string, 1> reasons;  ///< Reasons for incompatibility
        /** Exchange specific information about the creative or campaign, used
            by the exchange to cache results of eligibility and include pre-
            computed values for bidding.
        */
        std::shared_ptr<void> info;
    };

    /** Structure that tells whether a campaign itself, and each of its
        creatives, is compatible with the exchange.
    */
    struct CampaignCompatibility : public ExchangeCompatibility {
        CampaignCompatibility()
        {
        }
        
        CampaignCompatibility(const AgentConfig & config);

        std::vector<ExchangeCompatibility> creatives;  ///< Per-creative compatibility information
    };

    /** Given an agent configuration, return a structure that describes
        the compatibility of each campaign and creative with the
        exchange.

        If includeReasons is true, then the reasons structure should be
        filled in with a list of reasons for which the exchange rejected
        the creative or campaign.  If includeReasons is false, the reasons
        should be all empty to save memory allocations.  Note that it
        doesn't make much sense to have the reasons non-empty for creatives
        or campaigns that are approved.

        The default implementation assumes that all campaigns and
        creatives are compatible with the exchange.
    */
    virtual ExchangeCompatibility
    getCampaignCompatibility(const AgentConfig & config,
                             bool includeReasons) const;
    
    /** Tell if a given creative is compatible with the given exchange.
        See getCampaignCompatibility().
    */
    virtual ExchangeCompatibility
    getCreativeCompatibility(const Creative & creative,
                             bool includeReasons) const;


    /*************************************************************************/
    /* FACTORY INTERFACE                                                     */
    /*************************************************************************/

    /** Type of a callback which is registered as an exchange factory. */
    typedef std::function<ExchangeConnector * (Router * owner,std::string name)>
        Factory;
    
    /** Register the given exchange factory. */
    static void registerFactory(const std::string & exchange, Factory factory);

    /** Create a new exchange connector from a factory. */
    static std::unique_ptr<ExchangeConnector>
    create(const std::string & exchangeType,
           std::shared_ptr<Router> owner,
           const std::string & name);

    /** Start up a new exchange and connect it to the router.  The exchange
        will read its configuration from the given JSON blob.
    */
    static void startExchange(std::shared_ptr<Router> router,
                              const std::string & exchangeType,
                              const Json::Value & exchangeConfig);

    /** Start up a new exchange and connect it to the router.  The exchange
        will read its configuration and type from the given JSON blob.
    */
    static void startExchange(std::shared_ptr<Router> router,
                              const Json::Value & exchangeConfig);

protected:
    /** The router that the exchange is connected to. */
    Router * router;
};


} // namespace RTBKIT
    

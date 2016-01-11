/* exchange_connector.h                                            -*- C++ -*-
   Jeremy Barnes, 13 December 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Exchange connector.
*/

#pragma once

#include "soa/service/service_base.h"
#include "rtbkit/common/auction.h"
#include "rtbkit/common/win_cost_model.h"
#include "jml/utils/unnamed_bool.h"
#include "rtbkit/common/plugin_interface.h"

namespace RTBKIT {

class AgentConfig;
class Creative;
class BidSource;

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

    /** Function that will be called to notify of a new auction. */
    typedef boost::function<void (std::shared_ptr<Auction> Auction)>
        OnAuction;
    
    /** Callback for a) when there is a new auction, and b) when an auction
        is finished.

        These are used to hook the exchange connector into the router.
    */
    OnAuction onNewAuction, onAuctionDone;

    int numRequests;
    int numAuctions;

    /** Probability that we will accept a given auction. */
    double acceptAuctionProbability;

    typedef boost::function<void (const std::string & channel,
                                  std::shared_ptr<Auction> auction,
                                  const std::string & message)> OnAuctionError;
    OnAuctionError onAuctionError;

    /*************************************************************************/
    /* METHODS CALLED BY THE ROUTER TO CONTROL THE EXCHANGE CONNECTOR        */
    /*************************************************************************/

    /** Start logging requests */
    virtual void startRequestLogging(std::string const & filename, int count = 1000) {
    }

    /** Stop logging */
    virtual void stopRequestLogging() {
    }

    /** Configure the exchange connector.  The JSON provided is entirely
        interpreted by the exchange connector itself.
    */
    virtual void configure(const Json::Value & parameters);

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
    virtual void setAcceptBidRequestProbability(double prob)
    {
        if (prob < 0 || prob > 1)
            throw ML::Exception("invalid probability for "
                                "setBidRequestProbability: "
                                "%f is not between 0 and 1", prob);
        this->acceptAuctionProbability = prob;
    }

    /** Returns a function that can be used to sample the load of the exchange
        connector. See LoopMonitor documentation for more details.
     */
    virtual std::function<double(double)> getLoadSampleFn() const
    {
        return [] (double) { return 0.0; };
    }

    /** Return the name of the exchange, as it would be written as an
        identifier.
    */
    virtual std::string exchangeName() const = 0;

    /** Return the win cost model for the bid of an agent
    */
    virtual WinCostModel getWinCostModel(Auction const & auction,
                                         AgentConfig const & agent);

    /** Return a bid source
    */
    virtual std::string getBidSourceConfiguration() const;

    /*************************************************************************/
    /* EXCHANGE COMPATIBILITY                                                */
    /*************************************************************************/

    /* This functionality is used by the router to determine which campaigns
       may bid on inventory from the exchange, and which creatives are
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

    /** Structure is used to determine if the bid is valid.
    */
    struct BidValidity {
        BidValidity()
            :isValidbid(true)
        {
        }

        void setValidBid() {
            isValidbid = true;
        }

        void setInvalidBid(const std::string &reason) {
            isValidbid = false;
            reason_ = reason;
         }

        std::string  reason_;
        bool isValidbid;
    };

    /** Sets not valid bid if bid price is lower then the bidfloor price
    */
    virtual BidValidity
    getBidValidity(const Bid  & bid, const  std::vector<AdSpot> & imp,
                   int spotIndex) const;


    /*************************************************************************/
    /* FILTERING                                                             */
    /*************************************************************************/

    /** This is where the exchange can provide any extra filtering capability
        of its bid requests that is not exposed through RTBkit.

        This should mostly be used to implement restrictions, for example
        advertiser domain restrictions that come in with the bid request.

        In general, generic functionality applicable to multiple exchanges
        should not go here; only functionality specific to a given
        exchange.
    */

    /** Pre-filter a bid request according to the exchange's filtering
        rules.

        This function should return true if the given bidding agent is
        allowed to bid on the bid request, and false otherwise.  It should
        do any work that is not expensive.

        In order for a bid request to pass, it will have to pass the
        bidRequestPreFilter AND bidRequestPostFilter functions.  The only
        difference between the two is that the pre filter is called early
        in the filtering pipeline and should not do expensive computation,
        and the post filter is called later on (when the bid request stream
        has been further reduced) and can perform expensive computation.

        The default implementation will return true, which implements the
        policy that all bid requests are compatible with all bidding agents
        that are compatible with the exchange (see getCampaignCompatibility()
        above).

        \param request     The bid request being filtered
        \param config      The agent configuration for the agent being filtered
        \param info        The contents of the "info" field in the return
                           value of getCampaignCompatibility().  This can be
                           used to cache information to make this computation
                           more efficient.

        \seealso bidRequestPostFilter
    */
    virtual bool bidRequestPreFilter(const BidRequest & request,
                                     const AgentConfig & config,
                                     const void * info) const;

    /** Post-filter a bid request according to the exchange's filtering
        rules.

        This function should return true if the given bidding agent is
        allowed to bid on the bid request, and false otherwise.  It can
        perform expensive computations.

        In order for a bid request to pass, it will have to pass the
        bidRequestPreFilter AND bidRequestPostFilter functions.  The only
        difference between the two is that the pre filter is called early
        in the filtering pipeline and should not do expensive computation,
        and the post filter is called later on (when the bid request stream
        has been further reduced) and can perform expensive computation.

        The default implementation will return true, which implements the
        policy that all bid requests are compatible with all bidding agents
        that are compatible with the exchange (see getCampaignCompatibility()
        above).

        \param request     The bid request being filtered
        \param config      The agent configuration for the agent being filtered
        \param info        The contents of the "info" field in the return
                           value of getCampaignCompatibility().  This can be
                           used to cache information to make this computation
                           more efficient.

        \seealso bidRequestPreFilter
    */
    virtual bool bidRequestPostFilter(const BidRequest & request,
                                      const AgentConfig & config,
                                      const void * info) const;

    /** Filter a creative according to the exchange's filtering rules.

        This function should return true if the given creative is compatible
        with the given bid request, and false otherwise.
    */
    
    virtual bool bidRequestCreativeFilter(const BidRequest & request,
                                          const AgentConfig & config,
                                          const void * info) const;



    /*************************************************************************/
    /* FACTORY INTERFACE                                                     */
    /*************************************************************************/

    /** Type of a callback which is registered as an exchange factory. */
    typedef std::function<ExchangeConnector * (ServiceBase * owner, std::string name)>
        Factory;

    /** plugin interface needs to be able to request the root name of the plugin library */
    static const std::string libNameSufix() {return "exchange";};

    /** Register the given exchange factory. */
    template<typename T>
    static void registerFactory() {
        PluginInterface<ExchangeConnector>::registerPlugin(T::exchangeNameString(),
						       [](ServiceBase * owner,
							  std::string const & name) {
							  return new T(*owner, name);
					    });
    }

    /** Create a new exchange connector from a factory. */
    static std::unique_ptr<ExchangeConnector>
    create(const std::string & exchangeType,
           ServiceBase & owner,
           const std::string & name);

    const std::string& getCurrencyAsString() const
    {
        return currency_;
    }

    CurrencyCode getCurrency() const
    {
        return currencyCode_;
    }

    template <typename Ratio = Micro>
    auto getAmountIn(const Amount & amount)
        const -> decltype(RTBKIT::getAmountIn<Ratio>(CurrencyCode::CC_NONE, amount))
    {
        return RTBKIT::getAmountIn<Ratio>(getCurrency(), amount);
    }

    bool hasCurrencyConfigured() const
    {
        return hasCurrencyConfigured_;
    }

private:
    bool hasCurrencyConfigured_;
    std::string currency_;
    RTBKIT::CurrencyCode currencyCode_;
};


} // namespace RTBKIT
    

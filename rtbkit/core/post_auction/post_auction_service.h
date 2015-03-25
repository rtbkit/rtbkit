/** post_auction_service.h                                 -*- C++ -*-
    Rémi Attab, 18 Apr 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    Post auction service that matches bids to win and campaign events.

*/

#pragma once

#include "event_matcher.h"
#include "rtbkit/core/monitor/monitor_provider.h"
#include "rtbkit/core/agent_configuration/agent_configuration_listener.h"
#include "rtbkit/common/bidder_interface.h"
#include "soa/service/logs.h"
#include "soa/service/service_base.h"
#include "soa/service/loop_monitor.h"
#include "soa/service/zmq_endpoint.h"
#include "soa/service/zmq_named_pub_sub.h"
#include "soa/service/zmq_message_router.h"
#include "soa/service/rest_request_router.h"
#include "rtbkit/common/analytics_publisher.h"
#include "rtbkit/core/banker/local_banker.h"

namespace RTBKIT {

struct BidderInterface;
struct EventForwarder;

/******************************************************************************/
/* POST AUCTION SERVICE                                                       */
/******************************************************************************/

struct PostAuctionService : public ServiceBase, public MonitorProvider
{

    enum {
        DefaultWinLossPipeTimeout = 10,
        DefaultCampaignEventPipeTimeout = 10
    };

    PostAuctionService(ServiceBase & parent,
                    const std::string & serviceName);
    PostAuctionService(std::shared_ptr<ServiceProxies> proxies,
                    const std::string & serviceName);


    ~PostAuctionService() { shutdown(); }


    void initBidderInterface(Json::Value const & json);
    void init(size_t externalShard = 0, size_t internalShards = 1);
    void initAnalytics(const std::string & baseUrl, const int numConnections);
    void start(std::function<void ()> onStop = std::function<void ()>());
    void shutdown();

    /// Start listening on ports for connections from agents, routers
    /// and event sources
    void bindTcp();

    void addSource(std::string name, AsyncEventSource & source, int priority = 0)
    {
        loop.addSource(std::move(name), source, priority);
    }


    /************************************************************************/
    /* BANKER                                                               */
    /************************************************************************/

    std::shared_ptr<Banker> getBanker() const
    {
        return banker;
    }

    void setBanker(const std::shared_ptr<Banker> & newBanker)
    {
        matcher->setBanker(banker = newBanker);
        monitorProviderClient.addProvider(banker.get());
    }


    /**************************************************************************/
    /* TIMEOUTS                                                               */
    /**************************************************************************/

    void setWinTimeout(float timeout)
    {
        if (timeout < 0.0)
            throw ML::Exception("Invalid timeout for Win timeout");

        winTimeout = timeout;
        if (matcher) matcher->setWinTimeout(timeout);
    }

    void setAuctionTimeout(float timeout)
    {
        if (timeout < 0.0)
            throw ML::Exception("Invalid timeout for Auction timeout");

        auctionTimeout = timeout;
        if (matcher) matcher->setAuctionTimeout(timeout);
    }

    void setWinLossPipeTimeout(int timeout)
    {
        if (timeout < 0)
            throw ML::Exception("Invalid timeout for WinLoss Pipe timeout");

        winLossPipeTimeout = timeout;
    }

    void setCampaignEventPipeTimeout(int timeout)
    {
        if (timeout < 0)
            throw ML::Exception("Invalid timeout for Campaign Event Pipe timeout");

        campaignEventPipeTimeout = timeout;
    }

    /************************************************************************/
    /* LOGGING                                                              */
    /************************************************************************/

    /** Log a given message to the given channel. */
    template<typename... Args>
    void logMessage(const std::string & channel, Args... args)
    {
        logger.publish(channel, Date::now().print(5), args...);
    }

    /** Log a router error. */
    template<typename... Args>
    void logPAError(const std::string & function,
                    const std::string & exception,
                    Args... args)
    {
        logger.publish("PAERROR",
                Date::now().print(5), function, exception, args...);
        recordHit("error.%s", function);
    }


    /************************************************************************/
    /* EVENT MATCHING                                                       */
    /************************************************************************/

    /** Transfer the given auction to the post auction loop.  This method
        assumes that the given auction was submitted with a non-empty
        bid, and adds it to the internal data structures so that any
        post-auction messages can be matched up with it.
    */
    void injectSubmittedAuction(
            const Id & auctionId,
            const Id & adSpotId,
            std::shared_ptr<BidRequest> bidRequest,
            const std::string & bidRequestStr,
            const std::string & bidRequestStrFormat,
            const JsonHolder & augmentations,
            const Auction::Response & bidResponse,
            Date lossTimeout);

    /** Inject a WIN into the post auction loop.  Thread safe and
        asynchronous. */
    void injectWin(
            const Id & auctionId,
            const Id & adspot,
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
    void injectLoss(
            const Id & auctionId,
            const Id & adspot,
            Date timestamp,
            const JsonHolder & lossMeta,
            const AccountKey & account,
            Date bidTimestamp);

    /** Inject a campaign event into the router, to be passed on to the agent
        that bid on it.

        If the spot ID is empty, then the click will be sent to all agents
        that had a win on the auction.
    */
    void injectCampaignEvent(
            const std::string & label,
            const Id & auctionId,
            const Id & adSpotId,
            Date timestamp,
            const JsonHolder & eventMeta,
            const UserIds & ids);


    /************************************************************************/
    /* PERSISTENCE                                                          */
    /************************************************************************/

    void initStatePersistence(const std::string & path)
    {
        throw ML::Exception(
                "post auction service persistence is not yet implemented.");
    }


    /************************************************************************/
    /* STATS                                                                */
    /************************************************************************/

    struct Stats
    {
        size_t auctions;
        size_t events;

        size_t matchedWins;
        size_t matchedLosses;
        size_t matchedCampaignEvents;
        size_t unmatchedEvents;
        size_t errors;

        Stats();
        Stats(const Stats& other);
        Stats& operator=(const Stats& other);
        Stats& operator-=(const Stats& other);
        Stats& operator+=(const Stats& other);

    } stats;

    float sampleLoad() const { return loopMonitor.sampleLoad().load; }

    static Logging::Category print;
    static Logging::Category error;
    static Logging::Category trace;

    /************************************************************************/
    /* MISC                                                                 */
    /************************************************************************/

    void forwardAuctions(const std::string& uri);
    
private:

    std::string getProviderClass() const;
    MonitorIndicator getProviderIndicators() const;


    /** Initialize all of our connections, hooking everything in to the
        event loop.
    */
    void initConnections(size_t shard);
    void initMatcher(size_t shards);
    void initRestEndpoint();

    void doAuction(std::shared_ptr< SubmittedAuctionEvent> event);
    void doEvent(std::shared_ptr<PostAuctionEvent> event);
    void checkExpiredAuctions();

    /** Decode from zeromq and handle a new auction that came in. */
    void doAuctionMessage(const std::vector<std::string> & message);

    /** Decode from zeromq and handle a new auction that came in. */
    void doWinMessage(const std::vector<std::string> & message);

    /** Decode from zeromq and handle a new auction that came in. */
    void doLossMessage(const std::vector<std::string> & message);

    /** Decode from zeromq and handle a new campaign event message that came
     * in. */
    void doCampaignEventMessage(const std::vector<std::string> & message);

    void doConfigChange(
            const std::string & agent,
            std::shared_ptr<const AgentConfig> config);


    void doMatchedWinLoss(std::shared_ptr<MatchedWinLoss> event);
    void doMatchedCampaignEvent(std::shared_ptr<MatchedCampaignEvent> event);
    void doUnmatched(std::shared_ptr<UnmatchedEvent> event);
    void doError(std::shared_ptr<PostAuctionErrorEvent> error);

    void deliverEvent(const std::string& label, const std::string& eventType,
                      const AccountKey& account,
                      std::function<void(const AgentConfigEntry& entry)> onAgent);


    float auctionTimeout;
    float winTimeout;

    int winLossPipeTimeout;
    int campaignEventPipeTimeout;

    Date lastWinLoss;
    Date lastCampaignEvent;

    MessageLoop loop;
    LoopMonitor loopMonitor;

    std::unique_ptr<EventMatcher> matcher;
    std::shared_ptr<Banker> banker;
    AgentConfigurationListener configListener;
    MonitorProviderClient monitorProviderClient;

    TypedMessageSink<std::shared_ptr<SubmittedAuctionEvent> > auctions;
    TypedMessageSink<std::shared_ptr<PostAuctionEvent> > events;

    ZmqNamedPublisher logger;
    ZmqNamedEndpoint endpoint;

    std::shared_ptr<BidderInterface> bidder;
    AgentBridge bridge;

    ZmqMessageRouter router;

    AnalyticsPublisher analytics;

    std::unique_ptr<RestServiceEndpoint> restEndpoint;
    std::unique_ptr<RestRequestRouter> restRouter;

    std::shared_ptr<EventForwarder> forwarder;

    size_t totalEvents;
    size_t orphanEvents;
    std::vector<double> orphanRatios;

};

} // namespace RTBKIT

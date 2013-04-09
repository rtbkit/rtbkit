/* post_auction_loop.h                                             -*- C++ -*-
   Jeremy Barnes, 30 May 2012
   Router post-auction loop.
*/

#ifndef __router__post_auction_loop_h__
#define __router__post_auction_loop_h__

#include "soa/service/service_base.h"
#include "soa/service/pending_list.h"
#include <unordered_map>
#include "soa/service/message_loop.h"
#include "soa/service/typed_message_channel.h"
#include "rtbkit/common/auction.h"
#include "rtbkit/common/auction_events.h"
#include "soa/service/zmq_endpoint.h"
#include "soa/service/zmq_message_router.h"
#include "soa/service/zmq_named_pub_sub.h"
#include "rtbkit/core/agent_configuration/agent_configuration_listener.h"
#include "rtbkit/core/banker/banker.h"
#include "rtbkit/core/monitor/monitor_provider.h"

namespace RTBKIT {

/*****************************************************************************/
/* SUBMISSION INFO                                                           */
/*****************************************************************************/

/** Information we track (persistently) about an auction that has been
    submitted and for which we are waiting for information about whether
    it is won or not.
*/

struct SubmissionInfo {
    SubmissionInfo()
        : fromOldRouter(false)
    {
    }

    std::shared_ptr<BidRequest> bidRequest;
    std::string bidRequestStr;
    std::string bidRequestStrFormat;
    JsonHolder augmentations;
    Auction::Response  bid;               ///< Bid we passed on
    bool fromOldRouter;                   ///< Was reconstituted

    /** If the timeout races with the last bid or the router event loop
        is very busy (as it only processes timeouts when it is idle),
        it is possible that we get a WIN message before we have finished
        the acution.  In this case, we record that message here and replay
        it after the auction has finished.
    */
    std::vector<std::shared_ptr<PostAuctionEvent> > earlyWinEvents;
    std::vector<std::shared_ptr<PostAuctionEvent> > earlyCampaignEvents;

    std::string serializeToString() const;
    void reconstituteFromString(const std::string & str);
};


/*****************************************************************************/
/* FINISHED INFO                                                             */
/*****************************************************************************/

/** Information we track (persistently) about an auction that has finished
    (either won or lost).  We keep this around for an hour waiting for
    impressions, clicks or conversions; this structure contains the
    information necessary to join them up.
*/

struct FinishedInfo {
    FinishedInfo()
        : fromOldRouter(false)
    {
    }

    Date auctionTime;            ///< Time at which the auction started
    Id auctionId;       ///< Auction ID from host
    Id adSpotId;          ///< Spot ID from host
    int spotIndex;
    std::shared_ptr<BidRequest> bidRequest;  ///< What we bid on
    std::string bidRequestStr;
    std::string bidRequestStrFormat;
    JsonHolder augmentations;
    std::set<Id> uids;                ///< All UIDs for this user

    /** The set of channels that are associated with this request.  They
        are copied here from the winning agent's configuration so that
        we know how to filter and route the visits.
    */
    SegmentList visitChannels;

    /** Add all of the given UIDs to the set.
    */
    void addUids(const UserIds & toAdd)
    {
        for (auto it = toAdd.begin(), end = toAdd.end();  it != end;  ++it) {
            auto jt = uids.find(it->second);
            if (jt != uids.end())
                return;
            uids.insert(it->second);
        }
    }

    Date bidTime;                ///< Time at which we bid
    Auction::Response bid;       ///< Bid response
    Json::Value bidToJson() const;

    bool hasWin() const { return winTime != Date(); }
    void setWin(Date winTime, BidStatus status, Amount winPrice,
                const std::string & winMeta)
    {
        if (hasWin())
            throw ML::Exception("already has win");
        this->winTime = winTime;
        this->reportedStatus = status;
        this->winPrice = winPrice;
        this->winMeta = winMeta;
    }

    Date winTime;                ///< Time at which win received
    BidStatus reportedStatus;    ///< Whether we think we won it or lost it
    Amount winPrice;             ///< Win price
    std::string winMeta;         ///< Metadata from win
    Json::Value winToJson() const;

    CampaignEvents campaignEvents;

    struct Visit {
        Date visitTime;           ///< Time at which visit received
        SegmentList channels;     ///< Channel(s) associated with visit
        std::string meta;         ///< Visit metadata

        Json::Value toJson() const;
        void serialize(ML::DB::Store_Writer & store) const;
        void reconstitute(ML::DB::Store_Reader & store);
    };

    std::vector<Visit> visits;

    /** Add a visit to the visits array. */
    void addVisit(Date visitTime,
                  const std::string & visitMeta,
                  const SegmentList & channels);

    Json::Value visitsToJson() const;

    Json::Value toJson() const;

    bool fromOldRouter;

    std::string serializeToString() const;
    void reconstituteFromString(const std::string & str);
};


/*****************************************************************************/
/* POST AUCTION LOOP                                                         */
/*****************************************************************************/

struct PostAuctionLoop : public ServiceBase, public MonitorProvider
{

    PostAuctionLoop(ServiceBase & parent,
                    const std::string & serviceName);
    PostAuctionLoop(std::shared_ptr<ServiceProxies> proxies,
                    const std::string & serviceName);

    ~PostAuctionLoop()
    {
        shutdown();
    }

    std::shared_ptr<Banker> getBanker() const
    {
        return banker;
    }

    void setBanker(const std::shared_ptr<Banker> & newBanker)
    {
        banker = newBanker;
    }

    std::shared_ptr<Banker> banker;

    /* ROUTERSHARED */
    uint64_t numWins;
    uint64_t numLosses;
    uint64_t numCampaignEvents;

    /* /ROUTERSHARED */

    /* ROUTERBASE */
    /************************************************************************/
    /* EXCEPTIONS                                                           */
    /************************************************************************/

    /** Throw an exception and log the error in Graphite and in the router
        log file.
    */
    void throwException(const std::string & key, const std::string & fmt,
                        ...) __attribute__((__noreturn__));

    /************************************************************************/
    /* LOGGING                                                              */
    /************************************************************************/
    
    ZmqNamedPublisher logger;

    /** Log a given message to the given channel. */
    template<typename... Args>
    void logMessage(const std::string & channel, Args... args)
    {
        using namespace std;
        //cerr << "********* logging message to " << channel << endl;
        logger.publish(channel, Date::now().print(5), args...);
    }

    /** Log a router error. */
    template<typename... Args>
    void logPAError(const std::string & function,
                    const std::string & exception,
                    Args... args)
    {
        logger.publish("PAERROR", Date::now().print(5),
                       function, exception, args...);
        recordHit("error.%s", function);
    }

    /* /ROUTERBASE */

    /// Start listening on ports for connections from agents, routers
    /// and event sources
    void bindTcp();

    void init();

    void start(std::function<void ()> onStop = std::function<void ()>());

    void shutdown();

    size_t numAwaitingWinLoss() const
    {
        return submitted.size();
    }

    size_t numFinishedAuctionsTracked() const
    {
        return finished.size();
    }

    /** The post auction loop has state which needs to hang around for a
        long time.  We don't want this state to be lost if the post auction
        loop crashes, so we allow for it to be optionally saved to disk.

        This call will read any old state which is in the given directory,
        and also start recording state changes to that directory.

        It uses leveldb under the hood.
    */
    void initStatePersistence(const std::string & path);

    /** Return service status. */
    virtual Json::Value getServiceStatus() const;

    /** Transfer the given auction to the post auction loop.  This method
        assumes that the given auction was submitted with a non-empty
        bid, and adds it to the internal data structures so that any
        post-auction messages can be matched up with it.
    */
    void injectSubmittedAuction(const Id & auctionId,
                                const Id & adSpotId,
                                std::shared_ptr<BidRequest> bidRequest,
                                const std::string & bidRequestStr,
                                const std::string & bidRequestStrFormat,
                                const JsonHolder & augmentations,
                                const Auction::Response & bidResponse,
                                Date lossTimeout);

    /** Inject a WIN into the post auction loop.  Thread safe and
        asynchronous. */
    void injectWin(const Id & auctionId,
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
    void injectLoss(const Id & auctionId,
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
    void injectCampaignEvent(const std::string & label,
                             const Id & auctionId,
                             const Id & adSpotId,
                             Date timestamp,
                             const JsonHolder & eventMeta,
                             const UserIds & ids);

    /** Notify the loop that the given auction/spot will never receive
        another message and should be forgotten.  This is mostly for the
        simulation.

        This message is the preferred way for a simulation to notify the
        router that things are finished; normally auctions finish by
        themselves (once submitted) but the auction/spot combination stays
        alive much longer waiting for further messages.  You should only
        call finishedAuction in rare cases when you want to cancel an
        auction before bidding has finished.

        If the auction doesn't exist, the message will be silently ignored.

        There is no penalty for not calling this, apart from potentially
        having to wait one hour before the simulation decides that it is
        finished.
    */
    void notifyFinishedSpot(const Id & auctionId, const Id & adSpotId);

    /* Post service health status to Monitor */
    MonitorProviderClient monitorProviderClient;

    /* MonitorProvider interface */
    std::string getProviderName() const;
    Json::Value getProviderIndicators() const;

    Date lastWinLoss;
    Date lastCampaignEvent;

private:
    /** Initialize all of our connections, hooking everything in to the
        event loop.
    */
    void initConnections();

    /** Handle a new auction that came in. */
    void doAuction(const SubmittedAuctionEvent & event);

    /** Handle a post-auction event that came in. */
    void doEvent(const std::shared_ptr<PostAuctionEvent> & event);

    /** Decode from zeromq and handle a new auction that came in. */
    void doAuctionMessage(const std::vector<std::string> & message);

    /** Decode from zeromq and handle a new auction that came in. */
    void doWinMessage(const std::vector<std::string> & message);

    /** Decode from zeromq and handle a new auction that came in. */
    void doLossMessage(const std::vector<std::string> & message);

    /** Decode from zeromq and handle a new campaign event message that came
     * in. */
    void doCampaignEventMessage(const std::vector<std::string> & message);

    /** Periodic auction expiry. */
    void checkExpiredAuctions();

    /** We got a win/loss.  Match it up with its bid and pass on to the
        winning bidder.
    */
    void doWinLoss(const std::shared_ptr<PostAuctionEvent> & event,
                   bool isReplay);

    /** An auction was submitted... record that */
    void doSubmitted(const std::shared_ptr<PostAuctionEvent> & event);

    /** We got an impression or click on the control socket */
    void doCampaignEvent(const std::shared_ptr<PostAuctionEvent> & event);

    /** Send out a post-auction event to anything that may be listening. */
    bool routePostAuctionEvent(const std::string & label,
                               const FinishedInfo & finished,
                               const SegmentList & channels,
                               bool filterChannels);

    /** Communicate the result of a bid message to an agent. */
    void doBidResult(const Id & auctionId,
                     const Id & adSpotId,
                     const SubmissionInfo & submission,
                     Amount price,
                     Date timestamp,
                     BidStatus status,
                     const std::string & confidence,
                     const std::string & winLossMeta,
                     const UserIds & uids);

    /** List of auctions we're currently tracking as submitted.  Note that an
        auction may be both submitted and in flight (if we had submitted a bid
        from one agent but were waiting on bids for another agent).

        The key is the (auction id, spot id) pair since after submission,
        the result from every auction comes back separately.
    */
    typedef PendingList<std::pair<Id, Id>,
                        SubmissionInfo> Submitted;
    Submitted submitted;

    /** List of auctions we've won and we're waiting for a campaign event
        from, or otherwise we're keeping around in case a duplicate WIN or a
        campaign event message comes through, or otherwise we're looking for a
        late WIN message for.

        We keep this list around for 5 minutes for those that were lost,
        and one hour for those that were won.
    */
    typedef PendingList<std::pair<Id, Id>, FinishedInfo> Finished;
    Finished finished;

    /// This provides the thread we use to actually process with
    MessageLoop loop;

    /// Auctions come in on this when running in-process
    TypedMessageSink<SubmittedAuctionEvent> auctions;

    /// Events come in on this when running in-process
    TypedMessageSink<std::shared_ptr<PostAuctionEvent> > events;

    /// Endpoint that routers and event sources connect to
    ZmqNamedEndpoint endpoint;

    /// Object to route zeromq messages from the endpoint to the appropriate
    /// place
    ZmqMessageRouter router;

    /// Messages to the agents go out on this
    ZmqNamedClientBus toAgents;

    /** Send the given message to the given bidding agent. */
    template<typename... Args>
    void sendAgentMessage(const std::string & agent,
                          const std::string & messageType,
                          const Date & date,
                          Args... args)
    {
        toAgents.sendMessage(agent, messageType, date,
                             std::forward<Args>(args)...);
    }

    /** Turn an auction and agent into the bid ID for the banker */
    static std::string makeBidId(Id auctionId, Id spotId, const std::string & agent);

    AgentConfigurationListener configListener;
};


} // namespace RTBKIT


#endif /* __router__post_auction_loop_h__ */

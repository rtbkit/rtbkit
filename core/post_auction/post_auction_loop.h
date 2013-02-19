/* post_auction_loop.h                                             -*- C++ -*-
   Jeremy Barnes, 30 May 2012
   Router post-auction loop.
*/

#ifndef __router__post_auction_loop_h__
#define __router__post_auction_loop_h__

#include "rtbkit/core/router/router_base.h"
#include "soa/service/pending_list.h"
#include <unordered_map>
#include "soa/service/message_loop.h"
#include "soa/service/typed_message_channel.h"
#include <boost/shared_ptr.hpp>
#include "rtbkit/common/auction.h"
#include "soa/service/zmq_endpoint.h"
#include "soa/service/zmq_message_router.h"
#include "rtbkit/core/agent_configuration/agent_configuration_listener.h"
#include "rtbkit/core/router/augmentor_events_publisher.h"
#include "rtbkit/core/banker/banker.h"
#include "rtbkit/core/monitor/monitor_provider.h"

namespace RTBKIT {


/*****************************************************************************/
/* SUBMITTED AUCTION EVENT                                                   */
/*****************************************************************************/

/** When a submitted bid is transferred from the router to the post auction
    loop, it looks like this.
*/

struct SubmittedAuctionEvent {
    Id auctionId;                  ///< ID of the auction
    Id adSpotId;                   ///< ID of the adspot
    Date lossTimeout;              ///< Time at which a loss is to be assumed
    JsonHolder augmentations;      ///< Augmentations active
    std::shared_ptr<BidRequest> bidRequest;  ///< Bid request
    std::string bidRequestStr;     ///< Bid request as string on the wire
    Auction::Response bidResponse; ///< Bid response that was sent
    std::string bidRequestFormatStr;  ///< Format of stringified request(i.e "datacratic")

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);
};


/*****************************************************************************/
/* POST AUCTION EVENT TYPE                                                   */
/*****************************************************************************/

enum PostAuctionEventType {
    PAE_INVALID,
    PAE_WIN,
    PAE_LOSS,
    PAE_IMPRESSION,
    PAE_CLICK,
    PAE_VISIT
};

const char * print(PostAuctionEventType type);


/*****************************************************************************/
/* POST AUCTION EVENT                                                        */
/*****************************************************************************/

/** Holds an event that was submitted after an auction.  Needs to be
    possible to serialize/reconstitute as early events that haven't yet
    been matched may need to be saved until they can be matched up.
*/

struct PostAuctionEvent {
    PostAuctionEvent();

    PostAuctionEventType type;
    Id auctionId;
    Id adSpotId;
    Date timestamp;
    JsonHolder metadata;
    AccountKey account;
    Amount winPrice;
    UserIds uids;
    SegmentList channels;
    Date bidTimestamp;

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);

    std::string print() const;
};

std::ostream &
operator << (std::ostream & stream, const PostAuctionEvent & event);


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
    std::string bidRequestFormatStr;
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
    std::vector<std::shared_ptr<PostAuctionEvent> > earlyImpressionClickEvents;

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
    std::string bidRequestFormatStr;
    JsonHolder augmentations;
    std::set<Id> uids;                ///< All UIDs for this user

    /** The set of channels that are associated with this request.  They
        are copied here from the winning agent's configuration so that
        we know how to filter and route the visits.
    */
    SegmentList visitChannels;

    /** Add all of the given UIDs to the set, and for any new IDs call the
        given function.
    */
    template<typename Fn>
    void addUids(const UserIds & toAdd, Fn fn)
    {
        for (auto it = toAdd.begin(), end = toAdd.end();  it != end;  ++it) {
            auto jt = uids.find(it->second);
            if (jt != uids.end())
                return;
            uids.insert(it->second);
            fn(it->first, it->second);
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

    bool hasImpression() const { return impressionTime != Date(); }
    void setImpression(Date impressionTime,
                       const std::string & impressionMeta)
    {
        if (hasImpression())
            throw ML::Exception("already has impression");
        this->impressionTime = impressionTime;
        this->impressionMeta = impressionMeta;
    }
    Date impressionTime;         ///< Time at which impression received
    std::string impressionMeta;  ///< Metadata from impression
    Json::Value impressionToJson() const;

    bool hasClick() const { return clickTime != Date(); }
    void setClick(Date clickTime,
                  const std::string & clickMeta)
    {
        if (hasClick())
            throw ML::Exception("already has click");
        this->clickTime = clickTime;
        this->clickMeta = clickMeta;
    }

    Date clickTime;              ///< Time at which click received
    std::string clickMeta;       ///< Metadata from click
    Json::Value clickToJson() const;

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

struct PostAuctionLoop : public RouterServiceBase,
                         public MonitorProvider
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

    size_t numUidsTracked() const
    {
        return uidIndex.size();
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

    /** Inject an IMPRESSION into the router, to be passed on to the
        agent that bid on it.

        If the spot ID is empty, then the click will be sent to all
        agents that had a win on the auction.
    */
    void injectImpression(const Id & auctionId,
                          const Id & adSpotId,
                          Date timestamp,
                          const JsonHolder & impressionMeta,
                          const UserIds & ids);

    /** Inject a CLICK into the router, to be passed on to the agent that
        bid on it.

        If the spot ID is empty, then the click will be sent to all agents
        that had a win on the auction.
    */
    void injectClick(const Id & auctionId,
                     const Id & adSpotId,
                     Date timestamp,
                     const JsonHolder & clickMeta,
                     const UserIds & ids);

    /** Inject a VISIT into the router, to be passed onto any agent that is
        listening for the given visit ID.

        These are routed by matching the segments in the SegmentList
        for the agent configuration with the segments in this message.
    */
    void injectVisit(Date timestamp,
                     const SegmentList & segments,
                     const JsonHolder & visitMeta,
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

    /** Decode from zeromq nd handle a new impression message that came in. */
    void doImpressionMessage(const std::vector<std::string> & message);

    /** Decode from zeromq nd handle a new impression message that came in. */
    void doClickMessage(const std::vector<std::string> & message);

    /** Decode from zeromq nd handle a new impression message that came in. */
    void doVisitMessage(const std::vector<std::string> & message);

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
    void doImpressionClick(const std::shared_ptr<PostAuctionEvent> & event);

    /** We got a visit event on the control socket. */
    void doVisit(const std::shared_ptr<PostAuctionEvent> & event);

    /** Send out a post-auction event to anything that may be listening. */
    bool routePostAuctionEvent(PostAuctionEventType type,
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

    /** List of auctions we've won and we're waiting for an IMPRESSION
        or a CLICK message from, or otherwise we're keeping around in case
        a duplicate WIN or IMPRESSION or CLICK message comes through,
        or otherwise we're looking for a late WIN message for.

        We keep this list around for 5 minutes for those that were lost,
        and one hour for those that were won.
    */
    typedef PendingList<std::pair<Id, Id>, FinishedInfo> Finished;
    Finished finished;

    /** Map of user IDs to (auction, slot) pairs so that we can match our
        visits up to the original auctions.
    */
    struct UidIndexEntry {
        Id auctionId;
        Id adSpotId;
        Date timestamp;
    };

    // UserId -> ((auctionId, slotId) -> date)
    typedef std::unordered_map<Id, std::map<std::pair<Id, Id>, Date> >
        UidIndex;
    UidIndex uidIndex;

    /** Add the given User ID to the user ID index.  Calling this function
        sets things up such that a visit from the given user ID will be
        associated with the finished auction with the given auction Id and
        slot ID.
    */
    void addToUidIndex(const std::string & uidDomain,
                       const Id & uid,
                       const Id & auctionId,
                       const Id & slotId);

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

    /* Reponds to Monitor requests */
    MonitorProviderEndpoint monitorProviderEndpoint;

    /* MonitorProvider interface */
    Date lastWinLoss;
    Date lastImpression;

    Json::Value getMonitorIndicators();
};


} // namespace RTBKIT


#endif /* __router__post_auction_loop_h__ */

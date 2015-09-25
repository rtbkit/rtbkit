/* bidding_agent.h                                                   -*- C++ -*-
   Rémi Attab, 14 December 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Simple remote interface to the router.
*/


#ifndef __rtb__bidding_agent_h__
#define __rtb__bidding_agent_h__


#include "rtbkit/common/auction.h"
#include "rtbkit/common/bids.h"
#include "rtbkit/common/auction_events.h"
#include "rtbkit/common/win_cost_model.h"
#include "soa/service/zmq.hpp"
#include "soa/service/carbon_connector.h"
#include "soa/jsoncpp/json.h"
#include "soa/types/id.h"
#include "soa/service/service_base.h"
#include "soa/service/zmq_endpoint.h"
#include "soa/service/typed_message_channel.h"

#include <boost/function.hpp>
#include <boost/noncopyable.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/make_shared.hpp>
#include <string>
#include <vector>
#include <thread>
#include <map>


namespace RTBKIT {

/******************************************************************************/
/* BIDDING AGENT                                                              */
/******************************************************************************/

/** Proxy class that a bidding agent uses to communicate with the rest of the
    system (routers, post auction loops, agent configuration service).

    In order for a router to start sending bid request to an agent, the agent
    must first set up its configuration using the doConfig function. The bidding
    agent

*/

struct BiddingAgent : public ServiceBase, public MessageLoop {

    BiddingAgent(std::shared_ptr<ServiceProxies> proxies,
                 const std::string & name = "bidding_agent",
                 double maxAddedLatency = 0.002);
    BiddingAgent(ServiceBase & parent,
                 const std::string & name = "bidding_agent",
                 double maxAddedLatench = 0.002);

    ~BiddingAgent();

    BiddingAgent(const BiddingAgent& other) = delete;
    BiddingAgent& operator=(const BiddingAgent& other) = delete;

    /** Name of the agent which defaults to the <service name>_<pid>. This
        should be set before calling init().
     */
    std::string agentName;

    /** If set to true then an exception will thrown if a callback is not
        registered. Defaults to true.
    */
    void strictMode(bool strict) { requiresAllCB = strict; }

    void init();
    void shutdown();


    /**************************************************************************/
    /* AGENT CONTROLS                                                         */
    /**************************************************************************/

    /** Send a bid response to the router in answer to a received auction.

        \param id auction id given in the auction callback.
        \param response a Bids struct converted to json.
        \param meta A json blob that will be returned as is in the bid result.
        \param wcm win cost model for this bid.
     */
    void doBid(Id id, Bids bids,
                      const Json::Value& meta = Json::Value(),
                      const WinCostModel& wmc = WinCostModel());

    /** Notify the AgentConfigurationService that the configuration of the
        bidding agent has changed.

        Note that bidding agent will remember the given configuration which will
        be usedto answer any further configuration requests that are received.
        This function is thread-safe and can be called at any time to change the
        bid request which will be received by the agent. Note that update are
        done asynchronously and changes might not take effect immediately.
     */
    void doConfig(const AgentConfig& config);
    void doConfigJson(Json::Value config);


    /** Sent in response of a onPing callback and is used to notify the router
        that our agent is still alive and responsive. The fromRouter, sent and
        payload arguments should be passed as is from the callback arguments.
        The received argument should be the time at which the message was
        received.

        \todo received should be sampled by the doPong function.
     */
    void doPong(const std::string & fromRouter, Date sent, Date received,
                const std::vector<std::string> & payload);


    /**************************************************************************/
    /* CALLBACKS                                                              */
    /**************************************************************************/
    // The odd double typedefs is to simplify the JS wrappers.


    typedef void (BidRequestCb) (
            double timestamp,           // Start time of the auction.
            Id id,                      // Auction id
            std::shared_ptr<BidRequest> bidRequest,
            const Bids& bids,           // Impressions available for bidding
            double timeLeftMs,          // Time left of the bid request.
            Json::Value augmentations,  // Data from the augmentors.
            WinCostModel const & wcm);  // Win cost model.
    typedef boost::function<BidRequestCb> BidRequestCbFn;

    /** Called whenever bid request is received that matches the agent's
        filters specified in its configuration.

        To place a bid the target of this callback should call the doBid
        function with the id and the bids object. The router is in charge of
        enforcing the time constraint and will raise either the onTooLate or the
        onDroppedBid callback to notify the agent of late or missing bids.

        Note that it's recomended to always call the doBid function even if no
        bids are to be placed. Doing so will allow the router to resolve
        auctions more quickly and will therefor improve overall performances.

        Once the bids have been placed, one of the ResultCbFn callbacks will be
        triggered asynchronously to notify the agent of the result of the
        bid. In the case of a win, one of the DeliveryCbFn callbacks will be
        triggered asynchronously to notify the agent of delivery events. Note
        that it's possible for delivery callbacks to be triggered for an auction
        before its respective ResultCbFn callback.

        Note that bid results and delivery events will be triggered once per
        impression that was bid on. As an example, if we bid on 3 impressions in
        an auction with 5 available impressions then we will receive 3 bid
        result notification.
     */
    BidRequestCbFn onBidRequest;


    typedef void (ResultCb) (const BidResult & args);
    typedef boost::function<ResultCb> ResultCbFn;

    /** We won the auction and we should expect to receive delivery noticifation
        shortly if not already received. */
    ResultCbFn onWin;

    /** We won the auction and we should expect to receive delivery noticifation
        shortly if not already received. */
    ResultCbFn onLateWin;
    
    /** We lost either the internal router auction or the exchange auction. */
    ResultCbFn onLoss;

    /** No bids were placed because the the account for our agent does not
        contain enough funds. */
    ResultCbFn onNoBudget;

    /** No bids were placed because the agent placed its bid after the auction
        was sent back to the exchange. */
    ResultCbFn onTooLate;

    /** Triggered when the router did not receive a bid for given bid request.*/
    ResultCbFn onDroppedBid;

    /** An error was found in the placed bid. This will usually be triggered if
          an invalid creative or impression was selected for the bid. */
    ResultCbFn onInvalidBid;


    typedef void (DeliveryCb) (const DeliveryEvent & args);
    typedef boost::function<DeliveryCb> DeliveryCbFn;


    /** Triggered when an impression we bid on is shown to the user. */
    DeliveryCbFn onImpression;

    /** Triggered when a user clicks on one of our creatives. */
    DeliveryCbFn onClick;

    /** Triggered when a user visits the landing page of our creative.  */
    DeliveryCbFn onVisit;


    /** Triggered for all campaign event (click, impression, visit ...).  */
    DeliveryCbFn onCampaignEvent;



    typedef void (PingCb) (const std::string & fromRouter,
                           Date timestamp,
                           const std::vector<std::string> & args);
    typedef boost::function<PingCb> PingCbFn;

    /** Triggered periodically by the router to determine whether the agent is
        still alive and responsive.

        If overriden, then the target should call the doPong function with the
        fromRouter, timestamp and args field untouched. If not overriding, the
        agent is assumed to be responsive as long as this message loop is
        responsive.
     */
    PingCbFn onPing;


    typedef void (ErrorCb) (double timestamp,
                            std::string description,
                            std::vector<std::string> originalError);
    typedef boost::function<ErrorCb> ErrorCbFn;

    /** Triggered whenever  router receives an invalid message from the
        agent. This can either be caused by an invalid config, and invalid bid
     */
    ErrorCbFn onError;

    typedef void (ByebyeCb) (const std::string& fromRouter,
                             Date timestamp);
    typedef boost::function<ByebyeCb> ByebyeCbFn;

    /** Triggered whenever  router considers this agent as dead
     */
    ByebyeCbFn onByebye;



private:

    /** Format of a message to a router. */
    struct RouterMessage {
        RouterMessage(const std::string & toRouter = "",
                      const std::string & type = "",
                      const std::vector<std::string> & payload
                          = std::vector<std::string>())
            : toRouter(toRouter),
              type(type),
              payload(payload)
        {
        }

        std::string toRouter;
        std::string type;
        std::vector<std::string> payload;
    };

    ZmqMultipleNamedClientBusProxy toRouters;
    ZmqMultipleNamedClientBusProxy toPostAuctionServices;
    ZmqNamedClientBusProxy toConfigurationAgent;
    TypedMessageSink<RouterMessage> toRouterChannel;

    struct RequestStatus {
        Date timestamp;
        std::string fromRouter;
    };

    std::map<Id, RequestStatus> requests;
    std::mutex requestsLock; // Protects concurrent writes to requests

    bool requiresAllCB;


    /** Ensures that we can set the config and send it atomically. Prevents a
        situation where a call to the toConfigurationAgent's connectHandler
        callback would overwrite a newer configuration from a concurrent call to
        doConfig.
     */
    std::mutex configLock;
    std::string config; // The agent's configuration.
    AgentConfig agent_config;

    void sendConfig(const std::string& newConfig = "");

    void checkMessageSize(const std::vector<std::string>& msg, int expectedSize);

    // void doHeartbeat();

    void handleRouterMessage(const std::string & fromRouter,
                             const std::vector<std::string>& msg);
    void handleError(const std::vector<std::string>& msg, ErrorCbFn& callback);
    void handleBidRequest(const std::string & fromRouter,
            const std::vector<std::string>& msg, BidRequestCbFn& callback);
    void handleWin(
            const std::vector<std::string>& msg, ResultCbFn& callback);
    void handleResult(
            const std::vector<std::string>& msg, ResultCbFn& callback);
    void handleDelivery(
            const std::vector<std::string>& msg, DeliveryCbFn& callback);
    void handlePing(const std::string & fromRouter,
            const std::vector<std::string>& msg, PingCbFn& callback);
};



} // namespace RTBKIT

#endif // __rtb__bidding_agent_h__



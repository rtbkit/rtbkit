/* rtb_router.h                                                   -*- C++ -*-
   Jeremy Barnes, 13 March 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Router class for RTB.
*/

#ifndef __rtb__router_h__
#define __rtb__router_h__

#include "soa/service/zmq.hpp"
#include <unordered_map>
#include <boost/thread/thread.hpp>
#include <boost/scoped_ptr.hpp>
#include "jml/utils/filter_streams.h"
#include "soa/service/socket_per_thread.h"
#include "soa/service/timeout_map.h"
#include "soa/service/pending_list.h"
#include "augmentation_loop.h"
#include "router_types.h"
#include "soa/gc/gc_lock.h"
#include "jml/utils/ring_buffer.h"
#include "jml/arch/wakeup_fd.h"
#include "router_base.h"
#include <unordered_set>
#include <thread>
#include "rtbkit/plugins/exchange/exchange_connector.h"
#include "rtbkit/core/agent_configuration/blacklist.h"
#include "rtbkit/core/agent_configuration/agent_configuration_listener.h"
#include "rtbkit/core/agent_configuration/agent_config.h"
#include "rtbkit/core/monitor/monitor_provider.h"
#include "rtbkit/core/monitor/monitor_proxy.h"

namespace RTBKIT {

struct Banker;
struct BudgetController;
struct Accountant;


/*****************************************************************************/
/* AGENT INFO                                                                */
/*****************************************************************************/

/** A single entry in the agent info structure. */
struct AgentInfoEntry {
    std::string name;
    std::shared_ptr<const AgentConfig> config;
    std::shared_ptr<const AgentStatus> status;
    std::shared_ptr<AgentStats> stats;

    bool valid() const { return config && stats; }

    /** JSON version for debugging. */
    Json::Value toJson() const;
};


/** A read-only structure with information about all of the agents so
    that auctions can scan them without worrying about data dependencies.
    Uses RCU.
*/
struct AllAgentInfo : public std::vector<AgentInfoEntry> {
    std::unordered_map<std::string, int> agentIndex;
    std::unordered_map<AccountKey, std::vector<int> > accountIndex;
};


/*****************************************************************************/
/* ROUTER                                                                    */
/*****************************************************************************/

/** An RTB router.  Contains everything needed to run auctions. */

struct Router : public RouterServiceBase,
                public MonitorProvider
{

    Router(ServiceBase & parent,
           const std::string & serviceName = "router",
           double secondsUntilLossAssumed = 2.0,
           bool simulationMode = false,
           bool connectPostAuctionLoop = true);

    Router(std::shared_ptr<ServiceProxies> services,
           const std::string & serviceName = "router",
           double secondsUntilLossAssumed = 2.0,
           bool simulationMode = false,
           bool connectPostAuctionLoop = true);

    ~Router();

    double secondsUntilLossAssumed() const { return secondsUntilLossAssumed_; }

    void setSecondsUntilLossAssumed(double newValue)
    {
        if (newValue < 0.0)
            throw ML::Exception("invalid seconds until loss assumed");
        secondsUntilLossAssumed_ = newValue;
    }

    std::shared_ptr<Banker> getBanker() const;
    void setBanker(const std::shared_ptr<Banker> & newBanker);

    /** Initialize all of the internal data structures and configuration. */
    void init();

    /** Bind to TCP/IP ports and publish where to connect to. */
    void bindTcp();

    /** Bind a zeroMQ URI for the agent to listen on. */
    void bindAgents(std::string agentUri);

    /** Bind a zeroMQ URI to listen for augmentation messages on. */
    void bindAugmentors(const std::string & uri);

    /** Start the router running in a separate thread.  The given function
        will be called when the thread is stopped. */
    virtual void
    start(boost::function<void ()> onStop = boost::function<void ()>());

    /** Sleep until the router is idle (there are no more auctions and
        all connections are idle).
    */
    virtual void sleepUntilIdle();

    /** Simple logging method to output the current time on stderr. */
    void issueTimestamp();

    /** How many things (auctions, etc) are non-idle? */
    virtual size_t numNonIdle() const;
    
    virtual void shutdown();

    /** Iterate exchanges */
    template<typename F>
    void forAllExchanges(F functor) {
        for(auto & item : exchanges)
            functor(item);
    }

    /** Register the exchange */
    void addExchange(std::unique_ptr<ExchangeConnector> && exchange) {
        Guard guard(lock);
        exchanges.emplace_back(std::move(exchange));
    }

    /** Inject an auction into the router.
        auction:   the auction object
        lossTime:  the time at which, if no win was forthcoming, a loss will
                   be assumed and the message sent.  If the time is infinity
                   then no message will be sent; if it is equal to 0.0
                   then secondsSinceLossAssumed will be added to
                   the current time and that value used.
    */
    void injectAuction(std::shared_ptr<Auction> auction,
                       double lossTime = INFINITY);
    
    /** Inject an auction into the router given its components.
        
        onAuctionFinished: this is the callback that will be called once
                           the auction is finished
        id:                string ID to represent the auction.  Must be
                           globally unique for the router.
        request:           JSON string with the bid request
        startTime:         time at which the auction starts; if empty the
                           current time will be used
        expiryTime:        time at which the auction expires; if empty the
                           startTime + 30ms will be used.
        lossTime:  the time at which, if no win was forthcoming, a loss will
                   be assumed and the LOSS message sent.  If the time is
                   infinity then no message will be sent; if it is equal to
                   0.0 then secondsSinceLossAssumed will be added to
                   the current time and that value used.

        Returns the created auction object.

        Note that no error handling will be done on an invalid request;
        the auction will be silently dropped and the callback never called.

        This function does not deal with timeouts.  If you want the auction
        to time out, you should call notifyTooLate on the returned auction
        at the desired time, externally.
    */
    std::shared_ptr<Auction>
    injectAuction(Auction::HandleAuction onAuctionFinished,
                  std::shared_ptr<BidRequest> request,
                  const std::string & requestStr,
                  const std::string & requestStrFormat,
                  double startTime = 0.0,
                  double expiryTime = 0.0,
                  double lossTime = INFINITY);

    /** Notify the router that the given auction will never receive another
        message and should be forgotten.  This is mostly for simulation
        so that the router knows when it can shut down.

        Normally finishedSpot should be called, not finishedAuction.  In
        particular, this message will NOT remove anything from the finished
        auction structures, which is where auctions end up after a bid.

        There is no penalty for not calling this, apart from potentially
        having to wait one hour before the simulation decides that it is
        finished.

        If the auction doesn't exist, the message will be silently ignored.
    */
    void notifyFinishedAuction(const Id & auctionId);

    /** Return the number of auctions in progress. */
    int numAuctionsInProgress() const;

    /** Return the number of auctions awaiting a win/loss message. */
    int numAuctionsAwaitingResult() const;

    /** Return a stats object that tells us what's going on. */
    Json::Value getStats() const;

    /** Return information about a given agent. */
    Json::Value getAgentInfo(const std::string & agent) const;

    /** Return information about all agents. */
    Json::Value getAllAgentInfo() const;

    /** Return information about all agents bidding on the given
        account. */
    Json::Value getAccountInfo(const AccountKey & account) const;
    

    /** Enter simulation mode.  No real-time time source is used. */
    void enterSimulationMode();

    /** Multiplier for the bid probability of all agents. */
    void setGlobalBidProbability(double val) { globalBidProbability = val; }
    
    /** Proportion of bids that should be rejected with an arbitrary 
        error.
    */
    void setBidsErrorRate(double val) { bidsErrorRate = val; }

    /** Proportion of bids that should be rejected with an out of budget 
        error. 
    */
    void setBudgetErrorRate(double val) { budgetErrorRate = val; }

    /** Overwrite this function such that it causes the number of auctions
        coming in to be throttled.
    */
    boost::function<void (double)> acceptAuctionProbabilityFn;

    /** Auction accept probability */
    void setAcceptAuctionProbability(double val)
    {
        if (acceptAuctionProbabilityFn)
            acceptAuctionProbabilityFn(val);
        else if (val != 1.0)
            std::cerr << "warning: no way to change accept auction probability" << std::endl;
    }

    /** Return service status. */
    virtual Json::Value getServiceStatus() const;

    /** Function to override if other behaviour than sending a response to
        the post auction loop is desired.
    */
    std::function<void (std::shared_ptr<Auction>, Id, Auction::Response)>
        onSubmittedAuction;

    /** Function to pass a submitted auction on to the post auction loop.
     */
    virtual void submitToPostAuctionService(std::shared_ptr<Auction> auction,
                                            Id auctionId,
                                            const Auction::Response & bid);

protected:
    // This thread contains the main router loop
    boost::scoped_ptr<boost::thread> runThread;

    // This thread wakes up every now and again to run the destructors
    // on auction objects, which are expensive to destroy, so that they
    // don't have to run in the main loop
    boost::scoped_ptr<boost::thread> cleanupThread;

    typedef std::recursive_mutex Lock;
    typedef std::unique_lock<Lock> Guard;

    int shutdown_;

public:
    // Connection to the agents
    ZmqNamedClientBus agentEndpoint;

    // Connection to the post auction loop
    ZmqNamedProxy postAuctionEndpoint;

    void updateAllAgents();

    /** Map from the configured name of the agent to the agent info. */
    typedef std::map<std::string, AgentInfo> Agents;
    Agents agents;

    ML::RingBufferSRMW<std::pair<std::string, std::shared_ptr<const AgentConfig> > > configBuffer;
    ML::RingBufferSRMW<std::shared_ptr<AugmentationInfo> > startBiddingBuffer;
    ML::RingBufferSRMW<std::shared_ptr<Auction> > submittedBuffer;
    ML::RingBufferSWMR<std::shared_ptr<Auction> > auctionGraveyard;

    ML::Wakeup_Fd wakeupMainLoop;

    AugmentationLoop augmentationLoop;
    Blacklist blacklist;

    /** List of auctions we're currently tracking as active. */
    typedef TimeoutMap<Id, AuctionInfo> InFlight;
    InFlight inFlight;

    /** Add the given auction to our data structures. */
    AuctionInfo &
    addAuction(std::shared_ptr<Auction> auction, Date timeout);

    DutyCycleEntry dutyCycleCurrent;
    std::vector<DutyCycleEntry> dutyCycleHistory;

    void run();

    void handleAgentMessage(const std::vector<std::string> & message);

    void checkDeadAgents();

    void checkExpiredAuctions();

    void returnErrorResponse(const std::vector<std::string> & message,
                             const std::string & error);

    void doShutdown();

    /** Perform initial auction processing to see how it can be used.  Returns a
        null pointer if the auction has no potential bidders.

        This can be called from any thread.
    */
    std::shared_ptr<AugmentationInfo>
    preprocessAuction(const std::shared_ptr<Auction> & auction) const;

    /** Send the auction for augmentation.  Once that is done, doStartBidding
        will be called.
    */
    void augmentAuction(const std::shared_ptr<AugmentationInfo> & info);

    /** We've finished augmenting our auctions.  Allow the agents to bid
        on them.
    */
    void doStartBidding(const std::vector<std::string> & message);

    /** Ditto but taking the augmented auction directly. */
    void doStartBidding(const std::shared_ptr<AugmentationInfo> & augInfo);

    /** Auction has been submitted.  Do the final cleanup here and send
        it off to the post auction loop. */
    void doSubmitted(std::shared_ptr<Auction> auction);

    //std::unordered_set<Id> recentlySubmitted;  // DEBUG

    /** An agent bid on an auction.  Arrange for this bid to be recorded. */
    void doBid(const std::vector<std::string> & message);

    /** An agent responded to a ping message.  Arrange for the ping time
        to be recorded. */
    void doPong(int level, const std::vector<std::string> & message);

    /** Send out a "ping" message to each agent, and interpret the results
        of the previous set of pings (do we need to throttle down?)
    */
    void sendPings();

    /** Someone wants stats. */
    void doStats(const std::vector<std::string> & message);

    /** We got a new auction. */
    void onNewAuction(std::shared_ptr<Auction> auction);

    /** An auction finished. */
    void onAuctionDone(std::shared_ptr<Auction> auction);

    /** Got a configuration message; update our internal data structures */
    void doConfig(const std::string & agent,
                  std::shared_ptr<const AgentConfig> config);

    /** Remove the given agent (with the given configuration) from the
        configuration structures.
    */
    void unconfigure(const std::string & agent, const AgentConfig & config);

    /** Add the given agent (with the given configuration) to the
        configuration structures.
    */
    void configure(const std::string & agent, const AgentConfig & config);

    /** Send the given message to the given bidding agent. */
    template<typename... Args>
    void sendAgentMessage(const std::string & agent,
                          const std::string & messageType,
                          const Date & date,
                          Args... args)
    {
        agentEndpoint.sendMessage(agent, messageType, date, args...);
    }

    /** Send the given bid response to the given bidding agent. */
    void sendBidResponse(const std::string & agent,
                         const AgentInfo & info,
                         BidStatus status,
                         Date timestamp,
                         const std::string & message,
                         const Id & auctionId,
                         int spotNum = -1,
                         Amount price = Amount(),
                         const Auction * auction = 0,
                         const std::string & bidData = "",
                         const Json::Value & unused1 = Json::Value(),
                         const Json::Value & metadata = Json::Value(),
                         const std::string & augmentationsStr = "");
                         

    mutable Lock lock;

    std::shared_ptr<Banker> banker;

    double secondsUntilLossAssumed_;
    double globalBidProbability;
    double bidsErrorRate;
    double budgetErrorRate;
    bool connectPostAuctionLoop;


    /*************************************************************************/
    /* AGENT INTERACTIONS                                                    */
    /*************************************************************************/

    /** Pointer to current version.  Protected by allAgentsGc. */
    AllAgentInfo * allAgents;

    /** RCU protection for allAgents. */
    mutable GcLock allAgentsGc;

    typedef std::function<void (const AgentInfoEntry & info)> OnAgentFn;
    /** Call the given callback for each agent. */
    void forEachAgent(const OnAgentFn & onAgent) const;

    /** Call the given callback for each agent that is bidding on the given
        account.
    */
    void forEachAccountAgent(const AccountKey & account,
                             const OnAgentFn & onAgent) const;

    /** Find the information entry for the given agent.  All elements are
        guaranteed to be valid until the object is destroyed.
    */
    AgentInfoEntry getAgentEntry(const std::string & agent) const;

    /** Listen for changes in configuration and let the router know about
        them.
    */
    AgentConfigurationListener configListener;

    /** Are we initialized? */
    bool initialized;

    std::vector<std::unique_ptr<ExchangeConnector> > exchanges;

    /* Client connection to the Monitor, determines if we can process bid
       requests */
    MonitorProxy monitorProxy;
    Date slowModeLastAuction;
    int slowModeCount;

    /* MONITOR PROVIDER */
    /* Reponds to Monitor requests */
    MonitorProviderEndpoint monitorProviderEndpoint;

    /* MonitorProvider interface */
    Json::Value getMonitorIndicators();
};


} // namespace RTBKIT


#endif /* __rtb__router_h__ */

/* rtb_router.h                                                   -*- C++ -*-
   Jeremy Barnes, 13 March 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Router class for RTB.
*/

#ifndef __rtb__router_h__
#define __rtb__router_h__

#include <atomic>
#include "filter_pool.h"
#include "soa/service/zmq.hpp"
#include <unordered_map>
#include <boost/thread/thread.hpp>
#include <boost/scoped_ptr.hpp>
#include "jml/utils/filter_streams.h"
#include "soa/service/zmq_named_pub_sub.h"
#include "soa/service/socket_per_thread.h"
#include "soa/service/timeout_map.h"
#include "soa/service/pending_list.h"
#include "soa/service/loop_monitor.h"
#include "augmentation_loop.h"
#include "router_types.h"
#include "soa/gc/gc_lock.h"
#include "jml/utils/ring_buffer.h"
#include "jml/arch/wakeup_fd.h"
#include "jml/utils/smart_ptr_utils.h"
#include <unordered_set>
#include <thread>
#include "rtbkit/common/exchange_connector.h"
#include "rtbkit/common/post_auction_proxy.h"
#include "rtbkit/common/analytics_publisher.h"
#include "rtbkit/core/agent_configuration/blacklist.h"
#include "rtbkit/core/agent_configuration/agent_configuration_listener.h"
#include "rtbkit/core/agent_configuration/agent_config.h"
#include "rtbkit/core/monitor/monitor_provider.h"
#include "rtbkit/core/monitor/monitor_client.h"

namespace RTBKIT {

struct Banker;
struct BudgetController;
struct Accountant;
struct BidderInterface;

/*****************************************************************************/
/* AGENT INFO                                                                */
/*****************************************************************************/

/** A single entry in the agent info structure. */
struct AgentInfoEntry {
    std::string name;
    unsigned filterIndex;
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
/* DEBUG INFO                                                                */
/*****************************************************************************/

struct AuctionDebugInfo {
    void addAuctionEvent(Date date, std::string type,
                         const std::vector<std::string> & args);
    void addSpotEvent(const Id & spot, Date date, std::string type,
                      const std::vector<std::string> & args);
    void dumpAuction() const;
    void dumpSpot(Id spot) const;

    struct Message {
        Date timestamp;
        Id spot;
        std::string type;
        std::vector<std::string> args;
    };

    std::vector<Message> messages;
};

/*****************************************************************************/
/* ROUTER                                                                    */
/*****************************************************************************/

/** An RTB router.  Contains everything needed to run auctions. */

struct Router : public ServiceBase,
                public MonitorProvider
{
    Router(ServiceBase & parent,
           const std::string & serviceName = "router",
           double secondsUntilLossAssumed = 2.0,
           bool connectPostAuctionLoop = true,
           bool enableBidProbability = true,
           bool logAuctions = false,
           bool logBids = false,
           Amount maxBidAmount = USD_CPM(40),
           int secondsUntilSlowMode = MonitorClient::DefaultCheckTimeout,
           Amount slowModeAuthorizedMoneyLimit = USD_CPM(100),
           Seconds augmentationWindow = std::chrono::milliseconds(5),
           int numAugmentationLoops = 1);

    Router(std::shared_ptr<ServiceProxies> services = std::make_shared<ServiceProxies>(),
           const std::string & serviceName = "router",
           double secondsUntilLossAssumed = 2.0,
           bool connectPostAuctionLoop = true,
           bool enableBidProbability = true,
           bool logAuctions = false,
           bool logBids = false,
           Amount maxBidAmount = USD_CPM(40),
           int secondsUntilSlowMode = MonitorClient::DefaultCheckTimeout,
           Amount slowModeAuthorizedMoneyLimit = USD_CPM(100),
           Seconds augmentationWindow = std::chrono::milliseconds(5),
           int numAugmentationLoops = 1);

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

    /** Initialize the bidder interface. */
    void initBidderInterface(Json::Value const & json);

    /** Initialize analytics if it is used. */
    void initAnalytics(const std::string & baseUrl, const int numConnections);

    /** Initialize exchages from json configuration. */
    void initExchanges(const Json::Value & config);

    /** Initialize filters from json configuration. */
    void initFilters(const Json::Value & config = Json::Value::null);

    /** Initialize all of the internal data structures and configuration. */
    void init();

    /** Bind to TCP/IP ports and publish where to connect to. */
    void bindTcp();

    /** Bind a zeroMQ URI for the agent to listen on. */
    void bindAgents(std::string agentUri);

    /** Bind a zeroMQ URI to listen for augmentation messages on. */
    void bindAugmentors(const std::string & uri);

    /** Disable the monitor for testing purposes.  In production this could lead
        to unbounded overspend, so please do really only use it for testing.
    */
    void unsafeDisableMonitor();

    /** Disable the auction probability for testing purposes.  In production this could lead
        to unbounded overspend, so please do really only use it for testing.
    */
    void unsafeDisableAuctionProbability();

    /** Disable the monitor client
    */
    void unsafeDisableSlowMode();

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

    /** Connect the exchange connector to the router, but do not make the router
        know about or own the exchange.

        Used mostly for testing where we want to control the exchange connector
        objects independently of the router.

        This method should almost never be used, as the given exchange will
        not participate in validation of bidding agent configuration.
    */
    void connectExchange(ExchangeConnector & exchange)
    {
        exchange.onNewAuction  = [=] (std::shared_ptr<Auction> a) { this->injectAuction(a, secondsUntilLossAssumed_); };
        exchange.onAuctionDone = [=] (std::shared_ptr<Auction> a) { this->onAuctionDone(a); };
        exchange.onAuctionError = [=] (const std::string & channel,
                                       std::shared_ptr<Auction> auction,
                                       const std::string message) { this->onAuctionError(channel, auction, message); };
    }

    /** Register the exchange with the router and make it take ownership of it */
    void addExchange(ExchangeConnector * exchange)
    {
        loopMonitor.addCallback(
                "exchanges." + exchange->exchangeName(),
                exchange->getLoadSampleFn());

        Guard guard(lock);
        exchanges.push_back(std::shared_ptr<ExchangeConnector>(exchange));
        connectExchange(*exchange);
    }

    /** Register the exchange with the router.  The router will not take
        ownership of the exchange, which means that it needs to be
        freed by the calling code after the router has exited.
    */
    void addExchange(ExchangeConnector & exchange)
    {
        loopMonitor.addCallback(
                "exchanges." + exchange.exchangeName(),
                exchange.getLoadSampleFn());

        Guard guard(lock);
        exchanges.emplace_back(ML::make_unowned_std_sp(exchange));
        connectExchange(exchange);
    }

    /** Register the exchange */
    void addExchange(std::unique_ptr<ExchangeConnector> && exchange)
    {
        addExchange(exchange.release());
    }

    /** Register the exchange */
    void addExchange(std::shared_ptr<ExchangeConnector> const & exchange)
    {
        loopMonitor.addCallback(
                "exchanges." + exchange->exchangeName(),
                exchange->getLoadSampleFn());

        Guard guard(lock);
        exchanges.push_back(exchange);
        connectExchange(*exchange);
    }

    void addExchangeNoConnect(std::shared_ptr<ExchangeConnector> const & exchange)
    {
        loopMonitor.addCallback(
                "exchanges." + exchange->exchangeName(),
                exchange->getLoadSampleFn());

        Guard guard(lock);
        exchanges.push_back(exchange);
    }

    /** Start up a new exchange from type and configuration from the given JSON blob. */
    void initExchange(const std::string & exchangeType,
                       const Json::Value & exchangeConfig);

    /** Init a new exchange from configuration and type from the given JSON blob. */
    void initExchange(const Json::Value & exchangeConfig);

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
        requestStr:        JSON string with the bid request
        requestStrFormat:  format of the bid request
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

    /** Auction accept probability */
    void setAcceptAuctionProbability(double val)
    {
        Guard guard(lock);

        for (auto& exchange : exchanges)
            exchange->setAcceptBidRequestProbability(val);
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
    // Connection to the post auction loop
    PostAuctionProxy postAuctionEndpoint;

    void updateAllAgents();

    /** Map from the configured name of the agent to the agent info. */
    typedef std::map<std::string, AgentInfo> Agents;
    Agents agents;

    ML::RingBufferSRMW<std::pair<std::string, std::shared_ptr<const AgentConfig> > > configBuffer;
    ML::RingBufferSRMW<std::shared_ptr<ExchangeConnector> > exchangeBuffer;
    ML::RingBufferSRMW<std::shared_ptr<AugmentationInfo> > startBiddingBuffer;
    ML::RingBufferSRMW<std::shared_ptr<Auction> > submittedBuffer;
    ML::RingBufferSWMR<std::shared_ptr<Auction> > auctionGraveyard;
    ML::RingBufferSRMW<BidMessage> doBidBuffer;

    ML::Wakeup_Fd wakeupMainLoop;

    FilterPool filters;

    AugmentationLoop augmentationLoop;
    Blacklist blacklist;

    LoopMonitor loopMonitor;
    LoadStabilizer loadStabilizer;

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

    void returnInvalidBid(const std::string &agent, const std::string &bidData,
                          const std::shared_ptr<Auction> &auction,
                          const char *reason, const char *message, ...);

    void returnInvalidBid(const std::string &agent, const std::string &bidData,
                          const std::shared_ptr<Auction> &auction,
                          const std::string &reason, const char *message, ...);
    void doShutdown();

    /** Perform initial auction processing to see how it can be used.  Returns a
        null pointer if the auction has no potential bidders.

        This can be called from any thread.
    */
    std::shared_ptr<AugmentationInfo>
    preprocessAuction(const std::shared_ptr<Auction> & auction);

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

    void doBidImpl(const BidMessage &message,
                   const std::vector<std::string> &originalMessage = std::vector<std::string>());

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

    /** An auction error. */
    void onAuctionError(const std::string & channel,
                        std::shared_ptr<Auction> auction,
                        const std::string & message);

    /** Got a configuration message; update our internal data structures */
    void doConfig(const std::string & agent,
                  std::shared_ptr<const AgentConfig> config);

    /* Add a given agent (with the given configuration) to the exchange */
    void configureAgentOnExchange(std::shared_ptr<ExchangeConnector> const & exchange,
                                  std::string const & agent,
                                  AgentConfig & config,
                                  bool includeReasons = true);

    /** Remove the given agent (with the given configuration) from the
        configuration structures.
    */
    void unconfigure(const std::string & agent, const AgentConfig & config);

    /** Add the given agent (with the given configuration) to the
        configuration structures.
    */
    void configure(const std::string & agent, AgentConfig & config);

    mutable Lock lock;

    std::shared_ptr<Banker> banker;

    double secondsUntilLossAssumed_;
    double globalBidProbability;
    double bidsErrorRate;
    double budgetErrorRate;
    bool connectPostAuctionLoop;
    bool enableBidProbability;


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

    /** List of exchanges that are active. */
    std::vector<std::shared_ptr<ExchangeConnector> > exchanges;

    /** Bid price calculator */
    std::shared_ptr<BidderInterface> bidder;
    AgentBridge bridge;

    /*************************************************************************/
    /* EXCEPTIONS                                                            */
    /*************************************************************************/

    /** Throw an exception and log the error in Graphite and in the router
        log file.
    */
    void throwException(const std::string & key, const std::string & fmt,
                        ...) __attribute__((__noreturn__));


    /*************************************************************************/
    /* SYSTEM LOGGING                                                        */
    /*************************************************************************/

    /** Log a router error. */
    template<typename... Args>
    void logRouterError(const std::string & function,
                        const std::string & exception,
                        Args... args)
    {
        logger.publish("ROUTERERROR", Date::now().print(5),
                       function, exception, args...);
        analytics.publish("ROUTERERROR", Date::now().print(5),
                       function, exception, args...);
        recordHit("error.%s", function);
    }


    /************************************************************************/
    /* USAGE METRICS                                                        */
    /************************************************************************/

    struct AgentUsageMetrics {
        AgentUsageMetrics()
            : intoFilters(0), passedStaticFilters(0), passedDynamicFilters(0),
              auctions(0), bids(0)
        {}

        AgentUsageMetrics(uint64_t intoFilters,
                          uint64_t passedStaticFilters,
                          uint64_t passedDynamicFilters,
                          uint64_t auctions,
                          uint64_t bids)
            : intoFilters(intoFilters),
              passedStaticFilters(passedStaticFilters),
              passedDynamicFilters(passedDynamicFilters),
              auctions(auctions),
              bids(bids)
        {}

        AgentUsageMetrics operator - (const AgentUsageMetrics & other)
            const
        {
            AgentUsageMetrics result(*this);

            result.intoFilters -= other.intoFilters;
            result.passedStaticFilters -= other.passedStaticFilters;
            result.passedDynamicFilters -= other.passedDynamicFilters;
            result.auctions -= other.auctions;
            result.bids -= other.bids;

            return result;
        };

        uint64_t intoFilters;
        uint64_t passedStaticFilters;
        uint64_t passedDynamicFilters;
        uint64_t auctions;
        uint64_t bids;
    };

    struct RouterUsageMetrics {
        RouterUsageMetrics()
            : numRequests(0), numAuctions(0), numNoPotentialBidders(0),
              numBids(0), numAuctionsWithBid(0)
        {}

        RouterUsageMetrics operator - (const RouterUsageMetrics & other)
            const
        {
            RouterUsageMetrics result(*this);

            result.numRequests -= other.numRequests;
            result.numAuctions -= other.numAuctions;
            result.numNoPotentialBidders -= other.numNoPotentialBidders;
            result.numBids -= other.numBids;
            result.numAuctionsWithBid -= other.numAuctionsWithBid;

            return result;
        };

        uint64_t numRequests;
        uint64_t numAuctions;
        uint64_t numNoPotentialBidders;
        uint64_t numBids;
        uint64_t numAuctionsWithBid;
    };

    void logUsageMetrics(double period);

    std::map<std::string, AgentUsageMetrics> lastAgentUsageMetrics;
    RouterUsageMetrics lastRouterUsageMetrics;

    /*************************************************************************/
    /* DATA LOGGING                                                          */
    /*************************************************************************/

    /** Log auctions */
    bool logAuctions;

    /** Log bids */
    bool logBids;

    /** Log a given message to the given channel. */
    template<typename... Args>
    void logMessage(const std::string & channel, Args... args)
    {
        using namespace std;
        //cerr << "********* logging message to " << channel << endl;
        logger.publish(channel, Date::now().print(5), args...);
    }

    /** Log a given message to analytics endpoint on given channel. */
    template<typename... Args>
    void logMessageToAnalytics(const std::string & channel, Args... args)
    {
        analytics.publish(channel, Date::now().print(5), args...);
    }

    /** Log a given message to the given channel. */
    template<typename... Args>
    void logMessageNoTimestamp(const std::string & channel, Args... args)
    {
        using namespace std;
        //cerr << "********* logging message to " << channel << endl;
        logger.publish(channel, args...);
    }

    /*************************************************************************/
    /* DEBUGGING                                                             */
    /*************************************************************************/

    void debugAuction(const Id & auctionId, const std::string & type,
                      const std::vector<std::string> & args
                      = std::vector<std::string>())
    {
        if (JML_LIKELY(!doDebug)) return;
        debugAuctionImpl(auctionId, type, args);
    }

    void debugAuctionImpl(const Id & auctionId, const std::string & type,
                          const std::vector<std::string> & args);

    void debugSpot(const Id & auctionId,
                   const Id & spotId,
                   const std::string & type,
                   const std::vector<std::string> & args
                       = std::vector<std::string>())
    {
        if (JML_LIKELY(!doDebug)) return;
        debugSpotImpl(auctionId, spotId, type, args);
    }

    void debugSpotImpl(const Id & auctionId,
                       const Id & spotId,
                       const std::string & type,
                       const std::vector<std::string> & args);

    void expireDebugInfo();

    void dumpAuction(const Id & auctionId) const;
    void dumpSpot(const Id & auctionId, const Id & spotId) const;

    Date getCurrentTime() const { return Date::now(); }

    ZmqNamedPublisher logger;
    AnalyticsPublisher analytics;

    /** Debug only */
    bool doDebug;

    /* Disable auction probability for testing only : don't drop any BR*/
    bool disableAuctionProb;

    mutable ML::Spinlock debugLock;
    TimeoutMap<Id, AuctionDebugInfo> debugInfo;

    uint64_t numAuctions;
    uint64_t numBids;
    uint64_t numNonEmptyBids;
    uint64_t numAuctionsWithBid;
    uint64_t numNoPotentialBidders;
    uint64_t numNoBidders;

    /* Client connection to the Monitor, determines if we can process bid
       requests */
    MonitorClient monitorClient;
    // TODO Make this thread safe
    Date slowModeLastAuction;
    std::atomic<bool> slowModePeriodicSpentReached;
    Amount slowModeAuthorizedMoneyLimit;
    uint64_t accumulatedBidMoneyInThisPeriod;

    /* MONITOR PROVIDER */
    /* Post service health status to Monitor */
    MonitorProviderClient monitorProviderClient;

    Amount maxBidAmount;

    /* MonitorProvider interface */
    std::string getProviderClass() const;
    MonitorIndicator getProviderIndicators() const;

    double slowModeTolerance;
    Seconds augmentationWindow;

    int numAugmentationLoops;
    std::vector<std::shared_ptr<AugmentationLoop>> augmentationLoops;
};


} // namespace RTBKIT


#endif /* __rtb__router_h__ */

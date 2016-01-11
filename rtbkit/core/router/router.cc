/* rtb_router.cc
   Jeremy Barnes, 24 March 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   RTB router code.
*/

#include <atomic>
#include "router.h"
#include "soa/service/zmq_utils.h"
#include "jml/arch/backtrace.h"
#include "jml/arch/futex.h"
#include "jml/arch/exception_handler.h"
#include "soa/jsoncpp/writer.h"
#include <boost/foreach.hpp>
#include "jml/arch/atomic_ops.h"
#include "jml/utils/set_utils.h"
#include "jml/utils/environment.h"
#include "jml/arch/info.h"
#include "jml/utils/lightweight_hash.h"
#include "jml/math/xdiv.h"
#include <boost/tuple/tuple.hpp>
#include "jml/utils/pair_utils.h"
#include "jml/utils/exc_assert.h"
#include "jml/db/persistent.h"
#include "jml/utils/json_parsing.h"
#include "profiler.h"
#include "rtbkit/core/banker/banker.h"
#include "rtbkit/core/banker/null_banker.h"
#include <boost/algorithm/string.hpp>
#include "rtbkit/common/bids.h"
#include "rtbkit/common/auction_events.h"
#include "rtbkit/common/messages.h"
#include "rtbkit/common/win_cost_model.h"
#include "rtbkit/common/bidder_interface.h"

using namespace std;
using namespace ML;


namespace RTBKIT {

/*****************************************************************************/
/* AGENT INFO                                                                */
/*****************************************************************************/

Json::Value
AgentInfoEntry::
toJson() const
{
    Json::Value result;
    if (!valid()) return result;
    result["config"] = config->toJson();
    result["stats"] = stats->toJson();
    return result;
}

/*****************************************************************************/
/* AUCTION DEBUG INFO                                                        */
/*****************************************************************************/

void
AuctionDebugInfo::
addAuctionEvent(Date timestamp, std::string type,
                const std::vector<std::string> & args)
{
    Message message;
    message.timestamp = timestamp;
    message.type = type;
    //message.args = args;
    messages.push_back(message);
}

void
AuctionDebugInfo::
addSpotEvent(const Id & spot, Date timestamp, std::string type,
             const std::vector<std::string> & args)
{
    Message message;
    message.spot = spot;
    message.timestamp = timestamp;
    message.type = type;
    //message.args = args;
    messages.push_back(message);
}

void
AuctionDebugInfo::
dumpAuction() const
{
    for (unsigned i = 0;  i < messages.size();  ++i) {
        auto & m = messages[i];
        cerr << m.timestamp.print(6) << " " << m.spot << " " << m.type << endl;
    }
}

void
AuctionDebugInfo::
dumpSpot(Id spot) const
{
    dumpAuction();  // TODO
}


/*****************************************************************************/
/* ROUTER                                                                    */
/*****************************************************************************/

Router::
Router(ServiceBase & parent,
       const std::string & serviceName,
       double secondsUntilLossAssumed,
       bool connectPostAuctionLoop,
       bool enableBidProbability,
       bool logAuctions,
       bool logBids,
       Amount maxBidAmount,
       int secondsUntilSlowMode,
       Amount slowModeAuthorizedMoneyLimit,
       Seconds augmentationWindow)
    : ServiceBase(serviceName, parent),
      shutdown_(false),
      postAuctionEndpoint(*this),
      configBuffer(1024),
      exchangeBuffer(64),
      startBiddingBuffer(65536),
      submittedBuffer(65536),
      auctionGraveyard(65536),
      doBidBuffer(65536),
      augmentationLoop(*this),
      loopMonitor(*this),
      loadStabilizer(loopMonitor),
      secondsUntilLossAssumed_(secondsUntilLossAssumed),
      globalBidProbability(1.0),
      bidsErrorRate(0.0),
      budgetErrorRate(0.0),
      connectPostAuctionLoop(connectPostAuctionLoop),
      enableBidProbability(enableBidProbability),
      allAgents(new AllAgentInfo()),
      configListener(getZmqContext()),
      initialized(false),
      bridge(getZmqContext()),
      logAuctions(logAuctions),
      logBids(logBids),
      logger(getZmqContext()),
      doDebug(false),
      disableAuctionProb(false),
      numAuctions(0), numBids(0), numNonEmptyBids(0),
      numAuctionsWithBid(0), numNoPotentialBidders(0),
      numNoBidders(0),
      monitorClient(getZmqContext(), secondsUntilSlowMode),
      slowModePeriodicSpentReached(false),
      slowModeAuthorizedMoneyLimit(slowModeAuthorizedMoneyLimit),
      accumulatedBidMoneyInThisPeriod(0),
      monitorProviderClient(getZmqContext()),
      maxBidAmount(maxBidAmount),
      slowModeTolerance(MonitorClient::DefaultTolerance),
      augmentationWindow(augmentationWindow)
{
    monitorProviderClient.addProvider(this);
}

Router::
Router(std::shared_ptr<ServiceProxies> services,
       const std::string & serviceName,
       double secondsUntilLossAssumed,
       bool connectPostAuctionLoop,
       bool enableBidProbability,
       bool logAuctions,
       bool logBids,
       Amount maxBidAmount,
       int secondsUntilSlowMode,
       Amount slowModeAuthorizedMoneyLimit,
       Seconds augmentationWindow)
    : ServiceBase(serviceName, services),
      shutdown_(false),
      postAuctionEndpoint(*this),
      configBuffer(1024),
      exchangeBuffer(64),
      startBiddingBuffer(65536),
      submittedBuffer(65536),
      auctionGraveyard(65536),
      doBidBuffer(65536),
      augmentationLoop(*this),
      loopMonitor(*this),
      loadStabilizer(loopMonitor),
      secondsUntilLossAssumed_(secondsUntilLossAssumed),
      globalBidProbability(1.0),
      bidsErrorRate(0.0),
      budgetErrorRate(0.0),
      connectPostAuctionLoop(connectPostAuctionLoop),
      enableBidProbability(enableBidProbability),
      allAgents(new AllAgentInfo()),
      configListener(getZmqContext()),
      initialized(false),
      bridge(getZmqContext()),
      logAuctions(logAuctions),
      logBids(logBids),
      logger(getZmqContext()),
      doDebug(false),
      disableAuctionProb(false),
      numAuctions(0), numBids(0), numNonEmptyBids(0),
      numAuctionsWithBid(0), numNoPotentialBidders(0),
      numNoBidders(0),
      monitorClient(getZmqContext(), secondsUntilSlowMode),
      slowModePeriodicSpentReached(false),
      slowModeAuthorizedMoneyLimit(slowModeAuthorizedMoneyLimit),
      accumulatedBidMoneyInThisPeriod(0),
      monitorProviderClient(getZmqContext()),
      maxBidAmount(maxBidAmount),
      slowModeTolerance(MonitorClient::DefaultTolerance),
      augmentationWindow(augmentationWindow)

{
    monitorProviderClient.addProvider(this);
}

void
Router::
initBidderInterface(Json::Value const & json)
{
    bidder = BidderInterface::create(serviceName() + ".bidder", getServices(), json);
    bidder->init(&bridge, this);
    bidder->registerLoopMonitor(&loopMonitor);
}

void
Router::
initAnalytics(const string & baseUrl, const int numConnections)
{
    analytics.init(baseUrl, numConnections);
}

void
Router::
initExchanges(const Json::Value & config) {
    for (auto & exchange: config) {
        initExchange(exchange);
    }
}

void
Router::
initExchange(const std::string & type,
              const Json::Value & config)
{
    auto exchange = ExchangeConnector::create(type, *this, type);
    exchange->configure(config);

    std::shared_ptr<ExchangeConnector> item(exchange.release());
    addExchangeNoConnect(item);

    exchangeBuffer.push(item);
}

void
Router::
initExchange(const Json::Value & exchangeConfig)
{
    std::string exchangeType = exchangeConfig["exchangeType"].asString();
    initExchange(exchangeType, exchangeConfig);
}

void
Router::
initFilters(const Json::Value & config) {

    if (config != Json::Value::null) {

        Json::Value extraFilterFiles = config["extraFilterFiles"];
        if (extraFilterFiles != Json::Value::null) {
            if (!extraFilterFiles.isArray()) {
                throw Exception("Filter files must be an array");
            }
            for(size_t i=0; i<extraFilterFiles.size(); i++){
                std::string file="lib"+extraFilterFiles[i].asString()+".so";
                void * handle = dlopen(file.c_str(),RTLD_NOW);
                if (!handle) {
                    std::cerr << dlerror() << std::endl;
                    throw ML::Exception("couldn't load library from %s", file.c_str());
                }
            }
        }

        Json::Value filterMask = config["filterMask"];
        if(filterMask != Json::Value::null) {
            if (!filterMask.isArray()) {
                throw Exception("Filter mask must be an array");
            }
            filters.initWithFiltersFromJson(filterMask);
        } else {
            filters.initWithDefaultFilters();
        }

    } else {
        filters.initWithDefaultFilters();
    }
}

void
Router::
init()
{
    ExcAssert(!initialized);

    registerServiceProvider(serviceName(), { "rtbRequestRouter" });

    filters.init(this);

    banker.reset(new NullBanker());

    if(!bidder) {
        Json::Value json;
        json["type"] = "agents";
        initBidderInterface(json);
    }

    augmentationLoop.init();

    logger.init(getServices()->config, serviceName() + "/logger");

    bridge.agents.init(getServices()->config, serviceName() + "/agents");
    bridge.agents.clientMessageHandler
        = std::bind(&Router::handleAgentMessage, this, std::placeholders::_1);
    bridge.agents.onConnection = [=] (const std::string & agent)
        {
            cerr << "agent " << agent << " connected to router" << endl;
        };

    bridge.agents.onDisconnection = [=] (const std::string & agent)
        {
            cerr << "agent " << agent << " disconnected from router" << endl;
        };

    configListener.onConfigChange = [=] (const std::string & agent,
                                         std::shared_ptr<const AgentConfig> config)
        {
            configBuffer.push(make_pair(agent, config));
        };

    onSubmittedAuction = [=] (std::shared_ptr<Auction> auction,
                              Id adSpotId,
                              Auction::Response response)
        {
            submitToPostAuctionService(auction, adSpotId, response);
        };

    monitorClient.init(getServices()->config);
    monitorProviderClient.init(getServices()->config);

    loopMonitor.init();
    loopMonitor.addMessageLoop("augmentationLoop", &augmentationLoop);
    loopMonitor.addMessageLoop("logger", &logger);
    loopMonitor.addMessageLoop("configListener", &configListener);
    loopMonitor.addMessageLoop("monitorClient", &monitorClient);
    loopMonitor.addMessageLoop("monitorProviderClient", &monitorProviderClient);
    if (analytics.initialized) loopMonitor.addMessageLoop("analytics", &analytics);

    loopMonitor.onLoadChange = [=] (double)
        {
            double keepProb = 1.0;

            if(!disableAuctionProb) {
                keepProb -= loadStabilizer.shedProbability();
            }

            setAcceptAuctionProbability(keepProb);
            recordEvent("auctionKeepPercentage", ET_LEVEL, keepProb * 100.0);
        };

    initialized = true;
}

Router::
~Router()
{
    shutdown();
}

std::shared_ptr<Banker>
Router::
getBanker() const
{
    return banker;
}

void
Router::
setBanker(const std::shared_ptr<Banker> & newBanker)
{
    banker = newBanker;
    monitorProviderClient.addProvider(banker.get());
}

void
Router::
bindTcp()
{
    logger.bindTcp(getServices()->ports->getRange("logs"));
    bridge.agents.bindTcp(getServices()->ports->getRange("router"));
}

void
Router::
bindAgents(std::string agentUri)
{
    try {
        bridge.agents.bind(agentUri.c_str());
    } catch (const std::exception & exc) {
        throw Exception("error while binding agent URI %s: %s",
                            agentUri.c_str(), exc.what());
    }
}

void
Router::
bindAugmentors(const std::string & uri)
{
    try {
        augmentationLoop.bindAugmentors(uri);
    } catch (const std::exception & exc) {
        throw Exception("error while binding augmentation URI %s: %s",
                        uri.c_str(), exc.what());
    }
}

void
Router::
unsafeDisableMonitor()
{
    // TODO: we shouldn't be reaching inside these structures...
    monitorClient.testMode = true;
    monitorClient.testResponse = true;
    monitorProviderClient.disable();
}

void
Router::
unsafeDisableSlowMode()
{
    monitorClient.testMode = true;
    monitorClient.testResponse = true;
}

void
Router::
unsafeDisableAuctionProbability()
{
    disableAuctionProb = true;
}

void
Router::
start(boost::function<void ()> onStop)
{
    ExcAssert(initialized);

    static Lock lock;
    Guard guard(lock);

    if (runThread)
        throw Exception("router is already running");

    auto runfn = [=] ()
        {
            this->run();
            if (onStop) onStop();
        };

    for ( auto & exchange : exchanges) {
        exchange->start();
        connectExchange(*exchange);
    }

    bidder->start();
    logger.start();
    analytics.start();
    augmentationLoop.start();
    runThread.reset(new boost::thread(runfn));

    if (connectPostAuctionLoop) {
        postAuctionEndpoint.init();
    }

    configListener.init(getServices()->config);
    configListener.start();

    /* This is an extra thread which sits there deleting auctions
       to take this out of the hands of the main loop (it can easily use
       up nearly 20% of the capacity of the main loop).
    */
    auto auctionDeleter = [=] ()
        {
            while (!this->shutdown_) {
                std::shared_ptr<Auction> toDelete;
                auctionGraveyard.tryPop(toDelete, 0.05);
                int numDeleted = 1;
                while (this->auctionGraveyard.tryPop(toDelete))
                    ++numDeleted;
                //cerr << "deleted " << numDeleted << " auctions"
                //     << endl;
                ML::sleep(0.001);
            }
        };

    cleanupThread.reset(new boost::thread(auctionDeleter));

    monitorClient.start();
    monitorProviderClient.start();

    loopMonitor.start();
}

size_t
Router::
numNonIdle() const
{
    size_t numInFlight, numAwaitingAugmentation;
    {
        Guard guard(lock);
        numInFlight = inFlight.size();
        numAwaitingAugmentation = augmentationLoop.numAugmenting();
    }

    cerr << "numInFlight = " << numInFlight << endl;
    cerr << "numAwaitingAugmentation = " << numAwaitingAugmentation << endl;

    return numInFlight + numAwaitingAugmentation;
}

void
Router::
sleepUntilIdle()
{
    for (int iter = 0;;++iter) {
        augmentationLoop.sleepUntilIdle();
        size_t nonIdle = numNonIdle();
        if (nonIdle == 0) break;
        //cerr << "there are " << nonIdle << " non-idle" << endl;
        ML::sleep(0.001);
    }
}

void
Router::
issueTimestamp()
{
    Date now = Date::now();

    cerr << "timestamp: "
         << ML::format("%.6f", now.secondsSinceEpoch())
         << " - " << now.printClassic()
         << endl;
}

void
Router::
run()
{
    using namespace std;

    zmq_pollitem_t items [] = {
        { bridge.agents.getSocketUnsafe(), 0, ZMQ_POLLIN, 0 },
        { 0, wakeupMainLoop.fd(), ZMQ_POLLIN, 0 }
    };

    double last_check = ML::wall_time(), last_check_pace = last_check,
        lastPings = last_check;

    //cerr << "server listening" << endl;

    auto getTime = [&] () { return Date::now().secondsSinceEpoch(); };

    double beforeSleep, afterSleep = getTime();
    int numTimesCouldSleep = 0;
    int totalSleeps = 0;
    double lastTimestamp = 0;

    double totalActive = 0;
    double lastTotalActive = 0; // member variable for the lambda.
    loopMonitor.addCallback("routerLoop",
            [&, lastTotalActive] (double elapsed) mutable {
                double delta = totalActive - lastTotalActive;
                lastTotalActive = totalActive;
                return delta / elapsed;
            });

    recordHit("routerUp");

    //double lastDump = ML::wall_time();

    struct TimesEntry {
        TimesEntry()
            : time(0.0), count(0)
        {
        };

        void add(double time)
        {
            this->time += time;
            ++count;
        }

        double time;
        uint64_t count;
    };

    std::map<std::string, TimesEntry> times;
    auto recordTime = [&] (std::string name, double start) {
        times[std::move(name)].add(microsecondsBetween(getTime(), start));
    };


    // Attempt to wake up once per millisecond

    Date lastSleep = Date::now();

    while (!shutdown_) {
        beforeSleep = getTime();
        totalActive += beforeSleep - afterSleep;
        dutyCycleCurrent.nsProcessing += microsecondsBetween(beforeSleep, afterSleep);

        int rc = 0;

        {
            double atStart = getTime();

            for (unsigned i = 0;  i < 20 && rc == 0;  ++i)
                rc = zmq_poll(items, 2, 0);

            recordTime("spinPoll", atStart);
        }

        if (rc == 0) {
            ++numTimesCouldSleep;

            {
                double atStart = getTime();
                checkExpiredAuctions();
                recordTime("checkExpiredAuctions", atStart);
            }

            {
                double atStart = getTime();

                // Try to sleep only once per 1/2 a millisecond to avoid too
                // many context switches.
                Date now = Date::now();
                double timeSinceSleep = lastSleep.secondsUntil(now);
                double timeToWait = 0.0005 - timeSinceSleep;
                if (timeToWait > 0) {
                    ML::sleep(timeToWait);
                }
                lastSleep = now;

                recordTime("sleep", atStart);
            }

            double pollStart = getTime();
            rc = zmq_poll(items, 2, 50 /* milliseconds */);
            recordTime("sleepPoll", pollStart);
        }

        afterSleep = getTime();

        dutyCycleCurrent.nsSleeping += microsecondsBetween(afterSleep, beforeSleep);
        dutyCycleCurrent.nEvents += 1;

        if (rc == -1 && zmq_errno() != EINTR) {
            cerr << "zeromq error: " << zmq_strerror(zmq_errno()) << endl;
        }

        {
            double atStart = getTime();
            std::shared_ptr<AugmentationInfo> info;
            while (startBiddingBuffer.tryPop(info)) {
                doStartBidding(info);
            }

            recordTime("doStartBidding", atStart);
        }

        {
            double atStart = getTime();

            BidMessage message;
            while (doBidBuffer.tryPop(message)) {
                doBidImpl(message);
            }

            recordTime("doBid", atStart);
        }

        {
            double atStart = getTime();

            std::shared_ptr<ExchangeConnector> exchange;
            while (exchangeBuffer.tryPop(exchange)) {
                for (auto & agent : agents) {
                    configureAgentOnExchange(exchange,
                                             agent.first,
                                             *agent.second.config);
                };
            }

            recordTime("configureAgentOnExchange", atStart);
        }

        {
            double atStart = getTime();

            std::pair<std::string, std::shared_ptr<const AgentConfig> > config;
            while (configBuffer.tryPop(config)) {
                doConfig(config.first, config.second);
            }

            recordTime("doConfig", atStart);
        }

        {
            double atStart = getTime();

            std::shared_ptr<Auction> auction;
            while (submittedBuffer.tryPop(auction))
                doSubmitted(auction);

            recordTime("doSubmitted", atStart);
        }

        if (items[0].revents & ZMQ_POLLIN) {
            double atStart = getTime();
            // Agent message
            vector<string> message;
            try {
                message = recvAll(bridge.agents.getSocketUnsafe());
                bridge.agents.handleMessage(std::move(message));

            } catch (const std::exception & exc) {
                cerr << "error handling agent message " << message
                     << ": " << exc.what() << endl;
                logRouterError("handleAgentMessage", exc.what(),
                               message);
            }

            recordTime(message.at(1), atStart);
        }

        if (items[1].revents & ZMQ_POLLIN) {
            wakeupMainLoop.read();
        }

        double now = ML::wall_time();

        if (now - lastPings > 1.0) {
            double atStart = getTime();

            // Send out pings and interpret the results of the last lot of
            // pinging.
            sendPings();
            lastPings = now;

            recordTime("sendPings", atStart);
        }

        double beforeChecks = getTime();

        if (now - last_check_pace > 10.0) {
            recordEvent("numTimesCouldSleep", ET_LEVEL, numTimesCouldSleep);

            totalSleeps += numTimesCouldSleep;

            numTimesCouldSleep = 0;
            last_check_pace = now;
        }

        if (now - last_check > 10.0) {
            logUsageMetrics(10.0);

            logMessage("MARK",
                       Date::fromSecondsSinceEpoch(last_check).print(),
                       format("active: %zd augmenting, %zd inFlight, "
                              "%zd agents",
                              augmentationLoop.numAugmenting(),
                              inFlight.size(),
                              agents.size()));

            dutyCycleCurrent.ending = Date::now();
            dutyCycleHistory.push_back(dutyCycleCurrent);
            dutyCycleCurrent.clear();

            if (dutyCycleHistory.size() > 200)
                dutyCycleHistory.erase(dutyCycleHistory.begin(),
                                       dutyCycleHistory.end() - 100);

            checkDeadAgents();

            double total = 0.0;
            for (auto it = times.begin(); it != times.end();  ++it)
                total += it->second.time;

            cerr << "total of " << total << " microseconds and "
                 << totalSleeps << " sleeps" << endl;

            for (auto it = times.begin(); it != times.end();  ++it) {
                cerr << ML::format("%-30s %8lld %10.0f %6.2f%% %8.2fus/call\n",
                                   it->first.c_str(),
                                   (unsigned long long)it->second.count,
                                   it->second.time,
                                   100.0 * it->second.time / total,
                                   it->second.time / it->second.count);

                recordEvent(("routerLoop." + it->first).c_str(), ET_LEVEL,
                        1.0 * it->second.time / (now - last_check) / 1000000.0);

            }

            times.clear();
            totalSleeps = 0;

            last_check = now;
        }

        recordTime("checks", beforeChecks);

        if (now - lastTimestamp >= 1.0) {
            double atStart = getTime();

            banker->logBidEvents(*this);
            //issueTimestamp();
            lastTimestamp = now;

            recordTime("logBidEvents", atStart);
        }
    }

    //cerr << "finished run loop" << endl;

    recordHit("routerDown");

    //cerr << "server shutdown" << endl;
}

void
Router::
shutdown()
{
    loopMonitor.shutdown();

    configListener.shutdown();

    shutdown_ = true;
    futex_wake(shutdown_);
    wakeupMainLoop.signal();

    augmentationLoop.shutdown();

    if (runThread)
        runThread->join();
    runThread.reset();
    if (cleanupThread)
        cleanupThread->join();
    cleanupThread.reset();

    logger.shutdown();
    banker.reset();

    monitorClient.shutdown();
    monitorProviderClient.shutdown();
}

void
Router::
injectAuction(std::shared_ptr<Auction> auction, double lossTime)
{
    // cerr << "injectAuction was called!!!" << endl;
    if (!auction->handleAuction) {
        // Modify the auction to insert our auction done handling
        auction->handleAuction
            = [=] (std::shared_ptr<Auction> auction)
            {
                this->onAuctionDone(auction);
            };
    }

    auction->lossAssumed = getCurrentTime().plusSeconds(lossTime);
    onNewAuction(auction);
}

inline std::string chomp(const std::string & s)
{
    const char * start = s.c_str();
    const char * end = start + s.length();

    while (end > start && end[-1] == '\n') --end;

    if (end == start + s.length()) return s;
    return string(start, end);
}

std::shared_ptr<Auction>
Router::
injectAuction(Auction::HandleAuction onAuctionFinished,
              std::shared_ptr<BidRequest> request,
              const std::string & requestStr,
              const std::string & requestStrFormat,
              double startTime,
              double expiryTime,
              double lossTime)
{
    auto auction = std::make_shared<Auction>(
        nullptr,
        onAuctionFinished,
        request,
        chomp(requestStr),
        requestStrFormat,
        Date::fromSecondsSinceEpoch(startTime),
        Date::fromSecondsSinceEpoch(expiryTime));

    injectAuction(auction, lossTime);

    return auction;
}

void
Router::
notifyFinishedAuction(const Id & auctionId)
{
    throw ML::Exception("notifyFinishedAuction: not finished");
}

int
Router::
numAuctionsInProgress() const
{
    return -1;//inFlight.size();
}

void
Router::
handleAgentMessage(const std::vector<std::string> & message)
{
    try {
        using namespace std;
        //cerr << "got agent message " << message << endl;

        if (message.size() < 2) {
            returnErrorResponse(message, "not enough message parts");
            return;
        }

        const string & address = message[0];
        const string & request = message[1];

        if (request.empty()) {
            returnErrorResponse(message, "null request field");
            return;
        }

        if (request == "CONFIG") {
            string configName = message.at(2);
            if (!agents.count(configName)) {
                // We don't yet know about its configuration
                bidder->sendMessage(nullptr, address, "NEEDCONFIG");
                return;
            }
            agents[configName].address = address;
            return;
        }

        if (!agents.count(address)) {
            cerr << "doing NEEDCONFIG for " << address << endl;
            return;
        }

        AgentInfo & info = agents[address];
        info.gotHeartbeat(Date::now());

        if (!info.configured) {
            throw ML::Exception("message to unconfigured agent");
        }

        if (request[0] == 'B' && request == "BID") {
            doBid(message);
            return;
        }

        //cerr << "router got message " << message << endl;

        if (request[0] == 'P' && request == "PONG0") {
            doPong(0, message);
            return;
        }
        else if (request[0] == 'P' && request == "PONG1") {
            doPong(1, message);
            return;
        }

        returnErrorResponse(message, "unknown agent request");
    } catch (const std::exception & exc) {
        returnErrorResponse(message,
                            "threw exception: " + string(exc.what()));
    }
}

void
Router::
logUsageMetrics(double period)
{
    std::string p = std::to_string(period);

    for (auto it = lastAgentUsageMetrics.begin();
         it != lastAgentUsageMetrics.end();) {
        if (agents.count(it->first) == 0) {
            it = lastAgentUsageMetrics.erase(it);
        }
        else {
            it++;
        }
    }

    for (const auto & item : agents) {
        auto & info = item.second;
        auto & last = lastAgentUsageMetrics[item.first];

        AgentUsageMetrics newMetrics(info.stats->intoFilters,
                                     info.stats->passedStaticFilters,
                                     info.stats->passedDynamicFilters,
                                     info.stats->auctions,
                                     info.stats->bids);
        AgentUsageMetrics delta = newMetrics - last;

        logMessage("USAGE", "AGENT", p, item.first,
                   info.config->account.toString(),
                   delta.intoFilters,
                   delta.passedStaticFilters,
                   delta.passedDynamicFilters,
                   delta.auctions,
                   delta.bids,
                   info.config->bidProbability);
        logMessageToAnalytics("USAGE", "AGENT", p, item.first,
                   info.config->account.toString(),
                   delta.intoFilters,
                   delta.passedStaticFilters,
                   delta.passedDynamicFilters,
                   delta.auctions,
                   delta.bids,
                   info.config->bidProbability);


        last = move(newMetrics);
    }

    {
        RouterUsageMetrics newMetrics;
        int numExchanges = 0;
        float acceptAuctionProbability(0.0);

        forAllExchanges([&](std::shared_ptr<ExchangeConnector> const & item) {
            ++numExchanges;
            newMetrics.numRequests += item->numRequests;
            newMetrics.numAuctions += item->numAuctions;
            acceptAuctionProbability += item->acceptAuctionProbability;
        });
        newMetrics.numBids = numBids;
        newMetrics.numNoPotentialBidders = numNoPotentialBidders;
        newMetrics.numAuctionsWithBid = numAuctionsWithBid;

        RouterUsageMetrics delta = newMetrics - lastRouterUsageMetrics;

        logMessage("USAGE", "ROUTER", p,
                   delta.numRequests,
                   delta.numAuctions,
                   delta.numNoPotentialBidders,
                   delta.numBids,
                   delta.numAuctionsWithBid,
                   acceptAuctionProbability / numExchanges);
        logMessageToAnalytics("USAGE", "ROUTER", p,
                   delta.numRequests,
                   delta.numAuctions,
                   delta.numNoPotentialBidders,
                   delta.numBids,
                   delta.numAuctionsWithBid,
                   acceptAuctionProbability / numExchanges);


        lastRouterUsageMetrics = move(newMetrics);
    }
}

void
Router::
checkDeadAgents()
{
    //Date start = Date::now();

    using namespace std;
    //cerr << "checking for dead agents" << endl;

    std::vector<Agents::iterator> deadAgents;

    for (auto it = agents.begin(), end = agents.end();  it != end;
         ++it) {
        auto & info = it->second;

        const std::string & account = info.config->account.toString('.');

        Date now = Date::now();
        double oldest = 0.0;
        double total = 0.0;

        vector<Id> toExpire;

        // Check for in flight timeouts.  This shouldn't happen, but there
        // appears to be a way in which we lose track of an inflight auction
        auto onInFlight = [&] (const Id & id, const Date & date)
            {
                double secondsSince = now.secondsSince(date);

                oldest = std::max(oldest, secondsSince);
                total += secondsSince;

                if (secondsSince > 30.0) {

                    this->recordHit("accounts.%s.lostBids", account);

                    bidder->sendBidLostMessage(info.config, it->first, inFlight[id].auction);

                    toExpire.push_back(id);
                }
            };

        info.forEachInFlight(onInFlight);

        this->recordLevel(info.numBidsInFlight(),
                          "accounts.%s.inFlight.numInFlight", account);
        this->recordLevel(oldest,
                          "accounts.%s.inFlight.oldestAgeSeconds", account);
        double averageAge = 0.0;
        if (info.numBidsInFlight() != 0)
            averageAge = total / info.numBidsInFlight();

        this->recordLevel(averageAge,
                          "accounts.%s.inFlight.averageAgeSeconds", account);

        for (auto jt = toExpire.begin(), jend = toExpire.end();  jt != jend;
             ++jt) {
            info.expireBidInFlight(*jt);
        }

        double timeSinceHeartbeat
            = now.secondsSince(info.status->lastHeartbeat);

        this->recordLevel(timeSinceHeartbeat,
                          "accounts.%s.timeSinceHeartbeat", account);

        if (timeSinceHeartbeat > 5.0) {
            info.status->dead = true;
            if (it->second.numBidsInFlight() != 0) {
                cerr << "agent " << it->first
                     << " has " << it->second.numBidsInFlight()
                     << " undead auctions: " << endl;

                auto onInFlight = [&] (const Id & id, Date date)
                    {
                        cerr << "  " << id << " --> "
                        << date << " (" << now.secondsSince(date)
                        << "s ago)" << endl;
                    };

                info.forEachInFlight(onInFlight);
            }
            else {
                // agent is dead
                cerr << "agent " << it->first << " appears to be dead"
                     << endl;
                bidder->sendMessage(info.config, it->first, "BYEBYE");
                deadAgents.push_back(it);
            }
        }
    }

    for (auto it = deadAgents.begin(), end = deadAgents.end();
         it != end;  ++it) {
        cerr << "WARNING: dead agent doesn't clean up its state properly"
             << endl;
        // TODO: undo all bids in progress
        filters.removeConfig((*it)->first);
        agents.erase(*it);
    }

    if (!deadAgents.empty())
        // Broadcast that we have different agents
        updateAllAgents();

    //cerr << "dead agents took " << Date::now().secondsSince(start) << "s"
    //     << endl;
}

void
Router::
checkExpiredAuctions()
{
    //recentlySubmitted.clear();

    Date start = Date::now();

    {
        RouterProfiler profiler(dutyCycleCurrent.nsExpireInFlight);

        // Look for in flight timeout expiries
        auto onExpiredInFlight = [&] (const Id & auctionId,
                                      const AuctionInfo & auctionInfo)
            {
                this->debugAuction(auctionId, "EXPIRED", {});

                // Tell any remaining bidders that it's too late...
                for (auto it = auctionInfo.bidders.begin(),
                         end = auctionInfo.bidders.end();
                     it != end;  ++it) {
                    string agent = it->first;
                    if (!agents.count(agent)) continue;

                    if (agents[agent].expireBidInFlight(auctionId)) {
                        AgentInfo & info = this->agents[agent];
                        ++info.stats->tooLate;

                        this->recordHit("accounts.%s.EXPIRED",
                                        info.config->account.toString('.'));

                        bidder->sendBidDroppedMessage(info.config, agent, auctionInfo.auction);
                    }
                }

#if 0
                string msg = ML::format("in flight auction expiry: id %s "
                                        "status %s, %zd bidders:",
                                        auctionId.toString().c_str(),
                                        auctionInfo.auction->status().c_str(),
                                        auctionInfo.bidders.size());
                for (auto it = auctionInfo.bidders.begin(),
                         end = auctionInfo.bidders.end();
                     it != end;  ++it)
                    msg += ' ' + it->first + "->" + it->second.bidTime.print(5);
                cerr << Date::now().print(5) << " " << msg << endl;
                dumpAuction(auctionId);
                this->logRouterError("checkExpiredAuctions.inFlight",
                                     msg);

#endif

                // end the auction when it expires in case we're waiting on dead agents
        if(!auctionInfo.auction->getResponses().empty()) {
                    if(!auctionInfo.auction->finish()) {
                this->recordHit("tooLateToFinish");
            }
        }

                return Date();
            };

        inFlight.expire(onExpiredInFlight, start);
    }

    {
        RouterProfiler profiler(dutyCycleCurrent.nsExpireBlacklist);
        blacklist.doExpiries();
    }

    if (doDebug) {
        RouterProfiler profiler(dutyCycleCurrent.nsExpireDebug);
        expireDebugInfo();
    }
}

void
Router::
returnErrorResponse(const std::vector<std::string> & message,
                    const std::string & error)
{
    using namespace std;
    if (message.empty()) return;
    logMessage("ERROR", error, message);
    logMessageToAnalytics("ERROR", error, message);
    const auto& agent = message[0];
    AgentInfo & info = this->agents[agent];
    bidder->sendErrorMessage(info.config, agent, error, message);
}

void
Router::
returnInvalidBid(
        const std::string &agent, const std::string &bidData,
        const std::shared_ptr<Auction> &auction,
        const char *reason, const char *message, ...) {

    auto& agentInfo = agents[agent];
    const auto& agentConfig = agentInfo.config;
    this->recordHit("bidErrors.%s", reason);
    this->recordHit("accounts.%s.bidErrors.total",
                    agentConfig->account.toString('.'));
    this->recordHit("accounts.%s.bidErrors.%s",
                    agentConfig->account.toString('.'),
                    reason);

    ++agentInfo.stats->invalid;

    va_list ap;
    va_start(ap, message);
    string formatted;
    try {
        formatted = vformat(message, ap);
    } catch (...) {
        va_end(ap);
        throw;
    }
    va_end(ap);

    cerr << "invalid bid for agent " << agent << ": "
         << formatted << endl;
    cerr << bidData << endl;

    logMessageToAnalytics("INVALID", agentConfig, agent, formatted, auction);
    bidder->sendBidInvalidMessage(agentConfig, agent, formatted, auction);
}

void
Router::
returnInvalidBid(
        const std::string &agent, const std::string &bidData,
        const std::shared_ptr<Auction> &auction,
        const std::string &reason, const char *message, ...) {

    auto& agentInfo = agents[agent];
    const auto& agentConfig = agentInfo.config;
    this->recordHit("bidErrors.%s", reason);
    this->recordHit("accounts.%s.bidErrors.total",
                    agentConfig->account.toString('.'));
    this->recordHit("accounts.%s.bidErrors.%s",
                    agentConfig->account.toString('.'),
                    reason);

    ++agentInfo.stats->invalid;

    va_list ap;
    va_start(ap, message);
    string formatted;
    try {
        formatted = vformat(message, ap);
    } catch (...) {
        va_end(ap);
        throw;
    }
    va_end(ap);

    cerr << "invalid bid for agent " << agent << ": "
         << formatted << endl;
    cerr << bidData << endl;

    logMessageToAnalytics("INVALID", agentConfig, agent, formatted, auction);
    bidder->sendBidInvalidMessage(agentConfig, agent, formatted, auction);
}

void
Router::
doStats(const std::vector<std::string> & message)
{
    Json::Value result(Json::objectValue);

    result["numAugmenting"] = augmentationLoop.numAugmenting();
    result["numInFlight"] = inFlight.size();
    result["blacklistUsers"] = blacklist.size();

    result["numAgents"] = agents.size();

    //result["accounts"] = banker->dumpAllCampaignsJson();

    Json::Value agentsVal(Json::objectValue);

    int totalAgentInFlight = 0;

    BOOST_FOREACH(auto agent, agents) {
        agentsVal[agent.first] = agent.second.toJson(false, false);
        totalAgentInFlight += agent.second.numBidsInFlight();
    }

    result["agents"] = agentsVal;

    result["totalAgentInFlight"] = totalAgentInFlight;

    if (dutyCycleHistory.empty())
        result["dutyCycle"] = dutyCycleCurrent.toJson();
    else result["dutyCycle"] = dutyCycleHistory.back().toJson();

    result["fileDescriptorCount"] = ML::num_open_files();

    addChildServiceStatus(result);

    result["numAuctions"] = numAuctions;
    result["numBids"] = numBids;
    result["numNonEmptyBids"] = numNonEmptyBids;
    result["numAuctionsWithBid"] = numAuctionsWithBid;
    result["numNoBidders"] = numNoBidders;
    result["numNoPotentialBidders"] = numNoPotentialBidders;

    //sendMessage(controlEndpoint, message[0], result);
}


Json::Value
Router::
getServiceStatus() const
{
    return getStats();
}

void
Router::
augmentAuction(const std::shared_ptr<AugmentationInfo> & info)
{
    if (!info || !info->auction)
        throw ML::Exception("augmentAuction with no auction to augment");

    if (info->auction->tooLate()) {
        recordHit("tooLateBeforeAdd");
        return;
    }

    auto onDoneAugmenting = [=] (const std::shared_ptr<AugmentationInfo> & info)
        {
            info->auction->doneAugmenting = Date::now();

            if (info->auction->tooLate()) {
                this->recordHit("tooLateAfterAugmenting");
                return;
            }

            // Send it off to be farmed out to the bidders
            startBiddingBuffer.push(info);
            wakeupMainLoop.signal();
        };

    augmentationLoop.augment(info, Date::now().plusSeconds(augmentationWindow.count()),
                             onDoneAugmenting);
}

std::shared_ptr<AugmentationInfo>
Router::
preprocessAuction(const std::shared_ptr<Auction> & auction)
{
    ML::atomic_inc(numAuctions);

    Date now = Date::now();
    auction->inPrepro = now;

    if (auction->lossAssumed == Date())
        auction->lossAssumed
            = Date::now().plusSeconds(secondsUntilLossAssumed_);
    Date lossTimeout = auction->lossAssumed;

    //cerr << "AUCTION " << auction->id << " " << auction->requestStr << endl;

    //cerr << "url = " << auction->request->url << endl;

    if (auction->tooLate()) {
        recordHit("tooLateBeforeRouting");
        //inFlight.erase(auctionId);
        return std::shared_ptr<AugmentationInfo>();
    }

    const string & exchange = auction->request->exchange;

    /* Parse out the adimp. */
    const vector<AdSpot> & imp = auction->request->imp;

    recordCount(imp.size(), "exchange.%s.imp", exchange.c_str());
    recordHit("exchange.%s.requests", exchange.c_str());

    // List of possible agents per round robin group
    std::map<string, GroupPotentialBidders> groupAgents;

    double timeLeftMs = auction->timeAvailable() * 1000.0;

    bool traceAuction = auction->id.hash() % 10 == 0;

    auto exchangeConnector = auction->exchangeConnector;


    auto doFilterStat = [&] (const AgentConfig& config, const char * reason) {
        if (!traceAuction) return;

        this->recordHit("accounts.%s.filter.%s",
                config.account.toString('.'),
                reason);
    };

    if (traceAuction) {
        forEachAgent([&] (const AgentInfoEntry& info) {
                    ML::atomic_inc(info.stats->intoFilters);
                    doFilterStat(*info.config, "intoStaticFilters");
                });
    }

    // Do the actual filtering.
    auto biddableConfigs = filters.filter(*auction->request, exchangeConnector);

    auto checkAgent = [&] (
            const AgentConfig & config,
            const AgentStatus & status,
            AgentStats & stats)
        {
            if (status.dead || status.lastHeartbeat.secondsSince(now) > 2.0) {
                doFilterStat(config, "static.agentAppearsDead");
                return false;
            }

            if (status.numBidsInFlight >= config.maxInFlight) {
                doFilterStat(config, "static.earlyTooManyInFlight");
                return false;
            }

            /* Check if we have enough time to process it. */
            if (config.minTimeAvailableMs != 0.0
                && timeLeftMs < config.minTimeAvailableMs)
            {
                ML::atomic_inc(stats.notEnoughTime);
                doFilterStat(config, "static.notEnoughTime");
                return false;
            }

            return true;
        };

    for (const auto& entry : biddableConfigs) {
        if (entry.biddableSpots.empty()) continue;
        if (!checkAgent(*entry.config, *entry.status, *entry.stats)) continue;

        ML::atomic_inc(entry.stats->passedStaticFilters);
        doFilterStat(*entry.config, "passedStaticFilters");

        string rrGroup = entry.config->roundRobinGroup;
        if (rrGroup == "") rrGroup = entry.name;

        PotentialBidder bidder;
        bidder.agent = entry.name;
        bidder.config = entry.config;
        bidder.stats = entry.stats;
        bidder.imp = std::move(entry.biddableSpots);

        groupAgents[rrGroup].push_back(bidder);
        groupAgents[rrGroup].totalBidProbability += entry.config->bidProbability;
    }


    std::vector<GroupPotentialBidders> validGroups;

    for(auto it = groupAgents.begin(), end = groupAgents.end(); it != end; ++it) {
        // Check for bid probability and skip if we don't bid
        if(enableBidProbability) {
            double bidProbability
                = it->second.totalBidProbability
                / it->second.size()
                * globalBidProbability;

            if (bidProbability < 1.0) {
                float val = (random() % 1000000) / 1000000.0;
                if (val > bidProbability) {
                    for (unsigned i = 0;  i < it->second.size();  ++i)
                        ML::atomic_inc(it->second[i].stats->skippedBidProbability);
                    continue;
                }
            }
        }

        // Group is valid for bidding; next step is to augment the bid
        // request
        validGroups.push_back(it->second);
    }

    this->recordLevel(validGroups.size(), "potentialBiddersPerRequest");

    if (validGroups.empty()) {
        // Now we need to end the auction
        //inFlight.erase(auctionId);
        if (!auction->finish()) {
            recordHit("tooLateToFinish");
        }

        //cerr << "no valid groups " << endl;
        return std::shared_ptr<AugmentationInfo>();
    }

    auto info = std::make_shared<AugmentationInfo>(auction, lossTimeout);
    info->potentialGroups.swap(validGroups);

    auction->outOfPrepro = Date::now();

    recordOutcome(auction->outOfPrepro.secondsSince(auction->inPrepro) * 1000.0,
                  "preprocessAuctionTimeMs");

    return info;
}

void
Router::
doStartBidding(const std::vector<std::string> & message)
{
    std::shared_ptr<AugmentationInfo> augInfo
        = sharedPtrFromMessage<AugmentationInfo>(message.at(2));
    doStartBidding(augInfo);
}

void
Router::
doStartBidding(const std::shared_ptr<AugmentationInfo> & augInfo)
{
    //static const char *fName = "Router::doStartBidding:";
    RouterProfiler profiler(dutyCycleCurrent.nsStartBidding);

    try {
        Id auctionId = augInfo->auction->id;
        if (inFlight.count(auctionId)) {
            throwException("doStartBidding.alreadyInFlight",
                           "auction with ID %s already in progress",
                           auctionId.toString().c_str());
        }
#if 0
        if (findAuction(finished, auctionId)) {
            throwException("doStartBidding.alreadyFinished",
                           "auction with ID %s already finished",
                           auctionId.toString().c_str());
        }
#endif

        //cerr << "doStartBidding " << auctionId << endl;

        auto groupAgents = augInfo->potentialGroups;

        AuctionInfo & auctionInfo = addAuction(augInfo->auction,
                                               augInfo->lossTimeout);
        auto auction = augInfo->auction;

        Date now = Date::now();

        auction->inStartBidding = now;

        double timeLeftMs = auction->timeAvailable(now) * 1000.0;
        double timeUsedMs = auction->timeUsed(now) * 1000.0;

        bool traceAuction = auction->id.hash() % 10 == 0;

        const auto& augList = augInfo->auction->augmentations;

        /* For each round-robin group, send the request off to exactly one
           element. */
        for (auto it = groupAgents.begin(), end = groupAgents.end();
             it != end;  ++it) {

            GroupPotentialBidders & bidders = *it;

            for (unsigned i = 0;  i < bidders.size();  ++i) {
                PotentialBidder & bidder = bidders[i];
                if (!agents.count(bidder.agent)) continue;
                AgentInfo & info = agents[bidder.agent];
                const AgentConfig & config = *bidder.config;

                auto doFilterStat = [&] (const char * reason)
                    {
                        if (!traceAuction) return;

                        this->recordHit("accounts.%s.filter.%s",
                                        config.account.toString('.'),
                                        reason);
                    };

                auto doFilterMetric = [&] (const char * reason, float val)
                    {
                        if (!traceAuction) return;

                        this->recordOutcome(val, "accounts.%s.filter.%s",
                                            config.account.toString('.'),
                                            reason);
                    };


                doFilterStat("intoDynamicFilters");

                /* Check if we have too many in flight. */
                if (info.numBidsInFlight() >= info.config->maxInFlight) {
                    ++info.stats->tooManyInFlight;
                    bidder.inFlightProp = PotentialBidder::NULL_PROP;
                    doFilterStat("dynamic.tooManyInFlight");
                    continue;
                }

                /* Check if we have enough time to process it. */
                if (config.minTimeAvailableMs != 0.0
                    && timeLeftMs < config.minTimeAvailableMs) {

                    static ML::Spinlock lock;

                    if (auction->id.hash() % 1000 == 999 &&
                        lock.try_lock()) {

                        Date now = Date::now();
                        Date last = auction->start;
                        auto printTime
                            = [&] (const char * what, const Date & date)
                            {
                                cerr << ML::format("%-30s %s %10.3f %10.3f\n",
                                                   what,
                                                   date.print(6).c_str(),
                                                   auction->start.secondsSince(date)
                                                   * 1000.0,
                                                   last.secondsSince(date)
                                                   * 1000.0);
                                last = date;
                            };

                        cerr << "no time available in dynamic" << endl;
                        printTime("start", auction->start);
                        printTime("doneParsing", auction->doneParsing);
                        printTime("inPrepro", auction->inPrepro);
                        printTime("outOfPrepro", auction->outOfPrepro);
                        printTime("doneAugmenting", auction->doneAugmenting);
                        printTime("inStartBidding", auction->inStartBidding);
                        printTime("expiry", auction->expiry);
                        printTime("now", now);

                        lock.unlock();
                    }

                    ML::atomic_inc(info.stats->notEnoughTime);
                    bidder.inFlightProp = PotentialBidder::NULL_PROP;
                    doFilterStat("dynamic.notEnoughTime");
                    doFilterMetric("metric.timeUsedBeforeDynamicFilter",
                                   timeUsedMs);
                    doFilterMetric("metric.timeLeftBeforeDynamicFilter",
                                   timeLeftMs);
                    doFilterMetric("metric.timeElapsedBeforePreproMs",
                                   auction->start.secondsUntil(auction->inPrepro) * 1000.0);
                    doFilterMetric("metric.timeElapsedDuringPreproMs",
                                   auction->inPrepro.secondsUntil(auction->outOfPrepro) * 1000.0);
                    doFilterMetric("metric.timeWindowMs",
                                   auction->expiry.secondsSince(auction->start) * 1000.0 - info.config->minTimeAvailableMs);
                    continue;
                }

                stringstream ss;
                ss << endl;

                /* Filter on the augmentation tags */
                bool filteredByAugmentation = false;
                for (const auto& augConfig : config.augmentations) {
                    auto it = augList.find(augConfig.name);

                    if (it == augList.end()) {
                        if (!augConfig.required) continue;
                        string stat = "dynamic." + augConfig.name + ".missing";
                        doFilterStat(stat.c_str());
                        filteredByAugmentation = true;
                        break;
                    }

                    vector<string> tags = it->second.tagsForAccount(config.account);
                    if (augConfig.filters.anyIsIncluded(tags)) continue;

                    ML::atomic_inc(info.stats->augmentationTagsExcluded);
                    string stat = "dynamic." + augConfig.name + ".tags";
                    doFilterStat(stat.c_str());
                    filteredByAugmentation = true;
                    break;
                }
                if (filteredByAugmentation) continue;


                /* Check that there is no blacklist hit on the user. */
                if (config.hasBlacklist()
                    && blacklist.matches(*auction->request, bidder.agent,
                                         config)) {
                    ML::atomic_inc(info.stats->userBlacklisted);
                    doFilterStat("dynamic.userBlacklisted");
                    continue;
                }

                bidder.inFlightProp
                    = info.numBidsInFlight() / max(info.config->maxInFlight, 1);

                ML::atomic_inc(info.stats->passedDynamicFilters);
                doFilterStat("passedDynamicFilters");
            }

            // Sort the roundrobin infos to find the best one
            std::sort(bidders.begin(), bidders.end());

            int numBest = 1;
            float bestInFlightProp = bidders[0].inFlightProp;

            if (bestInFlightProp == PotentialBidder::NULL_PROP) {
                // Excluded because too many in flight
                //cerr << "TOO MANY IN FLIGHT" << endl;
                continue;
            }

            for (;  numBest < bidders.size();  ++numBest) {
                float inFlightProp = bidders[numBest].inFlightProp;
                if (inFlightProp <= bestInFlightProp) continue;
                break;
            }

            // Take a random one from all which are equally good
            int best = random() % numBest;

            // Best one is the first one
            PotentialBidder & winner = bidders[best];
            string agent = winner.agent;

            if (!agents.count(agent)) {
                //cerr << "!!!AGENT IS GONE" << endl;
                continue;  // agent is gone
            }
            AgentInfo & info = agents[agent];

            ++info.stats->auctions;

            Json::Value aggregatedAug;
            for (const auto& aug : augList) {
                aggregatedAug[aug.first] =
                    aug.second.filterForAccount(winner.config->account).toJson();
            }
            auction->agentAugmentations[agent] = chomp(aggregatedAug.toString());

            //auctionInfo.activities.push_back("sent to " + agent);

            BidInfo bidInfo;
            bidInfo.agentConfig = winner.config;
            bidInfo.bidTime = Date::now();
            bidInfo.imp = winner.imp;

            auctionInfo.bidders.insert(make_pair(agent, std::move(bidInfo)));  // create empty bid response
            if (!info.trackBidInFlight(auctionId, bidInfo.bidTime))
                throwException("doStartBidding.agentAlreadyBidding",
                               "agent %s is already processing auction %s",
                               agent.c_str(),
                               auctionId.toString().c_str());
        }

        //cerr << " auction " << id << " with "
        //     << auctionInfo.bidders.size() << " bidders" << endl;

        //auctionInfo.activities.push_back(ML::format("total of %zd agents",
        //                                 auctionInfo.bidders.size()));
        if (auction->tooLate()) {
            recordHit("tooLateAfterRouting");
            // Unwind everything?
        }

        this->recordLevel(auctionInfo.bidders.size(), "bidRequestsSentToBiddersPerRequest");

        if (!auctionInfo.bidders.empty()) {
            bidder->sendAuctionMessage(
                    auctionInfo.auction, timeLeftMs, auctionInfo.bidders);
        }
        else {
            /* No bidders; don't bother with the bid */
            ML::atomic_inc(numNoBidders);
            inFlight.erase(auctionId);
            //cerr << fName << "About to call finish " << endl;
            if (!auction->finish()) {
                recordHit("tooLateToFinish");
                //cerr << "couldn't finish auction 1 " << auction->id << endl;
            }
        }

        debugAuction(auctionId, "AUCTION");
    } catch (const std::exception & exc) {
        cerr << "warning: auction threw exception: " << exc.what() << endl;
        if (augInfo)
            augInfo->auction->setError("auction processing error", exc.what());
    }
}

AuctionInfo &
Router::
addAuction(std::shared_ptr<Auction> auction, Date lossTimeout)
{
    const Id & id = auction->id;

    double bidMemoryWindow = 5.0;  // how many seconds we remember auctions

    try {
        AuctionInfo & result
            = inFlight.insert(id, AuctionInfo(auction, lossTimeout),
                              getCurrentTime().plusSeconds(bidMemoryWindow));
        return result;
    } catch (const std::exception & exc) {
        //cerr << "====================================" << endl;
        //cerr << exc.what() << endl;
        throwException("addAuction.alreadyInProgress",
                       "auction with ID %s already in progress: %s",
                       id.toString().c_str(), exc.what());
    }
}


static bool failBid(double proportion)
{
    if (proportion < 0.01)
        return false;

    return (random() % 100) < floor(proportion * 100.0);
}

void
Router::
doBid(const std::vector<std::string> & message)
{
    if (message.size() < 5 || message.size() > 6) {
        returnErrorResponse(message, "BID message has 4-5 parts");
        return;
    }

    Id auctionId(message[2]);

    const string & agent = message[0];
    const string & biddata = message[3];
    const string & model = message[4];

    WinCostModel wcm = WinCostModel::fromJson(model.empty() ? Json::Value() : Json::parse(model));

    static const string nullStr("null");
    const string & meta = (message.size() >= 6 ? message[5] : nullStr);

    BidMessage bidMessage;
    bidMessage.agents.push_back(agent);
    bidMessage.auctionId = auctionId;
    bidMessage.wcm = std::move(wcm);
    bidMessage.meta = std::move(meta);

    Bids bids;
    try {
        bids = Bids::fromJson(biddata);
    }
    catch (const std::exception & exc) {
        auto it = inFlight.find(auctionId);
        if (it == inFlight.end()) {
            recordHit("bidError.unknownAuction");
            returnErrorResponse(message, "unknown auction");
            return;
        }
        else {
            returnInvalidBid(agent, biddata, it->second.auction,
                    "bidParseError",
                    "couldn't parse bid JSON %s: %s", biddata.c_str(), exc.what());
        }
        return;
    }
    bidMessage.bids = std::move(bids);

    doBidImpl(bidMessage, message);
}

void
Router::
doBidImpl(const BidMessage &message, const std::vector<std::string> &originalMessage)
{
    Date dateGotBid = Date::now();

    if (failBid(bidsErrorRate)) {
        returnErrorResponse(originalMessage, "Intentional error response (--bids-error-rate)");
        return;
    }

    ExcAssert(!message.agents.empty());

    const auto& auctionId = message.auctionId;
    auto it = inFlight.find(auctionId);
    if (it == inFlight.end()) {
        recordHit("bidError.unknownAuction");
        returnErrorResponse(originalMessage, "unknown auction");
        return;
    }

    AuctionInfo & auctionInfo = it->second;

    for (const auto &agent: message.agents) {
        if (!agents.count(agent)) {
            returnErrorResponse(originalMessage, "unknown agent");
            return;
        }

        auto biddersIt = auctionInfo.bidders.find(agent);
        if (biddersIt == auctionInfo.bidders.end()) {
            recordHit("bidError.agentSkippedAuction");
            returnErrorResponse(originalMessage,
                                "agent shouldn't bid on this auction");
            return;
        }

        AgentInfo & info = agents[agent];
        /* One less in flight. */
        if (!info.expireBidInFlight(auctionId)) {
            recordHit("bidError.agentNotBidding");
            returnErrorResponse(originalMessage, "agent wasn't bidding on this auction");
            return;
        }
        auto & config = *biddersIt->second.agentConfig;
        recordHit("accounts.%s.bids", config.account.toString('.'));
    }


    const std::vector<AdSpot> & imp = auctionInfo.auction->request->imp;

    int numValidBids = 0;

    recordHit("bid");

    const auto& agent = message.agents[0];
    auto biddersIt = auctionInfo.bidders.find(agent);
    auto & config = *biddersIt->second.agentConfig;
    AgentInfo & info = agents[agent];
    const auto& agentConfig = info.config;

    const auto& bids = message.bids;
    auto bidsString = bids.toJson().toStringNoNewLine();

    BidInfo bidInfo(std::move(biddersIt->second));

    RouterProfiler profiler(dutyCycleCurrent.nsBid);

    ML::atomic_inc(numBids);

    int numPassedBids = 0;

    ExcCheckEqual(bids.size(), bidInfo.imp.size(),
            "invalid shape for bids array");

    auctionInfo.auction->addDataSources(bids.dataSources);

    this->recordLevel(bids.size(), "bidsPerBidRequest");

    for (int i = 0; i < bids.size(); ++i) {

        Bid bid = bids[i];

        if (bid.isNullBid()) {
            ++numPassedBids;
            continue;
        }

        int spotIndex = bidInfo.imp[i].first;

        if (bid.creativeIndex == -1) {
            returnInvalidBid(agent, bidsString, auctionInfo.auction,
                    "nullCreativeField",
                    "creative field is null in response %s",
                    bidsString.c_str());
            continue;
        }

        if (bid.creativeIndex < 0
                || bid.creativeIndex >= config.creatives.size())
        {
            returnInvalidBid(agent, bidsString, auctionInfo.auction,
                    "outOfRangeCreative",
                    "parsing field 'creative' of %s: creative "
                    "number %d out of range 0-%zd",
                    bidsString.c_str(), bid.creativeIndex,
                    config.creatives.size());
            continue;
        }

        if (bid.price.isNegative() || bid.price > maxBidAmount) {
            if (slowModePeriodicSpentReached) {
                bid.price = maxBidAmount;
            } else {
                returnInvalidBid(agent, bidsString, auctionInfo.auction,
                    "invalidPrice",
                    "bid price of %s is outside range of $0-%s parsing bid %s",
                    bid.price.toString().c_str(),
                    maxBidAmount.toString().c_str(),
                    bidsString.c_str());
                continue;
            }
        }

     auto getbid = auctionInfo.auction->exchangeConnector->getBidValidity(bid, imp, spotIndex);

        if (!getbid.isValidbid) {
            returnInvalidBid(agent, bidsString, auctionInfo.auction,
                getbid.reason_,
                "no bid");
            continue;
        }
        const Creative & creative = config.creatives.at(bid.creativeIndex);

        if (!creative.compatible(imp[spotIndex])) {
#if 1
            cerr << "creative not compatible with spot: " << endl;
            cerr << "auction: " << auctionInfo.auction->requestStr
                << endl;
            cerr << "config: " << config.toJson().toStringNoNewLine() << endl;
            cerr << "bid: " << bidsString << endl;
            cerr << "spot: " << imp[i].toJson().toStringNoNewLine() << endl;
            cerr << "spot num: " << spotIndex << endl;
            cerr << "bid num: " << i << endl;
            cerr << "creative num: " << bid.creativeIndex << endl;
            cerr << "creative: " << creative.toJson().toStringNoNewLine() << endl;
#endif
            returnInvalidBid(agent, bidsString, auctionInfo.auction,
                    "creativeNotCompatibleWithSpot",
                    "creative %s not compatible with spot %s",
                    creative.toJson().toString().c_str(),
                    imp[spotIndex].toJson().toString().c_str());
            continue;
        }

        if (!creative.biddable(auctionInfo.auction->request->exchange,
                        auctionInfo.auction->request->protocolVersion)) {
            returnInvalidBid(agent, bidsString, auctionInfo.auction,
                    "creativeNotBiddableOnExchange",
                    "creative not biddable on exchange/version");
            continue;
        }

        string auctionKey
            = auctionId.toString() + "-"
            + imp[spotIndex].id.toString() + "-"
            + agent;

        // authorize an amount of money computed from the win cost model.
        Amount price = message.wcm.evaluate(bid, bid.price);

        if (!monitorClient.getStatus(slowModeTolerance)) {
            Date now = Date::now();
            if ((uint32_t) slowModeLastAuction.secondsSinceEpoch()
                    < (uint32_t) now.secondsSinceEpoch()) {
                slowModeLastAuction = now;
                slowModePeriodicSpentReached = false;
                // TODO Insure in router.cc (not router_runner) that
                // maxBidPrice <= slowModeAuthorizedMoneyLimit
                // Here we're garanteed that price.value >= slowModeAuthorizedMoneyLimit
                accumulatedBidMoneyInThisPeriod = price.value;

                recordHit("monitor.systemInSlowMode"); 
            }

            else {
                accumulatedBidMoneyInThisPeriod += price.value;
                // Check if we're spending more in this period than what slowModeAuthorizedMoneyLimit
                // allows us to.
                if (accumulatedBidMoneyInThisPeriod > slowModeAuthorizedMoneyLimit.value) {
                    slowModePeriodicSpentReached = true;
                    bidder->sendBidDroppedMessage(agentConfig, agent, auctionInfo.auction);
                    recordHit("slowMode.droppedBid");
                    recordHit("accounts.%s.IGNORED", config.account.toString('.'));
                continue;
                }
            }
        } else {
            // Make sure slowModePeriodicSpentReached is false if monitor success is satisfied.
            // There is a possible (90% sure..) code path where slowModePeriodicSpentReached is not resetted
            slowModePeriodicSpentReached = false;
        }

        if (!banker->authorizeBid(config.account, auctionKey, price) || failBid(budgetErrorRate))
        {
            ++info.stats->noBudget;

            bidder->sendNoBudgetMessage(agentConfig, agent, auctionInfo.auction);

            this->logMessage("NOBUDGET", agent, auctionId,
                    bidsString, message.meta);
            this->logMessageToAnalytics("NOBUDGET", agent, auctionId);
            recordHit("accounts.%s.NOBUDGET", config.account.toString('.'));
            continue;
        }
        
        recordCount(bid.price.value, "cummulatedBidPrice");
        recordCount(price.value, "cummulatedAuthorizedPrice");


        if (doDebug)
            this->debugSpot(auctionId, imp[spotIndex].id,
                    ML::format("BID %s %s %f",
                            auctionKey.c_str(),
                            bid.price.toString().c_str(),
                            (double)bid.priority));

        std::string meta;
        if (!bid.ext.isNull()) meta = bid.ext.toStringNoNewLine();
        else meta = message.meta;

        Auction::Response response(
                Auction::Price(bid.price, bid.priority),
                creative.id,
                config.account,
                config.test,
                agent,
                bids,
                meta,
                info.config,
                config.visitChannels,
                bid.creativeIndex,
                message.wcm);

        response.creativeName = creative.name;

        Auction::WinLoss localResult
            = auctionInfo.auction->setResponse(spotIndex, response);

        ++numValidBids;

        // Possible results:
        // PENDING: we're currently winning the local auction
        // LOSS: we lost the local auction
        // TOOLATE: we bid too late
        // INVALID: bid was invalid

        string msg = Auction::Response::print(localResult);

        if (doDebug)
            this->debugSpot(auctionId, imp[spotIndex].id,
                    ML::format("BID %s %s",
                            auctionKey.c_str(), msg.c_str()));


        switch (localResult.val) {
        case Auction::WinLoss::PENDING: {
            ++info.stats->bids;
            info.stats->totalBid += bid.price;
            break; // response will be sent later once local winning bid known
        }
        case Auction::WinLoss::LOSS:
            ++info.stats->bids;
            info.stats->totalBid += bid.price;
            // fall through
        case Auction::WinLoss::TOOLATE:
        case Auction::WinLoss::INVALID: {
            if (localResult.val == Auction::WinLoss::TOOLATE)
                ++info.stats->tooLate;
            else if (localResult.val == Auction::WinLoss::INVALID)
                ++info.stats->invalid;

            banker->cancelBid(config.account, auctionKey);

            BidStatus status;
            switch (localResult.val) {
            case Auction::WinLoss::LOSS:
                status = BS_LOSS;
                bidder->sendLossMessage(agentConfig, agent, auctionId.toString ());
                recordHit("accounts.%s.LOCAL_LOSS", config.account.toString('.'));
                break;
            case Auction::WinLoss::TOOLATE:
                status = BS_TOOLATE;
                bidder->sendTooLateMessage(agentConfig, agent, auctionInfo.auction);
                recordHit("accounts.%s.TOOLATE", config.account.toString('.'));
                break;
            case Auction::WinLoss::INVALID:
                status = BS_INVALID;
                bidder->sendBidInvalidMessage(agentConfig, agent, msg, auctionInfo.auction);
                recordHit("accounts.%s.INVALID", config.account.toString('.'));
                break;
            default:
                throw ML::Exception("logic error");
            }

            this->logMessage(msg, agent, auctionId, bidsString, message.meta);
            this->logMessageToAnalytics(msg, agent, auctionId, bidsString);
            continue;
        }
        case Auction::WinLoss::WIN:
            this->throwException("doBid.localWinsNotPossible",
                    "local wins can't be known until auction has closed");

        default:
            this->throwException("doBid.unknownBidResult",
                    "unknown bid result returned by auction");
        }

    }

    if (numValidBids > 0) {
        if (logBids)
            // Send BID to logger
            logMessage("BID", agent, auctionId, bidsString, message.meta);
        logMessageToAnalytics("BID", agent, auctionId, bidsString);
        ML::atomic_add(numNonEmptyBids, 1);
    }
    else if (numPassedBids > 0) {
        // Passed on the ... add to the blacklist
        if (config.hasBlacklist()) {
            const BidRequest & bidRequest = *auctionInfo.auction->request;
            blacklist.add(bidRequest, agent, *info.config);
        }
    }

    for (const auto& agent: message.agents) {
        auctionInfo.bidders.erase(agent);
    }

    double bidTime = dateGotBid.secondsSince(bidInfo.bidTime);

    //cerr << "now " << auctionInfo.bidders.size() << " bidders" << endl;

    //cerr << "campaign " << info.config->campaign << " bidTime "
    //     << 1000.0 * bidTime << endl;

    recordOutcome(1000.0 * bidTime,
                  "accounts.%s.bidResponseTimeMs",
                  config.account.toString('.'));


    if (auctionInfo.bidders.empty()) {
        debugAuction(auctionId, "FINISH", originalMessage);
        if (!auctionInfo.auction->finish()) {
            debugAuction(auctionId, "FINISH TOO LATE", originalMessage);
            recordHit("accounts.%s.FINISH_TOOLATE", agentConfig->account.toString('.'));
        }
        inFlight.erase(auctionId);
        //cerr << "couldn't finish auction " << auctionInfo.auction->id
        //<< " after bid " << message << endl;
    }

}


void
Router::
doSubmitted(std::shared_ptr<Auction> auction)
{
    // Auction was submitted

    // Either a) move it across to the win queue, or b) drop it if we
    // didn't bid anything

    RouterProfiler profiler(dutyCycleCurrent.nsSubmitted);

    const Id & auctionId = auction->id;

#if 0 // debug
    if (recentlySubmitted.count(auctionId)) {
        cerr << "ERROR: auction" << auctionId << " was double submitted"
             << endl;
        return;
    }
    recentlySubmitted.insert(auctionId);
#endif

    //cerr << "SUBMITTED " << auctionId << endl;

    const std::vector<std::vector<Auction::Response> > & allResponses
        = auction->getResponses();

    if (doDebug)
        debugAuction(auctionId, ML::format("SUBMITTED %d slots",
                                           (int)allResponses.size()),
                     {});

    //ExcAssertEqual(allResponses.size(),
    //               auction->bidRequest->imp.size());
    //cerr << "got a win for auction id " << auctionId << " with num imp:" << allResponses.size() << endl;
    // Go through the imp one by one
    for (unsigned spotNum = 0;  spotNum < allResponses.size();  ++spotNum) {

        bool hasSubmittedBid = false;
        Id spotId = auction->request->imp[spotNum].id;

        const std::vector<Auction::Response> & responses
            = allResponses[spotNum];

        if (doDebug)
            debugSpot(auctionId, spotId,
                      ML::format("has %zd bids", responses.size()));

        // For all but the winning bid we tell them what's going on
        for (unsigned i = 0;  i < responses.size();  ++i) {

            const Auction::Response & response = responses[i];
            Auction::WinLoss status = response.localStatus;

            //cerr << "got a response " << response.toJson() << endl;
            //cerr << "response.valid() = " << response.valid() << endl;

            // Don't deal with response 0
            if (i == 0 && response.valid() && response.localStatus.val == Auction::WinLoss::WIN) {
                hasSubmittedBid = true;
                continue;
            }

            //cerr << "doing response " << i << endl;

            if (!agents.count(response.agent)) continue;

            AgentInfo & info = agents[response.agent];
            const auto& agentConfig = info.config;

            Amount bid_price = response.price.maxPrice;

            string auctionKey
                = auctionId.toString() + "-"
                + spotId.toString() + "-"
                + response.agent;

            // Make sure we account for the bid no matter what
            ML::Call_Guard guard
                ([&] ()
                 {
                     banker->cancelBid(response.agentConfig->account, auctionKey);
                 });

            // No bid
            if (bid_price == 0 && response.price.priority == 0) {
                cerr << "warning: auction had no bid result" << endl;
                continue;
            }

            string msg;
            BidStatus bidStatus(BS_INVALID);

            switch (status.val) {
            case Auction::WinLoss::PENDING:
                throwException("doSubmitted.shouldNotBePending",
                               "non-winning auction should not be pending");
            case Auction::WinLoss::WIN:
                if(i == 0) break;
                throwException("doSubmitted.shouldNotBeWin",
                               "auction should not be a win");
            case Auction::WinLoss::INVALID:
                throwException("doSubmitted.shouldNotBeInvalid",
                               "auction should not be invalid");
            case Auction::WinLoss::LOSS:
                bidStatus = BS_LOSS;
                ++info.stats->losses;
                msg = "LOSS";
                bidder->sendLossMessage(agentConfig, response.agent, auctionId.toString());
                recordHit("accounts.%s.LOCAL_LOSS", agentConfig->account.toString('.'));
                break;
            case Auction::WinLoss::TOOLATE:
                bidStatus = BS_TOOLATE;
                ++info.stats->tooLate;
                msg = "TOOLATE";
                bidder->sendTooLateMessage(agentConfig, response.agent, auction);
                recordHit("accounts.%s.TOOLATE", agentConfig->account.toString('.'));
                break;
            default:
                throwException("doSubmitted.unknownStatus",
                               "unknown auction local status");
            };

            if (doDebug)
                debugSpot(auctionId, spotId,
                          ML::format("%s %s",
                                     msg.c_str(),
                                     auctionKey.c_str()));
        }

        // If we didn't actually submit a bid then nothing else to do
        if (!hasSubmittedBid) continue;
        this->recordHit("numRequestWithBid");
        ML::atomic_add(numAuctionsWithBid, 1);
        //cerr << fName << "injecting submitted auction " << endl;

        logMessageToAnalytics("SUBMITTED", auction->id, responses[0].agent, responses[0].price.toJsonStr());
        onSubmittedAuction(auction, spotId, responses[0]);
        //postAuctionLoop.injectSubmittedAuction(auction, spotId, responses[0]);
    }

    //cerr << "auction.use_count() = " << auction.use_count() << endl;

    if (auction.unique()) {
        auctionGraveyard.tryPush(auction);
    }
}

std::string
reduceUrl(const Url & url)
{
    static const boost::regex rex(".*://(www.)?([^/]+)");
    boost::match_results<string::const_iterator> mr;
    string s = url.toString();
    if (!boost::regex_search(s, mr, rex)) {
        //cerr << "warning: nothing matched in URL " << url << endl;
        return s;
    }

    if (mr.size() != 3) {
        cerr << "warning: wrong match results size "
             << mr.size() << " in URL " << url << endl;
        return s;
    }

    //cerr << "url " << url << " reduced to " << mr.str(2) << endl;

    return mr.str(2);
}


void
Router::
onNewAuction(std::shared_ptr<Auction> auction)
{
    if (!monitorClient.getStatus(slowModeTolerance)) {
        // check if slow mode active and in the same second then ignore the Auction
        Date now = Date::now();
        // TODO slowModeLastAuction is not atomic and a race condition could happen since it's
        // used by router loop AND exchange connector threads
        if (slowModePeriodicSpentReached && (uint32_t) slowModeLastAuction.secondsSinceEpoch()
                == (uint32_t) now.secondsSinceEpoch() ) {
            recordHit("monitor.ignoredAuctions");
            auction->finish();
            return;
        }
    }

    //cerr << "AUCTION GOT THROUGH" << endl;

    if (logAuctions)
        // Send AUCTION to logger
        logMessage("AUCTION", auction->id, auction->requestStr);
    logMessageToAnalytics("AUCTION", auction->id);

    const BidRequest & request = *auction->request;
    int numFields = 0;
    if (!request.url.empty()) ++numFields;
    if (request.userIds.exchangeId) ++numFields;
    if (request.userIds.providerId) ++numFields;

    auto info = preprocessAuction(auction);

    if (info) {
        recordHit("auctionPassedPreprocessing");
        augmentAuction(info);
    }
    else {
        recordHit("auctionDropped.noPotentialBidders");
        ML::atomic_inc(numNoPotentialBidders);
    }
}

void
Router::
onAuctionDone(std::shared_ptr<Auction> auction)
{
#if 0
    static std::mutex lock;
    std::unique_lock<std::mutex> guard(lock);

    cerr << endl;
    cerr << "Router::onAuctionDone with auction id " << auction->id << endl;
    backtrace();
#endif

    debugAuction(auction->id, "SENT SUBMITTED");
    submittedBuffer.push(auction);
}

void
Router::
onAuctionError(const std::string & channel,
               std::shared_ptr<Auction> auction,
               const std::string & message)
{
    if (auction) {
//         cout << channel << " " << auction->requestStr << " " << message << endl;
        logMessageToAnalytics(channel, auction->id, message);
    }
    else {
//         cout << channel << " " << message << endl;
        logMessageToAnalytics(channel, message);
    }
}

void
Router::
updateAllAgents()
{
    for (;;) {

        auto_ptr<AllAgentInfo> newInfo(new AllAgentInfo);

        AllAgentInfo * current = allAgents;

        for (auto it = agents.begin(), end = agents.end();  it != end;  ++it) {
            if (!it->second.configured) continue;
            if (!it->second.config) continue;
            if (!it->second.stats) continue;
            if (!it->second.status) continue;
            if (it->second.status->dead) continue;

            AgentInfoEntry entry;
            entry.name = it->first;
            entry.filterIndex = it->second.filterIndex;
            entry.config = it->second.config;
            entry.stats = it->second.stats;
            entry.status = it->second.status;
            int i = newInfo->size();
            newInfo->push_back(entry);

            newInfo->agentIndex[it->first] = i;
            newInfo->accountIndex[it->second.config->account].push_back(i);
        }

        if (ML::cmp_xchg(allAgents, current, newInfo.get())) {
            newInfo.release();
            ExcAssertNotEqual(current, allAgents);
            if (current)
                allAgentsGc.defer([=] () { delete current; });
            break;
        }
    }
}

void
Router::
doConfig(const std::string & agent,
         std::shared_ptr<const AgentConfig> config)
{
    RouterProfiler profiler(dutyCycleCurrent.nsConfig);

    if (!config) {
        auto it = agents.find(agent);
        // It might happen that we don't find the agent if for example we received
        // an empty configuration because the agent crashed prior to sending its initial
        // configuration to the ACS.
        if (it != std::end(agents)) {
            cerr << "agent " << agent << " lost configuration" << endl;
            filters.removeConfig(agent);
            agents.erase(it);
        }
    } else {
        AgentInfo & info = agents[agent];
        logMessage("CONFIG", agent, boost::trim_copy(config->toJson().toString()));
        logMessageToAnalytics("CONFIG", agent, boost::trim_copy(config->toJson().toString()));

        // TODO: no need for this...
        auto newConfig = std::make_shared<AgentConfig>(*config);
        if (newConfig->roundRobinGroup == "")
            newConfig->roundRobinGroup = agent;


        if (info.configured) {
            unconfigure(agent, *info.config);
            info.configured = false;
        }

        info.config = newConfig;
        //cerr << "configured " << agent << " strategy : " << info.config->strategy << " campaign "
        //     <<  info.config->campaign << endl;

        string bidRequestFormat = "jsonRaw";
        info.setBidRequestFormat(bidRequestFormat);

        configure(agent, *newConfig);
        info.configured = true;
        bidder->sendMessage(config, agent, "GOTCONFIG");

        info.filterIndex = filters.addConfig(agent, info);
    }

    // Broadcast that we have a new agent or it has a new configuration
    updateAllAgents();
}

void
Router::
unconfigure(const std::string & agent, const AgentConfig & config)
{
}

void
Router::
configureAgentOnExchange(std::shared_ptr<ExchangeConnector> const & exchange,
                         std::string const & agent,
                         AgentConfig & config,
                         bool includeReasons)
{
    auto name = exchange->exchangeName();

    auto ecomp = exchange->getCampaignCompatibility(config, includeReasons);
    if(!ecomp.isCompatible) {
        cerr << "campaign not compatible: " << ecomp.reasons << endl;
        this->recordHit("%s.compaignNotCompatible", name);
        return;
    }

    int numCompatibleCreatives = 0;

    for(auto & c : config.creatives) {
        auto ccomp = exchange->getCreativeCompatibility(c, includeReasons);
        if(!ccomp.isCompatible) {
            cerr << "creative not compatible: " << ccomp.reasons << endl;
            this->recordHit("%s.creativeNotCompatible", name);
        }
        else {
            std::lock_guard<ML::Spinlock> guard(c.lock);
            c.providerData[name] = ccomp.info;
            ++numCompatibleCreatives;
        }
    }

    if (numCompatibleCreatives == 0) {
        cerr << "no compatible creatives" << endl;
        this->recordHit("%s.noCompatibleCreative", name);
        return;
    }

    std::lock_guard<ML::Spinlock> guard(config.lock);
    config.providerData[name] = ecomp.info;
}

void
Router::
configure(const std::string & agent, AgentConfig & config)
{
    if (config.account.empty())
        throw ML::Exception("attempt to add an account with empty values");

    // For each exchange, check campaign and creative compatibility
    forAllExchanges([&] (const std::shared_ptr<ExchangeConnector> & exchange) {
        configureAgentOnExchange(exchange, agent, config);
    });

    auto onDone = [=] (std::exception_ptr exc, ShadowAccount&& ac)
        {
            //cerr << "got spend account for " << agent << ac << endl;
            try {
                if (exc)
                    logException(exc, "Banker addAccount");
            }
            catch(ML::Exception const & e) {
            }
        };

    banker->addSpendAccount(config.account, Amount(), onDone);
}

Json::Value
Router::
getStats() const
{
    return Json::Value();
#if 0
    sendMesg(control(), "STATS");
    vector<string> stats = recvAll(control());

    Json::Value result = Json::parse(stats.at(0));

    return result;
#endif
}

Json::Value
Router::
getAgentInfo(const std::string & agent) const
{
    return getAgentEntry(agent).toJson();
}

Json::Value
Router::
getAllAgentInfo() const
{
    Json::Value result;

    auto onAgent = [&] (const AgentInfoEntry & info)
        {
            result[info.name] = info.toJson();
        };

    forEachAgent(onAgent);

    return result;
}

void
Router::
sendPings()
{
    for (auto it = agents.begin(), end = agents.end();
         it != end;  ++it) {
        const string & agent = it->first;
        AgentInfo & info = it->second;
        const auto& agentConfig = info.config;

        // 1.  Send out new pings
        Date now = Date::now();
        if (info.sendPing(0, now))
            bidder->sendPingMessage(agentConfig, agent, 0);
        if (info.sendPing(1, now))
            bidder->sendPingMessage(agentConfig, agent, 1);

        // 2.  Look at the trend
        //double mean, max;
    }
}

void
Router::
doPong(int level, const std::vector<std::string> & message)
{
    //cerr << "dopong (router)" << message << endl;

    string agent = message.at(0);
    Date sentTime = Date::parseSecondsSinceEpoch(message.at(2));
    Date receivedTime = Date::parseSecondsSinceEpoch(message.at(3));
    Date now = Date::now();

    double roundTripTime = now.secondsSince(sentTime);
    double outgoingTime = receivedTime.secondsSince(sentTime);
    double incomingTime = now.secondsSince(receivedTime);

    auto it = agents.find(agent);
    if (it == agents.end()) {
        cerr << "warning: dead agent sent a pong: " << agent << endl;
        return;
    }

    if (!it->second.configured)
        return;

    auto & info = it->second;
    info.gotPong(level, sentTime, receivedTime, now);

    const string & account = it->second.config->account.toString('.');
    recordOutcome(roundTripTime * 1000.0,
                  "accounts.%s.ping%d.roundTripTimeMs", account, level);
    recordOutcome(outgoingTime * 1000.0,
                  "accounts.%s.ping%d.outgoingTimeMs", account, level);
    recordOutcome(incomingTime * 1000.0,
                  "accounts.%s.ping%d.incomingTimeMs", account, level);
}

void
Router::
forEachAgent(const OnAgentFn & onAgent) const
{
    GcLock::SharedGuard guard(allAgentsGc);
    const AllAgentInfo * ac = allAgents;
    if (!ac) return;

    std::for_each(ac->begin(), ac->end(), onAgent);
}

void
Router::
forEachAccountAgent(const AccountKey & account,
                    const OnAgentFn & onAgent) const
{
    GcLock::SharedGuard guard(allAgentsGc);
    const AllAgentInfo * ac = allAgents;
    if (!ac) return;

    auto it = ac->accountIndex.find(account);
    if (it == ac->accountIndex.end())
        return;

    for (auto jt = it->second.begin(), jend = it->second.end();
         jt != jend;  ++jt)
        onAgent(ac->at(*jt));
}

AgentInfoEntry
Router::
getAgentEntry(const std::string & agent) const
{
    GcLock::SharedGuard guard(allAgentsGc);
    const AllAgentInfo * ac = allAgents;
    if (!ac) return AgentInfoEntry();

    auto it = ac->agentIndex.find(agent);
    if (it == ac->agentIndex.end())
        return AgentInfoEntry();
    return ac->at(it->second);
}

void
Router::
submitToPostAuctionService(std::shared_ptr<Auction> auction,
                           Id adSpotId,
                           const Auction::Response & bid)
{
#if 0
    static std::mutex lock;
    std::unique_lock<std::mutex> guard(lock);

    cerr << endl;
    cerr << "submitted auction " << auction->id << ","
         << adSpotId << endl;

    backtrace();
#endif
    string auctionKey = auction->id.toString()
                        + "-" + adSpotId.toString()
                        + "-" + bid.agent;

    banker->detachBid(bid.account, auctionKey);

    recordHit("accounts.%s.submitted", bid.account.toString('.'));

    if (connectPostAuctionLoop) {
        auto event = std::make_shared<SubmittedAuctionEvent>();
        event->auctionId = auction->id;
        event->adSpotId = adSpotId;
        event->lossTimeout = auction->lossAssumed;
        event->augmentations = auction->agentAugmentations[bid.agent];
        event->bidRequest(auction->request);
        event->bidRequestStr = auction->requestStr;
        event->bidRequestStrFormat = auction->requestStrFormat ;
        event->bidResponse = bid;

        postAuctionEndpoint.sendAuction(event);
    }

    if (auction.unique()) {
        auctionGraveyard.tryPush(auction);
    }
}

void
Router::
throwException(const std::string & key, const std::string & fmt, ...)
{
    recordHit("error.exception");
    recordHit("error.exception.%s", key);

    string message;
    va_list ap;
    va_start(ap, fmt);
    try {
        message = vformat(fmt.c_str(), ap);
        va_end(ap);
    }
    catch (...) {
        va_end(ap);
        throw;
    }

    logRouterError("exception", key, message);
    throw ML::Exception("Router Exception: " + key + ": " + message);
}

void
Router::
debugAuctionImpl(const Id & auctionId, const std::string & type,
                 const std::vector<std::string> & args)
{
    Date now = Date::now();
    boost::unique_lock<ML::Spinlock> guard(debugLock);
    AuctionDebugInfo & entry
        = debugInfo.access(auctionId, now.plusSeconds(30.0));

    entry.addAuctionEvent(now, type, args);
}

void
Router::
debugSpotImpl(const Id & auctionId, const Id & spotId, const std::string & type,
              const std::vector<std::string> & args)
{
    Date now = Date::now();
    boost::unique_lock<ML::Spinlock> guard(debugLock);
    AuctionDebugInfo & entry
        = debugInfo.access(auctionId, now.plusSeconds(30.0));

    entry.addSpotEvent(spotId, now, type, args);
}

void
Router::
expireDebugInfo()
{
    boost::unique_lock<ML::Spinlock> guard(debugLock);
    debugInfo.expire();
}

void
Router::
dumpAuction(const Id & auctionId) const
{
    boost::unique_lock<ML::Spinlock> guard(debugLock);
    auto it = debugInfo.find(auctionId);
    if (it == debugInfo.end()) {
        //cerr << "*** unknown auction " << auctionId << " in "
        //     << debugInfo.size() << endl;
    }
    else it->second.dumpAuction();
}

void
Router::
dumpSpot(const Id & auctionId, const Id & spotId) const
{
    boost::unique_lock<ML::Spinlock> guard(debugLock);
    auto it = debugInfo.find(auctionId);
    if (it == debugInfo.end()) {
        //cerr << "*** unknown auction " << auctionId << " in "
        //     << debugInfo.size() << endl;
    }
    else it->second.dumpSpot(spotId);
}

/** MonitorProvider interface */
string
Router::
getProviderClass()
    const
{
    return "rtbRequestRouter";
}

MonitorIndicator
Router::
getProviderIndicators()
    const
{
    bool connectedToPal = !connectPostAuctionLoop || postAuctionEndpoint.isConnected();
    bool bankerOk = banker->getProviderIndicators().status;

    MonitorIndicator ind;

    ind.serviceName = serviceName();
    ind.status = connectedToPal && bankerOk;
    ind.message = string()
        + "Connection to PAL: " + (connectedToPal ? "OK" : "ERROR") + ", "
        + "Banker: " + (bankerOk ? "OK": "ERROR");

    return ind;
}



} // namespace RTBKIT

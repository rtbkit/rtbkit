/* rtb_router.cc
   Jeremy Barnes, 24 March 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   RTB router code.
*/

#include <set>
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

/* static */ const std::vector<std::string> Router::emptyDebugArgs{};

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
       bool logAuctions,
       bool logBids,
       Amount maxBidAmount)
    : ServiceBase(serviceName, parent),
      shutdown_(false),
      agentEndpoint(getZmqContext()),
      configBuffer(1024),
      exchangeBuffer(64),
      startBiddingBuffer(65536),
      submittedBuffer(65536),
      auctionGraveyard(65536),
      augmentationLoop(*this),
      loopMonitor(*this),
      loadStabilizer(loopMonitor),
      secondsUntilLossAssumed_(secondsUntilLossAssumed),
      globalBidProbability(1.0),
      bidsErrorRate(0.0),
      budgetErrorRate(0.0),
      connectPostAuctionLoop(connectPostAuctionLoop),
      allAgents(new AllAgentInfo()),
      configListener(getZmqContext()),
      initialized(false),
      logAuctions(logAuctions),
      logBids(logBids),
      logger(getZmqContext()),
      doDebug(false),
      numAuctions(0), numBids(0), numNonEmptyBids(0),
      numAuctionsWithBid(0), numNoPotentialBidders(0),
      numNoBidders(0),
      monitorClient(getZmqContext()),
      slowModeCount(0),
      monitorProviderClient(getZmqContext(), *this),
      maxBidAmount(maxBidAmount)
{
}

Router::
Router(std::shared_ptr<ServiceProxies> services,
       const std::string & serviceName,
       double secondsUntilLossAssumed,
       bool connectPostAuctionLoop,
       bool logAuctions,
       bool logBids,
       Amount maxBidAmount)
    : ServiceBase(serviceName, services),
      shutdown_(false),
      agentEndpoint(getZmqContext()),
      postAuctionEndpoint(getZmqContext()),
      configBuffer(1024),
      exchangeBuffer(64),
      startBiddingBuffer(65536),
      submittedBuffer(65536),
      auctionGraveyard(65536),
      augmentationLoop(*this),
      loopMonitor(*this),
      loadStabilizer(loopMonitor),
      secondsUntilLossAssumed_(secondsUntilLossAssumed),
      globalBidProbability(1.0),
      bidsErrorRate(0.0),
      budgetErrorRate(0.0),
      connectPostAuctionLoop(connectPostAuctionLoop),
      allAgents(new AllAgentInfo()),
      configListener(getZmqContext()),
      initialized(false),
      logAuctions(logAuctions),
      logBids(logBids),
      logger(getZmqContext()),
      doDebug(false),
      numAuctions(0), numBids(0), numNonEmptyBids(0),
      numAuctionsWithBid(0), numNoPotentialBidders(0),
      numNoBidders(0),
      monitorClient(getZmqContext()),
      slowModeCount(0),
      monitorProviderClient(getZmqContext(), *this),
      maxBidAmount(maxBidAmount)
{
}

void
Router::
init()
{
    ExcAssert(!initialized);

    registerServiceProvider(serviceName(), { "rtbRequestRouter" });

    filters.init(this);
    FilterPool::initWithDefaultFilters(filters);

    banker.reset(new NullBanker());

    augmentationLoop.init();

    logger.init(getServices()->config, serviceName() + "/logger");


    agentEndpoint.init(getServices()->config, serviceName() + "/agents");
    agentEndpoint.clientMessageHandler
        = std::bind(&Router::handleAgentMessage, this, std::placeholders::_1);
    agentEndpoint.onConnection = [=] (const std::string & agent)
        {
            cerr << "agent " << agent << " connected to router" << endl;
        };

    agentEndpoint.onDisconnection = [=] (const std::string & agent)
        {
            cerr << "agent " << agent << " disconnected from router" << endl;
        };

    postAuctionEndpoint.init(getServices()->config, ZMQ_XREQ);

    configListener.onConfigChange = [=] (const std::string & agent,
                                         std::shared_ptr<const AgentConfig> config)
        {
            cerr << endl << endl << "agent " << agent << " got new configuration" << endl;
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

    loopMonitor.onLoadChange = [=] (double)
        {
            double keepProb = 1.0 - loadStabilizer.shedProbability();

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
}

void
Router::
bindTcp()
{
    logger.bindTcp(getServices()->ports->getRange("logs"));
    agentEndpoint.bindTcp(getServices()->ports->getRange("router"));
}

void
Router::
bindAgents(std::string agentUri)
{
    try {
        agentEndpoint.bind(agentUri.c_str());
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
    monitorProviderClient.inhibit_ = true;
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

    logger.start();
    augmentationLoop.start();
    runThread.reset(new boost::thread(runfn));

    if (connectPostAuctionLoop) {
        postAuctionEndpoint.connectToServiceClass("rtbPostAuctionService", "events");
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
        { agentEndpoint.getSocketUnsafe(), 0, ZMQ_POLLIN, 0 },
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


    // Attempt to wake up once per millisecond

    Date lastSleep = Date::now();

    while (!shutdown_) {
        beforeSleep = getTime();

        totalActive += beforeSleep - afterSleep;
        dutyCycleCurrent.nsProcessing
            += microsecondsBetween(beforeSleep, afterSleep);

        int rc = 0;

        for (unsigned i = 0;  i < 20 && rc == 0;  ++i)
            rc = zmq_poll(items, 2, 0);
        if (rc == 0) {
            ++numTimesCouldSleep;
            checkExpiredAuctions();

#if 1
            // Try to sleep only once per 1/2 a millisecond to avoid too many
            // context switches.
            Date now = Date::now();
            double timeSinceSleep = lastSleep.secondsUntil(now);
            double timeToWait = 0.0005 - timeSinceSleep;
            if (timeToWait > 0) {
                ML::sleep(timeToWait);
            }
            lastSleep = now;
#endif


            rc = zmq_poll(items, 2, 50 /* milliseconds */);
        }

        //cerr << "rc = " << rc << endl;

        afterSleep = getTime();

        dutyCycleCurrent.nsSleeping
            += microsecondsBetween(afterSleep, beforeSleep);
        dutyCycleCurrent.nEvents += 1;

        times["asleep"].add(microsecondsBetween(afterSleep, beforeSleep));

        if (rc == -1 && zmq_errno() != EINTR) {
            cerr << "zeromq error: " << zmq_strerror(zmq_errno()) << endl;
        }

        {
            double atStart = getTime();
            std::shared_ptr<AugmentationInfo> info;
            while (startBiddingBuffer.tryPop(info)) {
                doStartBidding(info);
            }

            double atEnd = getTime();
            times["doStartBidding"].add(microsecondsBetween(atEnd, atStart));
        }

        {
            std::shared_ptr<ExchangeConnector> exchange;
            while (exchangeBuffer.tryPop(exchange)) {
                for (auto & agent : agents) {
                    configureAgentOnExchange(exchange,
                                             agent.first,
                                             *agent.second.config);
                };
            }
        }

        {
            double atStart = getTime();

            std::pair<std::string, std::shared_ptr<const AgentConfig> > config;
            while (configBuffer.tryPop(config)) {
                if (!config.second) {
                    // deconfiguration
                    // TODO
                    cerr << "agent " << config.first << " lost configuration"
                         << endl;
                }
                else {
                    doConfig(config.first, config.second);
                }
            }

            double atEnd = getTime();
            times["doConfig"].add(microsecondsBetween(atEnd, atStart));
        }

        {
            double atStart = getTime();
            std::shared_ptr<Auction> auction;
            while (submittedBuffer.tryPop(auction))
                doSubmitted(auction);

            double atEnd = getTime();
            times["doSubmitted"].add(microsecondsBetween(atEnd, atStart));
        }

        if (items[0].revents & ZMQ_POLLIN) {
            double beforeMessage = getTime();
            // Agent message
            vector<string> message;
            try {
                message = recvAll(agentEndpoint.getSocketUnsafe());
                agentEndpoint.handleMessage(std::move(message));
                double atEnd = getTime();
                times[message.at(1)].add(microsecondsBetween(atEnd, beforeMessage));
            } catch (const std::exception & exc) {
                cerr << "error handling agent message " << message
                     << ": " << exc.what() << endl;
                logRouterError("handleAgentMessage", exc.what(),
                               message);
            }
        }

        if (items[1].revents & ZMQ_POLLIN) {
            wakeupMainLoop.read();
        }

        //checkExpiredAuctions();

        double now = ML::wall_time();
        double beforeChecks = getTime();

        if (now - lastPings > 1.0) {
            // Send out pings and interpret the results of the last lot of
            // pinging.
            sendPings();

            lastPings = now;
        }

        if (now - last_check_pace > 10.0) {
            recordEvent("numTimesCouldSleep", ET_LEVEL,
                        numTimesCouldSleep);

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

        times["checks"].add(microsecondsBetween(getTime(), beforeChecks));

        if (now - lastTimestamp >= 1.0) {
            banker->logBidEvents(*this);
            //issueTimestamp();
            lastTimestamp = now;
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

        if (request.empty())
            returnErrorResponse(message, "null request field");

        if (request == "CONFIG") {
            string configName = message.at(2);
            if (!agents.count(configName)) {
                // We don't yet know about its configuration
                sendAgentMessage(address, "NEEDCONFIG", getCurrentTime());
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

    set<AccountKey> agentAccounts;
    for (const auto & item : agents) {
        auto & info = item.second;
        const AccountKey & account = info.config->account;
        if (!agentAccounts.insert(account).second) {
            continue;
        }

        auto & last = lastAgentUsageMetrics[item.first];

        AgentUsageMetrics newMetrics(info.stats->intoFilters,
                                     info.stats->passedStaticFilters,
                                     info.stats->auctions,
                                     info.stats->bids);
        AgentUsageMetrics delta = newMetrics - last;

        logMessage("USAGE", "AGENT", p, item.first,
                   info.config->account.toString(),
                   delta.intoFilters,
                   delta.passedStaticFilters,
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

                    this->sendBidResponse(it->first,
                                          info,
                                          BS_LOSTBID,
                                          this->getCurrentTime(),
                                          "guaranteed", id);

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
                sendAgentMessage(it->first, "BYEBYE", getCurrentTime());
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

                        this->recordHit("accounts.%s.droppedBids",
                                        info.config->account.toString('.'));

                        this->sendBidResponse(agent,
                                              info,
                                              BS_DROPPEDBID,
                                              this->getCurrentTime(),
                                              "guaranteed",
                                              auctionId,
                                              0, Amount(),
                                              auctionInfo.auction.get());
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
    sendAgentMessage(message[0], "ERROR", getCurrentTime(), error, message);
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

    double augmentationWindow = 0.005; // 5ms available to augment

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

    augmentationLoop.augment(info, Date::now().plusSeconds(augmentationWindow),
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

    AgentConfig::RequestFilterCache cache(*auction->request);

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

    for (auto it = groupAgents.begin(), end = groupAgents.end();
         it != end;  ++it) {
        // Check for bid probability and skip if we don't bid
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

        // Group is valid for bidding; next step is to augment the bid
        // request
        validGroups.push_back(it->second);
    }

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

            WinCostModel wcm = auction->exchangeConnector->getWinCostModel(*auction,
                                                                           *winner.config);

            //cerr << "sending to agent " << agent << endl;
            //cerr << fName << " sending AUCTION message " << endl;c
            /* Convert to JSON to send it on. */
            sendAgentMessage(agent,
                             "AUCTION",
                             auction->start,
                             auctionId,
                             info.getBidRequestEncoding(*auction),
                             info.encodeBidRequest(*auction),
                             winner.imp.toJsonStr(),
                             toString(timeLeftMs),
                             auction->agentAugmentations[agent],
                             wcm.toJson());

            //cerr << "done" << endl;
        }

        //cerr << " auction " << id << " with "
        //     << auctionInfo.bidders.size() << " bidders" << endl;

        //auctionInfo.activities.push_back(ML::format("total of %zd agents",
        //                                 auctionInfo.bidders.size()));
        if (auction->tooLate()) {
            recordHit("tooLateAfterRouting");
            // Unwind everything?
        }

        if (auctionInfo.bidders.empty()) {
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
    //static const char *fName = "Router::doBid:";
    if (failBid(bidsErrorRate)) {
        returnErrorResponse(message, "Intentional error response (--bids-error-rate)");
        return;
    }

    Date dateGotBid = Date::now();

    RouterProfiler profiler(dutyCycleCurrent.nsBid);

    ML::atomic_inc(numBids);

    if (message.size() < 5 || message.size() > 6) {
        returnErrorResponse(message, "BID message has 4-5 parts");
        return;
    }

    static std::map<const char *, unsigned long long> times;

    static Date lastPrinted = Date::now();

    if (lastPrinted.secondsUntil(dateGotBid) > 10.0) {
#if 0
        unsigned long long total = 0;
        for (auto it = times.begin(), end = times.end(); it != end;  ++it)
            total += it->second;

        cerr << "doBid of " << total << " microseconds" << endl;
        cerr << "id = " << message[2] << endl;
        for (auto it = times.begin(), end = times.end();
             it != end;  ++it) {
            cerr << ML::format("%-30s %8lld %6.2f%%\n",
                               it->first,
                               it->second,
                               100.0 * it->second / total);
        }
#endif
        lastPrinted = dateGotBid;
        times.clear();
    }

    double current = getProfilingTime();

    auto doProfileEvent = [&] (int i, const char * what)
        {
            return;
            double after = getProfilingTime();
            times[what] += microsecondsBetween(after, current);
            current = after;
        };

    recordHit("bid");

    doProfileEvent(0, "start");

    Id auctionId(message[2]);

    const string & agent = message[0];
    const string & biddata = message[3];
    const string & model = message[4];

    WinCostModel wcm = WinCostModel::fromJson(model.empty() ? Json::Value() : Json::parse(model));

    static const string nullStr("null");
    const string & meta = (message.size() >= 6 ? message[5] : nullStr);

    doProfileEvent(1, "params");

    debugAuction(auctionId, "BID", message);

    if (!agents.count(agent)) {
        returnErrorResponse(message, "unknown agent");
        return;
    }

    doProfileEvent(2, "agents");

    AgentInfo & info = agents[agent];

    /* One less in flight. */
    if (!info.expireBidInFlight(auctionId)) {
        recordHit("bidError.agentNotBidding");
        returnErrorResponse(message, "agent wasn't bidding on this auction");
        return;
    }

    doProfileEvent(3, "inFlight");

    auto it = inFlight.find(auctionId);
    if (it == inFlight.end()) {
        recordHit("bidError.unknownAuction");
        returnErrorResponse(message, "unknown auction");
        return;
    }

    doProfileEvent(4, "account");

    AuctionInfo & auctionInfo = it->second;

    auto biddersIt = auctionInfo.bidders.find(agent);
    if (biddersIt == auctionInfo.bidders.end()) {
        recordHit("bidError.agentSkippedAuction");
        returnErrorResponse(message,
                            "agent shouldn't bid on this auction");
        return;
    }

    auto & config = *biddersIt->second.agentConfig;

    recordHit("accounts.%s.bids", config.account.toString('.'));

    doProfileEvent(5, "auctionInfo");

    //cerr << "info.inFlight = " << info.inFlight << endl;

    const std::vector<AdSpot> & imp = auctionInfo.auction->request->imp;

    int numValidBids = 0;

    auto returnInvalidBid = [&] (int i, const char * reason,
                                 const char * message, ...)
        {
            this->recordHit("bidErrors.%s", reason);
            this->recordHit("accounts.%s.bidErrors.total",
                            config.account.toString('.'));
            this->recordHit("accounts.%s.bidErrors.%s",
                            config.account.toString('.'),
                            reason);

            ++info.stats->invalid;

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
            cerr << biddata << endl;

            this->sendBidResponse
                (agent, info, BS_INVALID, this->getCurrentTime(),
                 formatted, auctionId,
                 i, Amount(),
                 auctionInfo.auction.get(),
                 biddata, Json::Value(),
                 auctionInfo.auction->agentAugmentations[agent]);
        };

    BidInfo bidInfo(std::move(biddersIt->second));
    auctionInfo.bidders.erase(biddersIt);

    doProfileEvent(6, "bidInfo");

    int numPassedBids = 0;

    Bids bids;
    try {
        bids = Bids::fromJson(biddata);
    }
    catch (const std::exception & exc) {
        returnInvalidBid(-1, "bidParseError",
                "couldn't parse bid JSON %s: %s", biddata.c_str(), exc.what());
        return;
    }

    doProfileEvent(6, "parsing");

    ExcCheckEqual(bids.size(), bidInfo.imp.size(),
            "invalid shape for bids array");

    auctionInfo.auction->addDataSources(bids.dataSources);

    for (int i = 0; i < bids.size(); ++i) {

        const Bid& bid = bids[i];

        if (bid.isNullBid()) {
            ++numPassedBids;
            continue;
        }

        int spotIndex = bidInfo.imp[i].first;

        if (bid.creativeIndex == -1) {
            returnInvalidBid(i, "nullCreativeField",
                    "creative field is null in response %s",
                    biddata.c_str());
            continue;
        }

        if (bid.creativeIndex < 0
                || bid.creativeIndex >= config.creatives.size())
        {
            returnInvalidBid(i, "outOfRangeCreative",
                    "parsing field 'creative' of %s: creative "
                    "number %d out of range 0-%zd",
                    biddata.c_str(), bid.creativeIndex,
                    config.creatives.size());
            continue;
        }

        if (bid.price.isNegative() || bid.price > maxBidAmount) {
            returnInvalidBid(i, "invalidPrice",
                    "bid price of %s is outside range of $0-%s parsing bid %s",
                    bid.price.toString().c_str(),
                    maxBidAmount.toString().c_str(),
                    biddata.c_str());
            continue;
        }

        const Creative & creative = config.creatives.at(bid.creativeIndex);

        if (!creative.compatible(imp[spotIndex])) {
#if 1
            cerr << "creative not compatible with spot: " << endl;
            cerr << "auction: " << auctionInfo.auction->requestStr
                << endl;
            cerr << "config: " << config.toJson() << endl;
            cerr << "bid: " << biddata << endl;
            cerr << "spot: " << imp[i].toJson() << endl;
            cerr << "spot num: " << spotIndex << endl;
            cerr << "bid num: " << i << endl;
            cerr << "creative num: " << bid.creativeIndex << endl;
            cerr << "creative: " << creative.toJson() << endl;
#endif
            returnInvalidBid(i, "creativeNotCompatibleWithSpot",
                    "creative %s not compatible with spot %s",
                    creative.toJson().toString().c_str(),
                    imp[spotIndex].toJson().toString().c_str());
            continue;
        }

        if (!creative.biddable(auctionInfo.auction->request->exchange,
                        auctionInfo.auction->request->protocolVersion)) {
            returnInvalidBid(i, "creativeNotBiddableOnExchange",
                    "creative not biddable on exchange/version");
            continue;
        }

        doProfileEvent(6, "creativeCompatibility");

        string auctionKey
            = auctionId.toString() + "-"
            + imp[spotIndex].id.toString() + "-"
            + agent;

        // authorize an amount of money computed from the win cost model.
        Amount price = wcm.evaluate(bid, bid.price);

        if (!banker->authorizeBid(config.account, auctionKey, price)
                || failBid(budgetErrorRate))
        {
            ++info.stats->noBudget;
            const string& agentAugmentations =
                auctionInfo.auction->agentAugmentations[agent];

            this->sendBidResponse(agent, info, BS_NOBUDGET,
                    this->getCurrentTime(),
                    "guaranteed", auctionId, 0, Amount(),
                    auctionInfo.auction.get(),
                    biddata, meta, agentAugmentations);
            this->logMessage("NOBUDGET", agent, auctionId,
                    biddata, meta);
            continue;
        }

	recordCount(bid.price.value, "cummulatedBidPrice");
	recordCount(price.value, "cummulatedAuthorizedPrice");

        doProfileEvent(6, "banker");

        if (doDebug)
            this->debugSpot(auctionId, imp[spotIndex].id,
                    ML::format("BID %s %s %f",
                            auctionKey.c_str(),
                            bid.price.toString().c_str(),
                            (double)bid.priority));

        Auction::Price bidprice(bid.price, bid.priority);
        Auction::Response response(
                bidprice,
                creative.id,
                config.account,
                config.test,
                agent,
                biddata,
                meta,
                info.config,
                config.visitChannels,
                bid.creativeIndex,
                wcm);

        response.creativeName = creative.name;

        Auction::WinLoss localResult
            = auctionInfo.auction->setResponse(spotIndex, response);

        doProfileEvent(6, "bidSubmission");
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
            case Auction::WinLoss::LOSS:    status = BS_LOSS;     break;
            case Auction::WinLoss::TOOLATE: status = BS_TOOLATE;  break;
            case Auction::WinLoss::INVALID: status = BS_INVALID;  break;
            default:
                throw ML::Exception("logic error");
            }

            const string& agentAugmentations =
                auctionInfo.auction->agentAugmentations[agent];

            this->sendBidResponse(agent, info, status,
                    this->getCurrentTime(),
                    "guaranteed", auctionId, 0, Amount(),
                    auctionInfo.auction.get(),
                    biddata, meta, agentAugmentations);
            this->logMessage(msg, agent, auctionId, biddata, meta);
            continue;
        }
        case Auction::WinLoss::WIN:
            this->throwException("doBid.localWinsNotPossible",
                    "local wins can't be known until auction has closed");

        default:
            this->throwException("doBid.unknownBidResult",
                    "unknown bid result returned by auction");
        }

        doProfileEvent(6, "bidResponse");
    }

    if (numValidBids > 0) {
        if (logBids)
            // Send BID to logger
            logMessage("BID", agent, auctionId, biddata, meta);
        ML::atomic_add(numNonEmptyBids, 1);
    }
    else if (numPassedBids > 0) {
        // Passed on the ... add to the blacklist
        if (config.hasBlacklist()) {
            const BidRequest & bidRequest = *auctionInfo.auction->request;
            blacklist.add(bidRequest, agent, *info.config);
        }
        doProfileEvent(8, "blacklist");
    }

    doProfileEvent(8, "postParsing");

    double bidTime = dateGotBid.secondsSince(bidInfo.bidTime);

    //cerr << "now " << auctionInfo.bidders.size() << " bidders" << endl;

    //cerr << "campaign " << info.config->campaign << " bidTime "
    //     << 1000.0 * bidTime << endl;

    recordOutcome(1000.0 * bidTime,
                  "accounts.%s.bidResponseTimeMs",
                  config.account.toString('.'));

    doProfileEvent(9, "postTiming");

    if (auctionInfo.bidders.empty()) {
        debugAuction(auctionId, "FINISH", message);
        if (!auctionInfo.auction->finish()) {
            debugAuction(auctionId, "FINISH TOO LATE", message);
        }
        inFlight.erase(auctionId);
        //cerr << "couldn't finish auction " << auctionInfo.auction->id
        //<< " after bid " << message << endl;
    }

    doProfileEvent(10, "finishAuction");

    // TODO: clean up if no bids were made?
#if 0
    // Bids must be the same shape as the bid info or empty
    if (bidInfo.imp.size() != bids.size() && bids.size() != 0) {
        ++info.stats->bidErrors;
        returnInvalidBid(-1, "wrongBidResponseShape",
                         "number of imp in bid request doesn't match "
                         "those in bid: %d vs %d",
                         bidInfo.imp.size(), bids.size(),
                         bidInfo.imp.toJson().toString().c_str(),
                         biddata.c_str());

        if (auctionInfo.bidders.empty()) {
            auctionInfo.auction->finish();
            inFlight.erase(auctionId);
        }
    }
#endif
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
                break;
            case Auction::WinLoss::TOOLATE:
                bidStatus = BS_TOOLATE;
                ++info.stats->tooLate;
                msg = "TOOLATE";
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

            string confidence = "guaranteed";

            //cerr << fName << "sending agent message of type " << msg << endl;
            sendBidResponse(response.agent, info, bidStatus,
                            this->getCurrentTime(),
                            confidence, auctionId,
                            0, Amount(),
                            auction.get(),
                            response.bidData,
                            response.meta,
                            auction->agentAugmentations[response.agent]);
        }

        // If we didn't actually submit a bid then nothing else to do
        if (!hasSubmittedBid) continue;

        ML::atomic_add(numAuctionsWithBid, 1);
        //cerr << fName << "injecting submitted auction " << endl;

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
    if (!monitorClient.getStatus()) {
        Date now = Date::now();

        if ((uint32_t) slowModeLastAuction.secondsSinceEpoch()
            < (uint32_t) now.secondsSinceEpoch()) {
            slowModeLastAuction = now;
            slowModeCount = 1;
            recordHit("monitor.systemInSlowMode");
        }
        else {
            slowModeCount++;
        }

        if (slowModeCount > 100) {
            /* we only let the first 100 auctions take place each second */
            recordHit("monitor.ignoredAuctions");
            auction->finish();
            return;
        }
    }

    //cerr << "AUCTION GOT THROUGH" << endl;

    if (logAuctions)
        // Send AUCTION to logger
        logMessage("AUCTION", auction->id, auction->requestStr);

    const BidRequest & request = *auction->request;
    int numFields = 0;
    if (!request.url.empty()) ++numFields;
    if (request.userIds.exchangeId) ++numFields;
    if (request.userIds.providerId) ++numFields;

    if (numFields > 1) {
        logMessageNoTimestamp("BEHAVIOUR",
                              ML::format("%.2f", request.timestamp),
                              request.exchange,
                              reduceUrl(request.url),
                              request.userIds.exchangeId,
                              request.userIds.providerId);
    }
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
    //const string fName = "Router::doConfig:";
    logMessage("CONFIG", agent, boost::trim_copy(config->toJson().toString()));

    // TODO: no need for this...
    auto newConfig = std::make_shared<AgentConfig>(*config);
    if (newConfig->roundRobinGroup == "")
        newConfig->roundRobinGroup = agent;

    AgentInfo & info = agents[agent];

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
    sendAgentMessage(agent, "GOTCONFIG", getCurrentTime());

    info.filterIndex = filters.addConfig(agent, info);

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
        return;
    }

    int numCompatibleCreatives = 0;

    for(auto & c : config.creatives) {
        auto ccomp = exchange->getCreativeCompatibility(c, includeReasons);
        if(!ccomp.isCompatible) {
            cerr << "creative not compatible: " << ccomp.reasons << endl;
        }
        else {
            std::lock_guard<ML::Spinlock> guard(c.lock);
            c.providerData[name] = ccomp.info;
            ++numCompatibleCreatives;
        }
    }

    if (numCompatibleCreatives == 0) {
        cerr << "no compatible creatives" << endl;
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
            if (exc)
                logException(exc, "Banker addAccount");
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

        // 1.  Send out new pings
        Date now = Date::now();
        if (info.sendPing(0, now))
            sendAgentMessage(agent, "PING0", now, "null");
        if (info.sendPing(1, now))
            sendAgentMessage(agent, "PING1", now, "null");

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
sendBidResponse(const std::string & agent,
                const AgentInfo & info,
                BidStatus status,
                Date timestamp,
                const std::string & message,
                const Id & auctionId,
                int spotNum,
                Amount price,
                const Auction * auction,
                const std::string & bidData,
                const Json::Value & metadata,
                const std::string & augmentationsStr)
{
    BidResultFormat format;
    switch (status) {
    case BS_WIN:   format = info.config->winFormat;   break;
    case BS_LOSS:  format = info.config->lossFormat;  break;
    default:       format = info.config->errorFormat;  break;
    }

    const char * statusStr = bidStatusToChar(status);

    switch (format) {
    case BRF_FULL:
        sendAgentMessage(agent, statusStr, timestamp, message, auctionId,
                         to_string(spotNum),
                         price.toString(),
                         (auction ? info.getBidRequestEncoding(*auction) : ""),
                         (auction ? info.encodeBidRequest(*auction) : ""),
                         bidData, metadata, augmentationsStr);
        break;

    case BRF_LIGHTWEIGHT:
        sendAgentMessage(agent, statusStr, timestamp, message, auctionId,
                         to_string(spotNum), price.toString());
        break;

    case BRF_NONE:
        break;
    }
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

    SubmittedAuctionEvent event;
    event.auctionId = auction->id;
    event.adSpotId = adSpotId;
    event.lossTimeout = auction->lossAssumed;
    event.augmentations = auction->agentAugmentations[bid.agent];
    event.bidRequest = auction->request;
    event.bidRequestStr = auction->requestStr;
    event.bidRequestStrFormat = auction->requestStrFormat ;
    event.bidResponse = bid;

    Message<SubmittedAuctionEvent> message(std::move(event));
    postAuctionEndpoint.sendMessage("AUCTION", message.toString());

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
debugAuctionImpl(const Id & auction, const std::string & type,
                 const std::vector<std::string> & args)
{
    Date now = Date::now();
    boost::unique_lock<ML::Spinlock> guard(debugLock);
    AuctionDebugInfo & entry
        = debugInfo.access(auction, now.plusSeconds(30.0));

    entry.addAuctionEvent(now, type, args);
}

void
Router::
debugSpotImpl(const Id & auction, const Id & spot, const std::string & type,
              const std::vector<std::string> & args)
{
    Date now = Date::now();
    boost::unique_lock<ML::Spinlock> guard(debugLock);
    AuctionDebugInfo & entry
        = debugInfo.access(auction, now.plusSeconds(30.0));

    entry.addSpotEvent(spot, now, type, args);
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
dumpAuction(const Id & auction) const
{
    boost::unique_lock<ML::Spinlock> guard(debugLock);
    auto it = debugInfo.find(auction);
    if (it == debugInfo.end()) {
        //cerr << "*** unknown auction " << auction << " in "
        //     << debugInfo.size() << endl;
    }
    else it->second.dumpAuction();
}

void
Router::
dumpSpot(const Id & auction, const Id & spot) const
{
    boost::unique_lock<ML::Spinlock> guard(debugLock);
    auto it = debugInfo.find(auction);
    if (it == debugInfo.end()) {
        //cerr << "*** unknown auction " << auction << " in "
        //     << debugInfo.size() << endl;
    }
    else it->second.dumpSpot(spot);
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
    bool connectedToPal = postAuctionEndpoint.isConnected();

    MonitorIndicator ind;

    ind.serviceName = serviceName();
    ind.status = connectedToPal;
    ind.message = string()
        + "Connection to PAL: " + (connectedToPal ? "OK" : "ERROR");

    return ind;
}

void
Router::
startExchange(const std::string & type,
              const Json::Value & config)
{
    auto exchange = ExchangeConnector::create(type, *this, type);
    exchange->configure(config);
    exchange->start();

    std::shared_ptr<ExchangeConnector> item(exchange.release());
    addExchange(item);

    exchangeBuffer.push(item);
}

void
Router::
startExchange(const Json::Value & exchangeConfig)
{
    std::string exchangeType = exchangeConfig["exchangeType"].asString();
    startExchange(exchangeType, exchangeConfig);
}



} // namespace RTBKIT

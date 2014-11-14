/** post_auction_service.cc                                 -*- C++ -*-
    Rémi Attab, 18 Apr 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    Implementation of the post auction service.

*/

#include "post_auction_service.h"
#include "simple_event_matcher.h"
#include "sharded_event_matcher.h"
#include "rtbkit/common/messages.h"

using namespace std;
using namespace Datacratic;
using namespace ML;

namespace RTBKIT {


/******************************************************************************/
/* POST AUCTION SERVICE                                                       */
/******************************************************************************/

Logging::Category PostAuctionService::print("PostAuctionService");
Logging::Category PostAuctionService::error("PostAuctionService Error", PostAuctionService::print);
Logging::Category PostAuctionService::trace("PostAuctionService Trace", PostAuctionService::print);


PostAuctionService::
PostAuctionService(
        std::shared_ptr<ServiceProxies> proxies, const std::string & serviceName)
    : ServiceBase(serviceName, proxies),

      auctionTimeout(EventMatcher::DefaultAuctionTimeout),
      winTimeout(EventMatcher::DefaultWinTimeout),
      winLossPipeTimeout(DefaultWinLossPipeTimeout),
      campaignEventPipeTimeout(DefaultCampaignEventPipeTimeout),

      loopMonitor(*this),
      configListener(getZmqContext()),
      monitorProviderClient(getZmqContext()),

      auctions(65536),
      events(65536),

      logger(getZmqContext()),
      endpoint(getZmqContext()),
      bridge(getZmqContext()),
      router(!!getZmqContext())
{
    monitorProviderClient.addProvider(this);
}

PostAuctionService::
PostAuctionService(ServiceBase & parent, const std::string & serviceName)
    : ServiceBase(serviceName, parent),

      auctionTimeout(EventMatcher::DefaultAuctionTimeout),
      winTimeout(EventMatcher::DefaultWinTimeout),

      loopMonitor(*this),
      configListener(getZmqContext()),
      monitorProviderClient(getZmqContext()),

      auctions(65536),
      events(65536),

      logger(getZmqContext()),
      endpoint(getZmqContext()),
      bridge(getZmqContext()),
      router(!!getZmqContext())
{
    monitorProviderClient.addProvider(this);
}


void
PostAuctionService::
bindTcp()
{
    logger.bindTcp(getServices()->ports->getRange("logs"));
    endpoint.bindTcp(getServices()->ports->getRange("postAuctionLoop"));
    bridge.agents.bindTcp(getServices()->ports->getRange("postAuctionLoopAgents"));
}

void
PostAuctionService::
init(size_t externalShard, size_t internalShards)
{
    recordHit("up");

    // Loop monitor is purely for monitoring purposes. There's no message we can
    // just drop in the PAL to alleviate the load.
    loopMonitor.init();
    loopMonitor.addMessageLoop("postAuctionLoop", &loop);

    if(!bidder) {
        Json::Value json;
        json["type"] = "agents";
        initBidderInterface(json);
    }

    initMatcher(internalShards);
    initConnections(externalShard);
    monitorProviderClient.init(getServices()->config);
}

void
PostAuctionService::
initBidderInterface(Json::Value const & json)
{
    bidder = BidderInterface::create(serviceName() + ".bidder", getServices(), json);
    bidder->init(&bridge);
}

void
PostAuctionService::
initMatcher(size_t shards)
{
    if (shards <= 1) {
        LOG(print) << "Creating SimpleEventMatcher" << endl;
        matcher.reset(new SimpleEventMatcher(serviceName(), getServices()));
    }

    else {
        LOG(print) << "Creating ShardedEventMatcher with " << shards << " shards"
            << endl;

        ShardedEventMatcher* m;
        matcher.reset(m = new ShardedEventMatcher(serviceName(), getServices()));
        m->init(shards);
        loop.addSource("PostAuctionService::matcher", *m);
    }


    using std::placeholders::_1;

    matcher->onMatchedWinLoss =
        std::bind(&PostAuctionService::doMatchedWinLoss, this, _1);

    matcher->onMatchedCampaignEvent =
        std::bind(&PostAuctionService::doMatchedCampaignEvent, this, _1);

    matcher->onUnmatchedEvent =
        std::bind(&PostAuctionService::doUnmatched, this, _1);

    matcher->onError = std::bind(&PostAuctionService::doError, this, _1);

    matcher->setWinTimeout(winTimeout);
    matcher->setAuctionTimeout(auctionTimeout);
}


void
PostAuctionService::
initConnections(size_t shard)
{
    using std::placeholders::_1;
    using std::placeholders::_2;

    registerShardedServiceProvider(serviceName(), { "rtbPostAuctionService" }, shard);

    LOG(print) << "post auction logger on " << serviceName() + "/logger" << endl;
    logger.init(getServices()->config, serviceName() + "/logger");
    loop.addSource("PostAuctionService::logger", logger);

    auctions.onEvent = std::bind(&PostAuctionService::doAuction, this, _1);
    loop.addSource("PostAuctionService::auctions", auctions);

    events.onEvent = std::bind(&PostAuctionService::doEvent, this,_1);
    loop.addSource("PostAuctionService::events", events);

    // Initialize zeromq endpoints
    endpoint.init(getServices()->config, ZMQ_XREP, serviceName() + "/events");

    router.bind("AUCTION", std::bind(&PostAuctionService::doAuctionMessage, this, _1));
    router.bind("WIN", std::bind(&PostAuctionService::doWinMessage, this, _1));
    router.bind("LOSS", std::bind(&PostAuctionService::doLossMessage, this,_1));
    router.bind("EVENT", std::bind(&PostAuctionService::doCampaignEventMessage, this, _1));
    router.defaultHandler = [=](const std::vector<std::string> & message) {
        LOG(error) << "unroutable message: " << message[0] << std::endl;
    };

    endpoint.messageHandler = std::bind(
            &ZmqMessageRouter::handleMessage, &router, std::placeholders::_1);
    loop.addSource("PostAuctionService::endpoint", endpoint);

    bridge.agents.init(getServices()->config, serviceName() + "/agents");
    bridge.agents.clientMessageHandler = [&] (const std::vector<std::string> & msg)
        {
            // Clients should never send the post auction service anything,
            // but we catch it here just in case
            LOG(print) << "PostAuctionService got agent message " << msg << endl;
        };
    loop.addSource("PostAuctionService::bridge.agents", bridge.agents);

    configListener.init(getServices()->config);
    configListener.onConfigChange =
        std::bind(&PostAuctionService::doConfigChange, this, _1, _2);
    loop.addSource("PostAuctionService::configListener", configListener);

    // Every second we check for expired auctions
    loop.addPeriodic("PostAuctionService::checkExpiredAuctions", 0.1,
            std::bind(&EventMatcher::checkExpiredAuctions, matcher.get()));
}

void
PostAuctionService::
initAnalytics(const string & baseUrl, const int numConnections)
{
    LOG(print) << "analyticsURI: " << baseUrl << endl;
    analytics.init(baseUrl, numConnections);
}

void
PostAuctionService::
start(std::function<void ()> onStop)
{
    loop.start(onStop);
    monitorProviderClient.start();
    loopMonitor.start();
    matcher->start();
    bidder->start();
    analytics.start();
}

void
PostAuctionService::
shutdown()
{
    matcher->shutdown();
    loopMonitor.shutdown();
    loop.shutdown();
    logger.shutdown();
    bridge.shutdown();
    endpoint.shutdown();
    configListener.shutdown();
    monitorProviderClient.shutdown();
    analytics.shutdown();
}



void
PostAuctionService::
doConfigChange(
        const std::string & agent,
        std::shared_ptr<const AgentConfig> config)
{
    if (!config) return;
    if (config->account.empty())
        throw ML::Exception("attempt to add an account with empty values");

    banker->addSpendAccount(config->account, Amount(),
            [=] (std::exception_ptr error, ShadowAccount && acount) {
                if(error) logException(error, "Banker addSpendAccount");
            });
}

void
PostAuctionService::
doAuctionMessage(const std::vector<std::string> & message)
{
    recordHit("messages.AUCTION");
    auto event = std::make_shared<SubmittedAuctionEvent>(
            ML::DB::reconstituteFromString<SubmittedAuctionEvent>(message.at(2)));
    doAuction(std::move(event));
}

void
PostAuctionService::
doWinMessage(const std::vector<std::string> & message)
{
    recordHit("messages.WIN");
    auto event = std::make_shared<PostAuctionEvent>(
            ML::DB::reconstituteFromString<PostAuctionEvent>(message.at(2)));
    doEvent(event);
}

void
PostAuctionService::
doLossMessage(const std::vector<std::string> & message)
{
    recordHit("messages.LOSS");
    auto event = std::make_shared<PostAuctionEvent>(
            ML::DB::reconstituteFromString<PostAuctionEvent>(message.at(2)));
    doEvent(event);
}

void
PostAuctionService::
doCampaignEventMessage(const std::vector<std::string> & message)
{
    auto event = std::make_shared<PostAuctionEvent>(
            ML::DB::reconstituteFromString<PostAuctionEvent>(message.at(2)));
    recordHit("messages.EVENT." + event->label);
    doEvent(event);
}


void
PostAuctionService::
injectSubmittedAuction(
        const Id & auctionId,
        const Id & adSpotId,
        std::shared_ptr<BidRequest> bidRequest,
        const std::string & bidRequestStr,
        const std::string & bidRequestStrFormat,
        const JsonHolder & augmentations,
        const Auction::Response & bidResponse,
        Date lossTimeout)
{
    if (bidRequestStr.size() == 0) {
        throw ML::Exception("invalid bidRequestStr");
    }
    if (bidRequestStrFormat.size() == 0) {
        throw ML::Exception("invalid bidRequestStrFormat");
    }

    auto event = std::make_shared<SubmittedAuctionEvent>();
    event->auctionId = auctionId;
    event->adSpotId = adSpotId;
    event->bidRequest(bidRequest);
    event->bidRequestStr = bidRequestStr;
    event->bidRequestStrFormat = bidRequestStrFormat;
    event->augmentations = augmentations;
    event->bidResponse = bidResponse;
    event->lossTimeout = lossTimeout;

    auctions.push(event);
}
void
PostAuctionService::
injectWin(
        const Id & auctionId,
        const Id & adSpotId,
        Amount winPrice,
        Date timestamp,
        const JsonHolder & winMeta,
        const UserIds & uids,
        const AccountKey & account,
        Date bidTimestamp)
{
    auto event = std::make_shared<PostAuctionEvent>();
    event->type = PAE_WIN;
    event->auctionId = auctionId;
    event->adSpotId = adSpotId;
    event->timestamp = timestamp;
    event->winPrice = winPrice;
    event->metadata = winMeta;
    event->uids = uids;
    event->account = account;
    event->bidTimestamp = bidTimestamp;

    events.push(event);
}

void
PostAuctionService::
injectLoss(const Id & auctionId,
           const Id & adSpotId,
           Date timestamp,
           const JsonHolder & json,
           const AccountKey & account,
           Date bidTimestamp)
{
    if (timestamp == Date())
        timestamp = Date::now();

    auto event = std::make_shared<PostAuctionEvent>();
    event->type = PAE_LOSS;
    event->auctionId = auctionId;
    event->adSpotId = adSpotId;
    event->timestamp = timestamp;
    event->winPrice = Amount();
    event->account = account;
    event->bidTimestamp = bidTimestamp;

    events.push(event);
}

void
PostAuctionService::
injectCampaignEvent(const string & label,
                    const Id & auctionId,
                    const Id & adSpotId,
                    Date timestamp,
                    const JsonHolder & impressionMeta,
                    const UserIds & uids)
{
    auto event = std::make_shared<PostAuctionEvent>();
    event->type = PAE_CAMPAIGN_EVENT;
    event->label = label;
    event->auctionId = auctionId;
    event->adSpotId = adSpotId;
    event->timestamp = timestamp;
    event->metadata = impressionMeta;
    event->uids = uids;

    events.push(event);
}

void
PostAuctionService::
doAuction(std::shared_ptr<SubmittedAuctionEvent> event)
{
    stats.auctions++;
    matcher->doAuction(std::move(event));
}

void
PostAuctionService::
doEvent(std::shared_ptr<PostAuctionEvent> event)
{
    stats.events++;
    matcher->doEvent(std::move(event));
}

void
PostAuctionService::
checkExpiredAuctions()
{
    matcher->checkExpiredAuctions();
}


void
PostAuctionService::
doMatchedWinLoss(std::shared_ptr<MatchedWinLoss> event)
{
    if (event->type == MatchedWinLoss::Win || event->type == MatchedWinLoss::LateWin) {
        lastWinLoss = Date::now();
        stats.matchedWins++;
    }
    else stats.matchedLosses++;

    event->publish(logger);
    event->publish(analytics);

    deliverEvent("bidResult." + event->typeString(), "doWinLossEvent", event->response.account,
        [&](const AgentConfigEntry& entry)
        {
            bidder->sendWinLossMessage(entry.config, *event);
        });
}

void
PostAuctionService::
doMatchedCampaignEvent(std::shared_ptr<MatchedCampaignEvent> event)
{
    stats.matchedCampaignEvents++;

    lastCampaignEvent = Date::now();

    event->publish(logger);
    event->publish(analytics);

    // For the moment, send the message to all of the agents that are
    // bidding on this account
    //
    deliverEvent("delivery." + event->label, "doCampaignEvent", event->account,
        [&](const AgentConfigEntry& entry)
        {
            bidder->sendCampaignEventMessage(entry.config, entry.name, *event);
        });

}

void
PostAuctionService::
deliverEvent(const std::string& label, const std::string& eventType,
             const AccountKey& account,
             std::function<void(const AgentConfigEntry& entry)> onAgent)
{
    bool sent = false;
    auto onMatchingAgent = [&](const AgentConfigEntry& entry)
    {
        if (!entry.config) return;
        onAgent(entry);
        sent = true;
    };

    configListener.forEachAccountAgent(account, onMatchingAgent);
    if (!sent) {
        recordHit("%s.orphaned", label);
        logPAError(ML::format("%s.noListeners%s", eventType, label),
                   "nothing listening for account " + account.toString());
    }
    else {
        recordHit("%s.delivered", label);
    }
}

void
PostAuctionService::
doUnmatched(std::shared_ptr<UnmatchedEvent> event)
{
    stats.unmatchedEvents++;
    event->publish(logger);
    event->publish(analytics);
}

void
PostAuctionService::
doError(std::shared_ptr<PostAuctionErrorEvent> error)
{
    stats.errors++;
    error->publish(logger);
    error->publish(analytics);
}


std::string
PostAuctionService::
getProviderClass() const
{
    return "rtbPostAuctionService";
}

MonitorIndicator
PostAuctionService::
getProviderIndicators()
    const
{
    Json::Value value;

    /* PA health check:
       - last campaign event in the last 10 seconds */
    Date now = Date::now();
    bool winLossOk = now < lastWinLoss.plusSeconds(winLossPipeTimeout);
    bool campaignEventOk = now < lastCampaignEvent.plusSeconds(campaignEventPipeTimeout);
    bool bankerOk = banker->getProviderIndicators().status;

    // Kept around for posterity.
    // Earned the "Best Error Message in RTBKIT" award.
#if 0
    if (!status)  {
      cerr << "--- WRONGNESS DETECTED:"
          << " last event: " << (now - lastCampaignEvent)
          << endl;
    }
#endif

    MonitorIndicator ind;
    ind.serviceName = serviceName();
    ind.status = (winLossOk || campaignEventOk) && bankerOk;
    ind.message = string()
        + "WinLoss pipe: " + (winLossOk ? "OK" : "ERROR") + ", "
        + "CampaignEvent pipe: " + (campaignEventOk ? "OK" : "ERROR") + ", "
        + "Banker: " + (bankerOk ? "OK" : "ERROR");

    return ind;
}


/******************************************************************************/
/* STATS                                                                      */
/******************************************************************************/


PostAuctionService::Stats::
Stats() :
    auctions(0), events(0),
    matchedWins(0), matchedLosses(0), matchedCampaignEvents(0),
    unmatchedEvents(0), errors(0)
{}

PostAuctionService::Stats::
Stats(const Stats& other) :
    auctions(other.auctions),
    events(other.events),

    matchedWins(other.matchedWins),
    matchedLosses(other.matchedLosses),
    matchedCampaignEvents(other.matchedCampaignEvents),
    unmatchedEvents(other.unmatchedEvents),
    errors(other.errors)
{}

auto
PostAuctionService::Stats::
operator=(const Stats& other) -> Stats&
{
    auctions = other.auctions;
    events = other.events;

    matchedWins = other.matchedWins;
    matchedLosses = other.matchedLosses;
    matchedCampaignEvents = other.matchedCampaignEvents;
    unmatchedEvents = other.unmatchedEvents;
    errors = other.errors;

    return *this;
}

auto
PostAuctionService::Stats::
operator-=(const Stats& other) -> Stats&
{
    auctions -= other.auctions;
    events -= other.events;

    matchedWins -= other.matchedWins;
    matchedLosses -= other.matchedLosses;
    matchedCampaignEvents -= other.matchedCampaignEvents;
    unmatchedEvents -= other.unmatchedEvents;
    errors -= other.errors;

    return *this;
}

auto
PostAuctionService::Stats::
operator+=(const Stats& other) -> Stats&
{
    auctions += other.auctions;
    events += other.events;

    matchedWins += other.matchedWins;
    matchedLosses += other.matchedLosses;
    matchedCampaignEvents += other.matchedCampaignEvents;
    unmatchedEvents += other.unmatchedEvents;
    errors += other.errors;

    return *this;
}

} // namepsace RTBKIT

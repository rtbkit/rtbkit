/** post_auction_service.cc                                 -*- C++ -*-
    Rémi Attab, 18 Apr 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    Implementation of the post auction service.

*/

#include "post_auction_service.h"

#include "rtbkit/common/messages.h"

using namespace std;
using namespace Datacratic;
using namespace ML;

namespace RTBKIT {


/******************************************************************************/
/* POST AUCTION SERVICE                                                       */
/******************************************************************************/

PostAuctionService::
PostAuctionService(
        std::shared_ptr<ServiceProxies> proxies, const std::string & serviceName)
    : ServiceBase(serviceName, proxies),

      auctionTimeout(15),
      winTimeout(1 * 60 * 60),

      loopMonitor(*this),
      matcher(serviceName, proxies),
      configListener(getZmqContext()),
      monitorProviderClient(getZmqContext(), *this),

      auctions(65536),
      events(65536),

      logger(getZmqContext()),
      endpoint(getZmqContext()),
      toAgents(getZmqContext()),
      router(!!getZmqContext())
{}

PostAuctionService::
PostAuctionService(ServiceBase & parent, const std::string & serviceName)
    : ServiceBase(serviceName, parent),

      auctionTimeout(15),
      winTimeout(1 * 60 * 60),

      loopMonitor(*this),
      matcher(serviceName, getServices()),
      configListener(getZmqContext()),
      monitorProviderClient(getZmqContext(), *this),

      auctions(65536),
      events(65536),

      logger(getZmqContext()),
      endpoint(getZmqContext()),
      toAgents(getZmqContext()),
      router(!!getZmqContext())
{}


void
PostAuctionService::
bindTcp()
{
    logger.bindTcp(getServices()->ports->getRange("logs"));
    endpoint.bindTcp(getServices()->ports->getRange("postAuctionLoop"));
    toAgents.bindTcp(getServices()->ports->getRange("postAuctionLoopAgents"));
}

void
PostAuctionService::
init()
{
    initConnections();
    monitorProviderClient.init(getServices()->config);

    using std::placeholders::_1;

    matcher.onMatchedWinLoss =
        std::bind(&PostAuctionService::doMatchedWinLoss, this, _1);

    matcher.onMatchedCampaignEvent =
        std::bind(&PostAuctionService::doMatchedCampaignEvent, this, _1);

    matcher.onUnmatchedEvent =
        std::bind(&PostAuctionService::doUnmatched, this, _1);

    matcher.onError = std::bind(&PostAuctionService::doError, this, _1);
}

void
PostAuctionService::
initConnections()
{
    using std::placeholders::_1;
    using std::placeholders::_2;

    registerServiceProvider(serviceName(), { "rtbPostAuctionService" });

    cerr << "post auction logger on " << serviceName() + "/logger" << endl;
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

    endpoint.messageHandler = std::bind(
            &ZmqMessageRouter::handleMessage, &router, std::placeholders::_1);
    loop.addSource("PostAuctionService::endpoint", endpoint);

    toAgents.init(getServices()->config, serviceName() + "/agents");
    toAgents.clientMessageHandler = [&] (const std::vector<std::string> & msg)
        {
            // Clients should never send the post auction service anything,
            // but we catch it here just in case
            cerr << "PostAuctionService got agent message " << msg << endl;
        };
    loop.addSource("PostAuctionService::toAgents", toAgents);

    configListener.init(getServices()->config);
    configListener.onConfigChange =
        std::bind(&PostAuctionService::doConfigChange, this, _1, _2);
    loop.addSource("PostAuctionService::configListener", configListener);

    // Every second we check for expired auctions
    loop.addPeriodic("PostAuctionService::checkExpiredAuctions", 1.0,
                     std::bind(&EventMatcher::checkExpiredAuctions, &matcher));

    // Loop monitor is purely for monitoring purposes. There's no message we can
    // just drop in the PAL to alleviate the load.
    loopMonitor.init();
    loopMonitor.addMessageLoop("postAuctionLoop", &loop);
}

void
PostAuctionService::
start(std::function<void ()> onStop)
{
    loop.start(onStop);
    monitorProviderClient.start();
    loopMonitor.start();
}

void
PostAuctionService::
shutdown()
{
    loopMonitor.shutdown();
    loop.shutdown();
    logger.shutdown();
    toAgents.shutdown();
    endpoint.shutdown();
    configListener.shutdown();
    monitorProviderClient.shutdown();
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
    auto event = Message<SubmittedAuctionEvent>::fromString(message.at(2));
    if(event) {
        matcher.doAuction(event.payload);
    }
}

void
PostAuctionService::
doWinMessage(const std::vector<std::string> & message)
{
    recordHit("messages.WIN");
    auto event = std::make_shared<PostAuctionEvent>(
            ML::DB::reconstituteFromString<PostAuctionEvent>(message.at(2)));
    matcher.doWinLoss(event, false /* replay */);
}

void
PostAuctionService::
doLossMessage(const std::vector<std::string> & message)
{
    recordHit("messages.LOSS");
    auto event = std::make_shared<PostAuctionEvent>(
            ML::DB::reconstituteFromString<PostAuctionEvent>(message.at(2)));
    matcher.doWinLoss(event, false /* replay */);
}

void
PostAuctionService::
doCampaignEventMessage(const std::vector<std::string> & message)
{
    auto event = std::make_shared<PostAuctionEvent>(
            ML::DB::reconstituteFromString<PostAuctionEvent>(message.at(2)));
    recordHit("messages.EVENT." + event->label);
    matcher.doCampaignEvent(event);
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

    SubmittedAuctionEvent event;
    event.auctionId = auctionId;
    event.adSpotId = adSpotId;
    event.bidRequest = bidRequest;
    event.bidRequestStr = bidRequestStr;
    event.bidRequestStrFormat = bidRequestStrFormat;
    event.augmentations = augmentations;
    event.bidResponse = bidResponse;
    event.lossTimeout = lossTimeout;

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
    //cerr << "injecting loss for " << auctionId << endl;

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
doAuction(const SubmittedAuctionEvent & event)
{
    matcher.doAuction(event);
}

void
PostAuctionService::
doEvent(const std::shared_ptr<PostAuctionEvent> & event)
{
    matcher.doEvent(event);
}

void
PostAuctionService::
doCampaignEvent(const std::shared_ptr<PostAuctionEvent> & event)
{
    matcher.doCampaignEvent(event);
}

void
PostAuctionService::
checkExpiredAuctions()
{
    matcher.checkExpiredAuctions();
}


void
PostAuctionService::
doMatchedWinLoss(MatchedWinLoss event)
{
    lastWinLoss = Date::now();

    event.publish(logger);
    event.sendAgentMessage(toAgents);
}

void
PostAuctionService::
doMatchedCampaignEvent(MatchedCampaignEvent event)
{
    lastCampaignEvent = Date::now();

    event.publish(logger);

    // For the moment, send the message to all of the agents that are
    // bidding on this account

    bool sent = false;
    auto onMatchingAgent = [&] (const AgentConfigEntry & entry)
        {
            if (!entry.config) return;
            event.sendAgentMessage(entry.name, toAgents);
            sent = true;
        };

    const AccountKey & account = event.account;
    configListener.forEachAccountAgent(account, onMatchingAgent);

    if (!sent) {
        recordHit("delivery.%s.orphaned", event.label);
        logPAError("doCampaignEvent.noListeners" + event.label,
                "nothing listening for account " + account.toString());
    }
    else recordHit("delivery.%s.delivered", event.label);
}

void
PostAuctionService::
doUnmatched(UnmatchedEvent event)
{
    event.publish(logger);
}

void
PostAuctionService::
doError(PostAuctionErrorEvent error)
{
    error.publish(logger);
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
    bool winLossOk = now < lastWinLoss.plusSeconds(10);
    bool campaignEventOk = now < lastCampaignEvent.plusSeconds(10);

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
    ind.status = winLossOk || campaignEventOk;
    ind.message = string()
        + "WinLoss pipe: " + (winLossOk ? "OK" : "ERROR") + ", "
        + "CampaignEvent pipe: " + (campaignEventOk ? "OK" : "ERROR");

    return ind;
}

} // namepsace RTBKIT

/** sharded_event_matcher.c                                 -*- C++ -*-
    Rémi Attab, 24 Apr 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    Sharded event matcher implementation.

    The mess of callbacks in this file is due to having to ferry things between
    threads over queues. It's not very pretty but it works.

 */

#include "sharded_event_matcher.h"

using namespace std;
using namespace ML;

namespace RTBKIT {


/******************************************************************************/
/* SHARDED EVENT MATCHER                                                      */
/******************************************************************************/

Logging::Category ShardedEventMatcher::print("ShardedEventMatcher");
Logging::Category ShardedEventMatcher::error("ShardedEventMatcher Error", ShardedEventMatcher::print);
Logging::Category ShardedEventMatcher::trace("ShardedEventMatcher Trace", ShardedEventMatcher::print);


ShardedEventMatcher::
ShardedEventMatcher(std::string prefix, std::shared_ptr<EventService> events) :
    EventMatcher(std::move(prefix), std::move(events)),
    matchedWinLossEvents(1 << 10),
    matchedCampaignEvents(1 << 8),
    unmatchedEvents(1 << 4),
    errorEvents(1 << 4)
{}


ShardedEventMatcher::
ShardedEventMatcher(std::string prefix, std::shared_ptr<ServiceProxies> proxies) :
    EventMatcher(std::move(prefix), std::move(proxies)),
    matchedWinLossEvents(1 << 10),
    matchedCampaignEvents(1 << 8),
    unmatchedEvents(1 << 4),
    errorEvents(1 << 4)
{}

ShardedEventMatcher::Shard::
Shard(std::string prefix, std::shared_ptr<EventService> events) :
    matcher(std::move(prefix), std::move(events)),
    auctions(1 << 8),
    events(1 << 6)
{}

ShardedEventMatcher::Shard::
Shard(std::string prefix, std::shared_ptr<ServiceProxies> proxies) :
    matcher(std::move(prefix), std::move(proxies)),
    auctions(1 << 8),
    events(1 << 6)
{}

void
ShardedEventMatcher::
init(size_t numShards)
{
    if (numShards <= 1)
        THROW(error) << "Invalid number of shards: " << numShards;

    shards.reserve(numShards);

    for (size_t i = 0; i < numShards; ++i) {
        Shard* shard = events_ ?
            new Shard(eventPrefix_, events_) :
            new Shard(eventPrefix_, services_);

        shards.emplace_back(shard);
        shard->init(i, this);
    }

    using std::placeholders::_1;

    matchedWinLossEvents.onEvent =
        std::bind(&ShardedEventMatcher::doMatchedWinLoss, this, _1);
    addSource("SahrdedEventMatcher::matchedWinLossEvents", matchedWinLossEvents);

    matchedCampaignEvents.onEvent =
        std::bind(&ShardedEventMatcher::doMatchedCampaignEvent, this, _1);
    addSource("SahrdedEventMatcher::matchedCampaignEvents", matchedCampaignEvents);

    unmatchedEvents.onEvent =
        std::bind(&ShardedEventMatcher::doUnmatchedEvent, this, _1);
    addSource("SahrdedEventMatcher::unmatchedEvents", unmatchedEvents);

    errorEvents.onEvent = [=] (std::shared_ptr<PostAuctionErrorEvent> event) {
        doError(std::move(event));
    };
    addSource("SahrdedEventMatcher::errorEvents", errorEvents);
}

void
ShardedEventMatcher::Shard::
init(size_t shard, ShardedEventMatcher* parent)
{
    using std::placeholders::_1;

    auctions.onEvent = [=] (std::shared_ptr<SubmittedAuctionEvent> event) {
        parent->recordHit("shards.%d.messages.%s", shard, "AUCTION");

        this->matcher.doAuction(std::move(event));
    };
    addSource("ShardedEventMatcher::Shard::auctions", auctions);

    events.onEvent = [=] (std::shared_ptr<PostAuctionEvent> event) {
        parent->recordHit("shards.%d.messages.%s", shard, RTBKIT::print(event->type));
        if (event->type == PAE_CAMPAIGN_EVENT)
            parent->recordHit("shards.%d.messages.events.%s", shard, event->label);

        this->matcher.doEvent(std::move(event));
    };
    addSource("ShardedEventMatcher::Shard::events", events);

    addPeriodic("ShardedEventMatcher::checkExpiredAuctions", 0.1,
            std::bind(&SimpleEventMatcher::checkExpiredAuctions, &matcher));



    matcher.onMatchedWinLoss = [=] (std::shared_ptr<MatchedWinLoss> event) {
        parent->recordHit("shards.%d.results.MATCHED%s", shard, event->typeString());
        parent->matchedWinLossEvents.push(std::move(event));
    };

    matcher.onMatchedCampaignEvent = [=] (std::shared_ptr<MatchedCampaignEvent> event) {
        parent->recordHit("shards.%d.results.MATCHED%s", shard, event->label);
        parent->matchedCampaignEvents.push(std::move(event));
    };

    matcher.onUnmatchedEvent = [=] (std::shared_ptr<UnmatchedEvent> event) {
        parent->recordHit("shards.%d.results.%s", shard, "UNMATCHED");
        parent->unmatchedEvents.push(std::move(event));
    };

    matcher.onError = [=] (std::shared_ptr<PostAuctionErrorEvent> event) {
        parent->recordHit("shards.%d.results.%s", shard, "ERROR");
        parent->errorEvents.push(std::move(event));
    };
}


void
ShardedEventMatcher::
setBanker(const std::shared_ptr<Banker> & newBanker)
{
    for (auto& shard : shards) shard->matcher.setBanker(newBanker);
}

void
ShardedEventMatcher::
setWinTimeout(float timeout)
{
    for (auto& shard : shards) shard->matcher.setWinTimeout(timeout);
}

void
ShardedEventMatcher::
setAuctionTimeout(float timeout)
{
    for (auto& shard : shards) shard->matcher.setAuctionTimeout(timeout);
}


void
ShardedEventMatcher::
start()
{
    for (size_t i = 0; i < shards.size(); ++i)
        shards[i]->start();
}

void
ShardedEventMatcher::
shutdown()
{
    for (size_t i = 0; i < shards.size(); ++i)
        shards[i]->shutdown();
}


ShardedEventMatcher::Shard&
ShardedEventMatcher::
shard(const Id& auctionId)
{
    return *shards[auctionId.hash() % shards.size()];
}

void
ShardedEventMatcher::
doAuction(std::shared_ptr<SubmittedAuctionEvent> event)
{
    auto& s = shard(event->auctionId);
    s.auctions.push(std::move(event));
}

void
ShardedEventMatcher::
doEvent(std::shared_ptr<PostAuctionEvent> event)
{
    auto& s = shard(event->auctionId);
    s.events.push(std::move(event));
}

} // namepsace RTBKIT

/** events.cc                                 -*- C++ -*-
    Rémi Attab, 18 Apr 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    Publishable event implementation.

*/

#include "events.h"
#include "soa/service/zmq_endpoint.h"
#include "soa/service/zmq_named_pub_sub.h"

using namespace std;
using namespace ML;

namespace RTBKIT {


/******************************************************************************/
/* UTILS                                                                      */
/******************************************************************************/

namespace {

std::string publishTimestamp()
{
    return Date::now().print(5);
}

} // namespace anonymous


/******************************************************************************/
/* MATCHED WIN LOSS                                                           */
/******************************************************************************/

void
MatchedWinLoss::
initFinishedInfo(const FinishedInfo& info)
{
    auctionId = info.auctionId;
    impId = info.adSpotId;
    winPrice = info.winPrice;
    rawWinPrice = info.rawWinPrice;
    response = info.bid;
    requestStr = info.bidRequestStr;
    requestStrFormat = info.bidRequestStrFormat;
    request = info.bidRequest;
    meta = info.winMeta;
    augmentations = info.augmentations;
}

void
MatchedWinLoss::
initMisc(const PostAuctionEvent& event)
{
    uids = event.uids;
    timestamp = event.timestamp;
}

void
MatchedWinLoss::
initMisc(Date timestamp, UserIds uids)
{
    this->timestamp = timestamp;
    this->uids = std::move(uids);
}


MatchedWinLoss::
MatchedWinLoss(
        Type type,
        Confidence confidence,
        const PostAuctionEvent& event,
        const FinishedInfo& info) :
    type(type), confidence(confidence)
{
    initFinishedInfo(info);
    initMisc(event);
}


MatchedWinLoss::
    MatchedWinLoss(
            Type type,
            Confidence confidence,
            const FinishedInfo& info,
            Date timestamp,
            UserIds uids) :
    type(type), confidence(confidence)
{
    initFinishedInfo(info);
    initMisc(timestamp, std::move(uids));
}



std::string
MatchedWinLoss::
typeString() const
{
    switch (type) {
    case LateWin:
    case Win: return "WIN";
    case Loss: return "LOSS";
    }
    ExcAssert(false);
}

std::string
MatchedWinLoss::
confidenceString() const
{
    switch (confidence) {
    case Inferred: return "inferred";
    case Guaranteed: return "guaranteed";
    }

    ExcAssert(false);
}

size_t
MatchedWinLoss::
impIndex() const
{
    return request->findAdSpotIndex(impId);
}


void
MatchedWinLoss::
publish(ZmqNamedPublisher& logger) const
{
    logger.publish(
            "MATCHED" + typeString(),
            publishTimestamp(),

            auctionId.toString(),
            std::to_string(impIndex()),
            request->findAdSpotIndex(impId),
            response.agent,
            response.account.at(1, ""),

            winPrice.toString(),
            response.price.maxPrice.toString(),
            std::to_string(response.price.priority),

            requestStr,
            response.bidData,
            response.meta,

            // This is where things start to get weird.

            std::to_string(response.creativeId),
            response.creativeName,
            response.account.at(0, ""),

            uids.toJsonStr(),
            meta,

            // And this is where we lose all pretenses of sanity.

            response.account.at(0, ""),
            impId.toString(),
            response.account.toString(),

            // Ok back to sanity now.

            requestStrFormat,
            rawWinPrice.toString()
        );
}

void
MatchedWinLoss::
sendAgentMessage(ZmqNamedClientBus& agents) const
{
    std::string channel =
        type == LateWin ? "LATEWIN" : typeString();

    agents.sendMessage(
            response.agent,
            channel,
            timestamp,
            confidenceString(),

            auctionId.toString(),
            std::to_string(impIndex()),
            winPrice.toString(),

            requestStrFormat,
            requestStr,
            response.bidData,
            response.meta,
            augmentations.toJson()
        );
}


/******************************************************************************/
/* MATCHED CAMPAIGN EVENT                                                     */
/******************************************************************************/

MatchedCampaignEvent::
MatchedCampaignEvent(std::string label, const FinishedInfo& info) :
    label(std::move(label)),
    auctionId(info.auctionId),
    impId(info.adSpotId),
    account(info.bid.account),
    requestStr(info.bidRequestStr),
    request(info.bidRequest),
    requestStrFormat(info.bidRequestStrFormat),
    bid(info.bidToJson()),
    win(info.winToJson()),
    campaignEvents(info.campaignEvents.toJson()),
    visits(info.visitsToJson()),
    augmentations(info.augmentations)
{}

size_t
MatchedCampaignEvent::
impIndex() const
{
    return request->findAdSpotIndex(impId);
}

void
MatchedCampaignEvent::
publish(ZmqNamedPublisher& logger) const
{
    logger.publish(
            "MATCHED" + label,
            publishTimestamp(),

            auctionId.toString(),
            impId.toString(),
            requestStr,

            bid,
            win,
            campaignEvents,
            visits,

            account.at(0, ""),
            account.at(1, ""),
            account.toString(),

            requestStrFormat
    );
}

void
MatchedCampaignEvent::
sendAgentMessage(const std::string& agent, ZmqNamedClientBus& endpoint) const
{
    endpoint.sendMessage(
            agent,
            "CAMPAIGN_EVENT",
            label,
            Date::now(),

            auctionId.toString(),
            impId.toString(),
            std::to_string(impIndex()),

            requestStrFormat,
            requestStr,
            augmentations.toJson(),

            bid,
            win,
            campaignEvents,
            visits
        );
}


/******************************************************************************/
/* UNMATCHED EVENT                                                            */
/******************************************************************************/

UnmatchedEvent::
UnmatchedEvent(std::string reason, PostAuctionEvent event) :
    reason(std::move(reason)),
    event(std::move(event))
{}

void
UnmatchedEvent::
publish(ZmqNamedPublisher& logger) const
{
    logger.publish(
            "UNMATCHED" + event.label,
            publishTimestamp(),

            reason,
            event.auctionId.toString(),
            event.adSpotId.toString(),

            std::to_string(event.timestamp.secondsSinceEpoch()),
            event.metadata.toJson()
        );
}


/******************************************************************************/
/* POST AUCTION ERROR EVENT                                                   */
/******************************************************************************/


PostAuctionErrorEvent::
PostAuctionErrorEvent(std::string key, std::string message) :
    key(std::move(key)), message(std::move(message))
{}

void
PostAuctionErrorEvent::
publish(ZmqNamedPublisher& logger) const
{
    logger.publish("PAERROR", publishTimestamp(), key, message);
}


} // namepsace RTBKIT

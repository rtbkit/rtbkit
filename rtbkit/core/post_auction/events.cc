/** events.cc                                 -*- C++ -*-
    Rémi Attab, 18 Apr 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    Publishable event implementation.

*/

#include "events.h"
#include "soa/service/zmq_endpoint.h"
#include "soa/service/zmq_named_pub_sub.h"
#include "rtbkit/common/analytics.h"

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
    impIndex = info.spotIndex;
    winPrice = info.winPrice;
    rawWinPrice = info.rawWinPrice;
    response = info.bid;
    requestStr = info.bidRequestStr;
    requestStrFormat = info.bidRequestStrFormat;
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

void
MatchedWinLoss::
publish(Analytics& logger) const
{
    logger.logMatchedWinLoss(*this);
}

void
MatchedWinLoss::
publish(AnalyticsPublisher & logger) const
{
    logger.publish(
            "MATCHED" + typeString(),
            publishTimestamp(),
            auctionId.toString(),
            response.account.toString(),
            winPrice.toString(),
            rawWinPrice.toString(),
            uids.toJsonStr()
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
    impIndex(info.spotIndex),
    account(info.bid.account),
    requestStr(info.bidRequestStr),
    requestStrFormat(info.bidRequestStrFormat),
    response(info.bid),
    bid(info.bidToJson()),
    win(info.winToJson()),
    campaignEvents(info.campaignEvents.toJson()),
    visits(info.visitsToJson()),
    augmentations(info.augmentations)
{
    auto it = std::find_if(info.campaignEvents.begin(), info.campaignEvents.end(),
                    [&](const CampaignEvent& event) {
                        return event.label_ == this->label;
                    }
                );

    if(it != info.campaignEvents.end())
        timestamp = it->time_;
}
void
MatchedCampaignEvent::
publish(Analytics& logger) const
{
    logger.logMatchedCampaignEvent(*this);
}

void
MatchedCampaignEvent::
publish(AnalyticsPublisher & logger) const
{
    logger.publish(
            "MATCHED" + label,
            publishTimestamp(),
            auctionId.toString(),
            account.toString()
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
publish(Analytics& logger) const
{
    logger.logUnmatchedEvent(*this);
}

void
UnmatchedEvent::
publish(AnalyticsPublisher & logger) const
{
    logger.publish(
            "UNMATCHED" + string(print(event.type)),
            publishTimestamp(),
            reason,
            event.auctionId.toString(),
            std::to_string(event.timestamp.secondsSinceEpoch())
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
publish(Analytics& logger) const
{
    logger.logPostAuctionErrorEvent(*this);
}

void
PostAuctionErrorEvent::
publish(AnalyticsPublisher & logger) const
{
    logger.publish("PAERROR", publishTimestamp(), key, message);
}

} // namepsace RTBKIT

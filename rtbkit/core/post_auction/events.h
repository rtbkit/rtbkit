/** events.h                                 -*- C++ -*-
    Rémi Attab, 18 Apr 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    Publishable events

*/

#pragma once

#include "finished_info.h"
#include "rtbkit/common/auction.h"
#include "rtbkit/common/auction_events.h"
#include "soa/types/id.h"
#include "soa/types/string.h"
#include "rtbkit/common/analytics_publisher.h"


/******************************************************************************/
/* PROTOTYPES                                                                 */
/******************************************************************************/

namespace Datacratic {

struct ZmqNamedPublisher;
struct ZmqNamedClientBus;

} // namespace Datacratic


/******************************************************************************/
/* MATCHED WIN LOSS                                                           */
/******************************************************************************/

namespace RTBKIT {

struct MatchedWinLoss
{
    enum Type { Win, LateWin, Loss };
    Type type;

    enum Confidence { Guaranteed, Inferred };
    Confidence confidence;

    Datacratic::Date timestamp;

    Datacratic::Id auctionId;
    Datacratic::Id impId;
    int impIndex;

    Amount winPrice;    // post-WinCostModel
    Amount rawWinPrice; // pre-WinCostModel

    Auction::Response response;

    Datacratic::UnicodeString requestStr;
    std::string requestStrFormat;

    UserIds uids;
    std::string meta;
    JsonHolder augmentations;

    MatchedWinLoss(
            Type type,
            Confidence confidence,
            const PostAuctionEvent& event,
            const FinishedInfo& info);

    MatchedWinLoss(
            Type type,
            Confidence confidence,
            const FinishedInfo& info,
            Datacratic::Date timestamp,
            UserIds uids);

    std::string typeString() const;
    std::string confidenceString() const;

    void publish(ZmqNamedPublisher& logger) const;
    void publish(AnalyticsPublisher & logger) const;

private:
    void initFinishedInfo(const FinishedInfo& info);
    void initMisc(const PostAuctionEvent& event);
    void initMisc(Datacratic::Date timestamp, UserIds uids);

    void send(Datacratic::ZmqNamedClientBus& agent) const;
    void sendLateWin(Datacratic::ZmqNamedClientBus& agent) const;
};


/******************************************************************************/
/* MATCHED CAMPAIGN EVENT                                                     */
/******************************************************************************/

struct MatchedCampaignEvent
{
    std::string label;
    Datacratic::Id auctionId;
    Datacratic::Id impId;
    int impIndex;
    AccountKey account;

    Datacratic::Date timestamp;

    Datacratic::UnicodeString requestStr;
    std::string requestStrFormat;

    Auction::Response response;

    Json::Value bid;
    Json::Value win;
    Json::Value campaignEvents;
    Json::Value visits;
    JsonHolder augmentations;

    MatchedCampaignEvent(std::string label, const FinishedInfo& info);

    void publish(ZmqNamedPublisher& logger) const;
    void publish(AnalyticsPublisher & logger) const;
};


/******************************************************************************/
/* UNMATCHED                                                                  */
/******************************************************************************/

struct UnmatchedEvent
{
    std::string reason;
    PostAuctionEvent event;

    UnmatchedEvent(std::string reason, PostAuctionEvent event);

    std::string channel() const;
    void publish(ZmqNamedPublisher& logger) const;
    void publish(AnalyticsPublisher & logger) const;
};


/******************************************************************************/
/* ERROR EVENT                                                                */
/******************************************************************************/

struct PostAuctionErrorEvent
{
    std::string key;
    std::string message;
    std::string rest;

    PostAuctionErrorEvent(std::string key, std::string message);
    void publish(ZmqNamedPublisher& logger) const;
    void publish(AnalyticsPublisher & logger) const;
};


} // namespace RTBKIT





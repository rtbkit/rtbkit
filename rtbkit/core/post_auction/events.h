/** events.h                                 -*- C++ -*-
    Rémi Attab, 18 Apr 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    Publishable events

*/

#pragma once

namespace RTBKIT {


/******************************************************************************/
/* MATCHED WIN LOSS                                                           */
/******************************************************************************/

struct MatchedWinLoss
{
    enum Type { Win, LateWin, Loss };
    Type type;

    enum Confidence { Guaranteed, Inferred }:
    Confidence confidence;

    Id auctionId;
    Id impId;

    Amount winPrice;    // post-WinCostModel
    Amount rawWinPrice; // pre-WinCostModel

    Auction::Response response;

    std::string requestStr;
    std::string requestStrFormat;
    std::shared_ptr<BidRequest> request;

    UserIds uids;
    std::string meta;
    Json::Value augmentations;

    MatchedWinLoss(
            Type type,
            Confidence confidence,
            const PostAuctionEvent& event,
            const FinishedInfo& info);

    MatchedWinLoss(
            Type type,
            Confidence confidence,
            const FinishedInfo& info,
            Date timestamp,
            UserIds uids,
            Json::Value augmentations);

    std::string typeString() const;
    std::string confidenceString() const;
    size_t impIndex() const;

    void publish(ZmqNamedPublisher& logger) const;
    void sendAgentMessage(ZmqNamedClientBus& agent) const;

private:
    void initFinishedInfo(const FinishedInfo& info);
    void initMisc(const PostAuctionEvent& event);
    void initMisc(Date timestamp, UserIds uids, Json::value augmentations);

    void send(ZmqNamedClientBus& agent) const;
    void sendLateWin(ZmqNamedClientBus& agent) const;
};


/******************************************************************************/
/* MATCHED CAMPAIGN EVENT                                                     */
/******************************************************************************/

struct MatchedCampaignEvent
{
    std::string label;
    Id auctionId;
    Id impId;

    std::string agent;
    AccountKey account;

    std::string requestStr;
    std::shared_ptr<BidRequest> request;
    std::string requestStrFormat;

    Json::Value bid;
    Json::Value augemntations;
    Json::Value win;
    Json::Value campaignEvents;
    Json::Value visits;

    MatchedCampaignEvent(std::string label, const FinishedInfo& info);

    size_t impIndex() const;

    void publish(ZmqNamedPublisher& logger) const;
    void sendAgentMessage(ZmqNamedClientBus& agent) const;
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
};


} // namespace RTBKIT





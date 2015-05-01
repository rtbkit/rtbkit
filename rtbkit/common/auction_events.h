/* auction_events.h                                                -*- C++ -*-
   Jeremy Barnes, 30 May 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   PAuctionEvent and related classes
*/


#pragma once

#include "jml/db/persistent.h"
#include "soa/types/date.h"
#include "soa/types/id.h"
#include "soa/types/value_description.h"

#include "bids.h"
#include "account_key.h"
#include "auction.h"
#include "bid_request.h"
#include "currency.h"
#include "json_holder.h"
#include "win_cost_model.h"


namespace RTBKIT {

/*****************************************************************************/
/* SUBMITTED AUCTION EVENT                                                   */
/*****************************************************************************/

/** When a submitted bid is transferred from the router to the post auction
    loop, it looks like this.
*/

struct SubmittedAuctionEvent {
    Id auctionId;                  ///< ID of the auction
    Id adSpotId;                   ///< ID of the adspot
    Date lossTimeout;              ///< Time at which a loss is to be assumed
    JsonHolder augmentations;      ///< Augmentations active

    std::shared_ptr<BidRequest> bidRequest() const;
    void bidRequest(std::shared_ptr<BidRequest> event);

    Datacratic::UnicodeString bidRequestStr;     ///< Bid request as string on the wire
    Auction::Response bidResponse; ///< Bid response that was sent
    std::string bidRequestStrFormat;  ///< Format of stringified request(i.e "datacratic")

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);

private:
    mutable std::shared_ptr<BidRequest> bidRequest_;  ///< Bid request
};

CREATE_STRUCTURE_DESCRIPTION(SubmittedAuctionEvent)

/*****************************************************************************/
/* POST AUCTION EVENT TYPE                                                   */
/*****************************************************************************/

enum PostAuctionEventType {
    PAE_INVALID,
    PAE_WIN,
    PAE_LOSS,
    PAE_CAMPAIGN_EVENT
};

const char * print(PostAuctionEventType type);

COMPACT_PERSISTENT_ENUM_DECL(PostAuctionEventType);


/*****************************************************************************/
/* POST AUCTION EVENT                                                        */
/*****************************************************************************/

/** Holds an event that was submitted after an auction.  Needs to be
    possible to serialize/reconstitute as early events that haven't yet
    been matched may need to be saved until they can be matched up.
*/

struct PostAuctionEvent {
    PostAuctionEvent();
    PostAuctionEvent(Json::Value const & json);

    PostAuctionEventType type;
    std::string label; /* for campaign events */
    Datacratic::Id auctionId;
    Datacratic::Id adSpotId;
    Datacratic::Date timestamp;
    JsonHolder metadata;
    AccountKey account;
    Amount winPrice;
    UserIds uids;
    SegmentList channels;
    Date bidTimestamp;

    Json::Value toJson() const;

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);

    std::string print() const;
};

std::ostream &
operator << (std::ostream & stream, const PostAuctionEvent & event);

ML::DB::Store_Writer & operator
    << (ML::DB::Store_Writer & store,
        std::shared_ptr<PostAuctionEvent> event);
ML::DB::Store_Reader & operator
    >> (ML::DB::Store_Reader & store,
        std::shared_ptr<PostAuctionEvent> & event);

CREATE_STRUCTURE_DESCRIPTION(PostAuctionEvent)


/******************************************************************************/
/* CAMPAIGN EVENTS                                                            */
/******************************************************************************/

struct CampaignEvent
{
    CampaignEvent(const std::string & label = "", Date time = Date(),
            const JsonHolder & meta = JsonHolder())
        : label_(label), time_(time), meta_(meta)
    {}

    Json::Value toJson() const;
    static CampaignEvent fromJson(const Json::Value & jsonValue);

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);

    std::string label_;
    Date time_;
    JsonHolder meta_;
};


struct CampaignEvents : public std::vector<CampaignEvent>
{
    bool hasEvent(const std::string & label) const;
    void setEvent(const std::string & label,
                          Date eventTime,
                          const JsonHolder & eventMeta);

    Json::Value toJson() const;
    static CampaignEvents fromJson(const Json::Value&);
};


/******************************************************************************/
/* DELIVERY EVENTS                                                            */
/******************************************************************************/

/**

   \todo Annoyingly similar but cleaner version of PAL's FinishedInfo.
   \todo The toJson functions are only to preserve the old js interface.
*/
struct DeliveryEvent
{
    std::string event;
    Date timestamp;

    Id auctionId;
    Id spotId;
    int spotIndex;

    std::shared_ptr<BidRequest> bidRequest;
    std::string augmentations;


    // \todo Annoyingly similar but cleaner version of Auction::Response
    struct Bid
    {
        Bid() : present(false) {}
        bool present;

        Date time;                    ///< Time at which the bid was placed.

        // Information about the actual bid
        Auction::Price price;     ///< Price to bid on
        AccountKey account;       ///< Account we are bidding with
        bool test;                ///< Is this a test bid?
        WinCostModel wcm;         ///< Win cost model

        // Information about the agent who did the bidding
        std::string agent;        ///< Agent ID who's bidding
        Bids bids;                ///< Original bid
        std::string meta;         ///< Free form agent information about the bid
                                  ///< (Passed back to agent with notification)

        int creativeId;           ///< Number of the creative/placement
        std::string creativeName; ///< Name of the creative

        // Information about the status of the bid (what happened to it)
        Auction::WinLoss localStatus; ///< What happened in the local auction?

        static Bid fromJson(const Json::Value&);
        Json::Value toJson() const;
    } bid;


    /** Contains the metadata suround the wins. */
    struct Win
    {
        Win() : present(false) {}
        bool present;

        Date time;                ///< Time at which win received
        BidStatus reportedStatus; ///< Whether we think we won it or lost it
        Amount price;             ///< Win price post-WinCostModel
        Amount rawPrice;          ///< Win price pre-WinCostModel
        std::string meta;         ///< Metadata from win

        static Win fromJson(const Json::Value&);
        Json::Value toJson() const;
    } win;


    CampaignEvents campaignEvents;

    Json::Value impressionToJson() const;
    Json::Value clickToJson() const;


    struct Visit
    {
        Date time;            ///< Time at which visit received
        SegmentList channels; ///< Channel(s) associated with visit
        std::string meta;     ///< Visit metadata

        static Visit fromJson(const Json::Value&);
        Json::Value toJson() const;
    };

    std::vector<Visit> visits;
    Json::Value visitsToJson() const;

    static DeliveryEvent parse(const std::vector<std::string>&);
};

} // namespace RTBKIT

namespace Datacratic {
template<> struct DefaultDescription<PostAuctionEventType>;
}


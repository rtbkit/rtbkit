/** bids.h                                 -*- C++ -*-
    Rémi Attab, 27 Feb 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Representation of a bid response.

*/

#pragma once

#include "bid_request.h"
#include "currency.h"
#include "account_key.h"

#include "soa/jsoncpp/value.h"
#include "jml/utils/compact_vector.h"

namespace ML { struct Parse_Context; }

namespace RTBKIT {

/*****************************************************************************/
/* BID STATUS                                                                */
/*****************************************************************************/

enum BidStatus {
    BS_WIN,        ///< Bid was won
    BS_LOSS,       ///< Bid was lost
    BS_TOOLATE,    ///< Bid was too late and so not accepted
    BS_INVALID,    ///< Bid was invalid and so not accepted
    BS_LOSTBID,    ///< Bid was lost somewhere
    BS_DROPPEDBID, ///< Bid was dropped as way too late
    BS_NOBUDGET    ///< No budget
};


BidStatus bidStatusFromString(const std::string& str);

const char* bidStatusToChar(BidStatus status);
inline std::string bidStatusToString(BidStatus status)
{
    return std::string(bidStatusToChar(status));
}


/******************************************************************************/
/* BIDS                                                                       */
/******************************************************************************/

/** Holds a bid for a given spot.

    Note that availableCreatives and spotIndex indicates what we're bidding
    on. They will not be serialized because it's only meant to be used to
    inform bidding decisions.
 */
struct Bid
{

    Bid() : creativeIndex(-1), price(), priority(0.0), account() {}


    /** Indexes of the creatives into the agent config's creatives array
        that we can use to bid.
    */
    SmallIntVector availableCreatives;
    int spotIndex; // Index of the spot in the bid request.


    /** Index of the creatives into the agent config's creatives array that we
        want to bid on. Note that it should be one of the indexes in
        info.availableCreatives. Any other indexes will be rejected by the
        router.
     */
    int creativeIndex;
    Amount price;    // The amount we wish to bid. 0 means no bid.
    double priority; // Use as a tie breaker in the router.

    /** Account which placed the bid. This will eventually be required to
        support multiple agents within a single process which is currently a
        work in progress.

        \todo bid() should be changed to accept this string but ideally this
        should be handled in BiddingAgent.
     */
    AccountKey account;

    //extra data come with this bid
    Json::Value ext;

    bool isNullBid() const { return price.isZero() || price.isNegative(); }

    void bid(int creativeIndex, Amount price, double priority = 0.0);

    Json::Value toJson() const;
    static Bid fromJson(ML::Parse_Context&);
};

/** Vector that contains a Bid entry for each imp that are available for
    bidding.

    \todo The interface for this could be vastly improved and shared with
    the router which currently manually parses the json response.
*/
struct Bids : public ML::compact_vector<Bid, 4>
{
    /** Indicates which data sources were used to make the bid decisions.

        Part of an extension to the OpenRTB specs.
     */
    std::set<std::string> dataSources;

    /** Return the bid object associated with the given spot index. */
    Bid& bidForSpot(int spotIndex);

    Json::Value toJson() const;
    std::string toJsonStr() const;
    static Bids fromJson(const std::string& raw);
};


/******************************************************************************/
/* BID RESULT                                                                 */
/******************************************************************************/

/** Message sent to the agents containing the result of one of their bid. See
    BidStatus for the various messages that struct could represent.
 */
struct BidResult
{
    BidStatus result;       ///> Result of our bid
    double timestamp;       ///> Time at which the event occured

    Id auctionId;           ///> Unique auction id for the original bid
    int spotNum;            ///> Spot index into the bidRequest.imp or ourBid
    Amount secondPrice;     ///> Depends on result from router or the exchange
    std::string confidence; ///> ???

    std::shared_ptr<BidRequest> request; ///> Original request we bid on
    Bids ourBid;               ///> Original bids that was placed
    Json::Value metadata;      ///> Metadata that was attached to the bid
    Json::Value augmentations; ///> Original Augmentations sent with the request

    static BidResult parse(const std::vector<std::string>& msg);
};

} // namespace RTBKIT

namespace Datacratic{
using namespace RTBKIT;

template<>
struct DefaultDescription<Bids>
  :public ValueDescriptionI<Bids, ValueKind::ARRAY>{
  virtual void parseJsonTyped(Bids *val,
			      JsonParsingContext &context) const;
  virtual void printJsonTyped(const Bids *val,
			      JsonPrintingContext &context) const;
  virtual bool isDefaultTyped(const Bids *val) const;
};
}

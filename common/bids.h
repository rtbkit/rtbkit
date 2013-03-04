/** bids.h                                 -*- C++ -*-
    Rémi Attab, 27 Feb 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Representation of a bid response.

*/

#pragma once

#include "bid_request.h"
#include "currency.h"

#include "soa/jsoncpp/value.h"
#include "jml/utils/compact_vector.h"


namespace RTBKIT {

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
    Bid() : creativeIndex(-1), price(), priority(0.0) {}


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


    bool isNullBid() const { return price.isZero(); }

    void bid(int creativeIndex, Amount price, double priority = 0.0);

    Json::Value toJson() const;
};

/** Vector that contains a Bid entry for each spots that are available for
    bidding.

    \todo The interface for this could be vastly improved and shared with
    the router which currently manually parses the json response.
*/
struct Bids : public ML::compact_vector<Bid, 4>
{
    Json::Value toJson() const;
    static Bids fromJson(const std::string& raw);
};

} // namespace RTBKIT

/* auction_events.h                                                -*- C++ -*-
   Jeremy Barnes, 30 May 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   PAuctionEvent and related classes
*/


#pragma once

#include "jml/db/persistent.h"
#include "soa/types/date.h"
#include "soa/types/id.h"

#include "account_key.h"
#include "auction.h"
#include "bid_request.h"
#include "currency.h"
#include "json_holder.h"


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
    std::shared_ptr<BidRequest> bidRequest;  ///< Bid request
    std::string bidRequestStr;     ///< Bid request as string on the wire
    Auction::Response bidResponse; ///< Bid response that was sent
    std::string bidRequestStrFormat;  ///< Format of stringified request(i.e "datacratic")

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);
};


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

} // namespace RTBKIT

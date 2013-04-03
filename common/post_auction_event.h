/* post_auction_loop.h                                             -*- C++ -*-
   Jeremy Barnes, 30 May 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   PostAuctionEvent and related classes
*/


#pragma once

#include "jml/db/persistent.h"
#include "soa/types/date.h"
#include "soa/types/id.h"

#include "account_key.h"
#include "bid_request.h"
#include "currency.h"
#include "json_holder.h"


namespace RTBKIT {

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

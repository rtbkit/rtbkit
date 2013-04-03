/* post_auction_loop.h                                             -*- C++ -*-
   Jeremy Barnes, 31 May 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   *AuctionEvent and related classes
*/



#include <ostream>
#include <string>

#include "jml/utils/pair_utils.h"

#include "auction_events.h"

using namespace std;
using namespace ML;
using namespace RTBKIT;


/*****************************************************************************/
/* SUBMITTED AUCTION EVENT                                                   */
/*****************************************************************************/

void
SubmittedAuctionEvent::
serialize(ML::DB::Store_Writer & store) const
{
    store << (unsigned char)0
          << auctionId << adSpotId << lossTimeout << augmentations
          << bidRequestStr << bidResponse << bidRequestStrFormat;
}

void
SubmittedAuctionEvent::
reconstitute(ML::DB::Store_Reader & store)
{
    unsigned char version;
    store >> version;
    if (version != 0)
        throw ML::Exception("unknown SubmittedAuctionEvent type");

    store >> auctionId >> adSpotId >> lossTimeout >> augmentations
          >> bidRequestStr >> bidResponse >> bidRequestStrFormat;

    bidRequest.reset(BidRequest::parse(bidRequestStrFormat, bidRequestStr));
}


/*****************************************************************************/
/* POST AUCTION EVENT TYPE                                                   */
/*****************************************************************************/

const char *
RTBKIT::
print(PostAuctionEventType type)
{
    switch (type) {
    case PAE_INVALID: return "INVALID";
    case PAE_WIN: return "WIN";
    case PAE_LOSS: return "LOSS";
    case PAE_CAMPAIGN_EVENT: return "EVENT";
    default:
        return "UNKNOWN";
    }
}

namespace RTBKIT {
COMPACT_PERSISTENT_ENUM_IMPL(PostAuctionEventType);
}

/*****************************************************************************/
/* POST AUCTION EVENT                                                        */
/*****************************************************************************/

PostAuctionEvent::
PostAuctionEvent()
    : type(PAE_INVALID)
{
}

void
PostAuctionEvent::
serialize(ML::DB::Store_Writer & store) const
{
    unsigned char version = 2;
    store << version << type;
    if (type == PAE_CAMPAIGN_EVENT) {
        store << label;
    }
    store << auctionId << adSpotId << timestamp
          << metadata << account << winPrice
          << uids << channels << bidTimestamp;
}

void
PostAuctionEvent::
reconstitute(ML::DB::Store_Reader & store)
{
    unsigned char version;
    store >> version;
    if (version > 2)
        throw ML::Exception("reconstituting unknown version of "
                            "PostAuctionEvent");
    if (version <= 1) {
        string campaign, strategy;
        store >> type >> auctionId >> adSpotId >> timestamp
              >> metadata >> campaign >> strategy;
        account = { campaign, strategy };
    }
    else {
        store >> type;
        if (type == PAE_CAMPAIGN_EVENT) {
            store >> label;
        }
        store >> auctionId >> adSpotId >> timestamp
              >> metadata >> account;
    }
    if (version == 0) {
        int winCpmInMillis;
        store >> winCpmInMillis;
        winPrice = MicroUSD(winCpmInMillis);
    }
    else store >> winPrice;

    store >> uids >> channels >> bidTimestamp;
}

std::string
PostAuctionEvent::
print() const
{
    std::string result = RTBKIT::print(type);

    auto addVal = [&] (const std::string & val)
        {
            result += '\t' + val;
        };

    if (auctionId) {
        addVal(auctionId.toString());
        addVal(adSpotId.toString());
    }
    addVal(timestamp.print(6));
    if (metadata.isNonNull())
        addVal(metadata.toString());
    if (!account.empty())
        addVal(account.toString());
    if (type == PAE_WIN)
        addVal(winPrice.toString());
    if (!uids.empty())
        addVal(uids.toString());
    if (!channels.empty())
        addVal(channels.toString());
    if (bidTimestamp != Date())
        addVal(bidTimestamp.print(6));

    return result;
}

std::ostream &
RTBKIT::
operator << (std::ostream & stream, const PostAuctionEvent & event)
{
    return stream << event.print();
}

DB::Store_Writer &
RTBKIT::
operator << (DB::Store_Writer & store, shared_ptr<PostAuctionEvent> event)
{
    event->serialize(store);
    return store;
}

DB::Store_Reader &
RTBKIT::
operator >> (DB::Store_Reader & store, shared_ptr<PostAuctionEvent> & event)
{
    event.reset(new PostAuctionEvent());
    event->reconstitute(store);
    return store;
}

/** submission_info.h                                 -*- C++ -*-
    Rémi Attab, 18 Apr 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    Submitted auction message.

*/

#pragma once

#include "rtbkit/common/auction.h"
#include "rtbkit/common/auction_events.h"
#include "soa/types/string.h"

namespace RTBKIT {

/*****************************************************************************/
/* SUBMISSION INFO                                                           */
/*****************************************************************************/

/** Information we track (persistently) about an auction that has been
    submitted and for which we are waiting for information about whether
    it is won or not.
*/

struct SubmissionInfo {
    SubmissionInfo()
        : fromOldRouter(false)
    {
    }

    std::shared_ptr<BidRequest> bidRequest;
    std::string bidRequestStrFormat;

    Datacratic::UnicodeString bidRequestStr() const {
        return Datacratic::UnicodeString(bidRequest->toJsonStr());
    }

    JsonHolder augmentations;
    Auction::Response  bid;               ///< Bid we passed on
    bool fromOldRouter;                   ///< Was reconstituted

    /** If the timeout races with the last bid or the router event loop
        is very busy (as it only processes timeouts when it is idle),
        it is possible that we get a WIN message before we have finished
        the acution.  In this case, we record that message here and replay
        it after the auction has finished.
    */
    std::vector<std::shared_ptr<PostAuctionEvent> > pendingWinEvents;
    std::vector<std::shared_ptr<PostAuctionEvent> > earlyCampaignEvents;
};


} // namespace RTBKIT

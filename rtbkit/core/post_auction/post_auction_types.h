/* post_auction_loop_types.h                                             -*- C++ -*-
   Jeremy Barnes, 30 May 2012
   Router post-auction loop.
*/

#pragma once

#include <unordered_map>
#include "rtbkit/common/auction.h"
#include "rtbkit/common/auction_events.h"

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
    Datacratic::UnicodeString bidRequestStr;
    std::string bidRequestStrFormat;
    JsonHolder augmentations;
    Auction::Response  bid;               ///< Bid we passed on
    bool fromOldRouter;                   ///< Was reconstituted

    /** If the timeout races with the last bid or the router event loop
        is very busy (as it only processes timeouts when it is idle),
        it is possible that we get a WIN message before we have finished
        the acution.  In this case, we record that message here and replay
        it after the auction has finished.
    */
    std::vector<std::shared_ptr<PostAuctionEvent> > earlyWinEvents;
    std::vector<std::shared_ptr<PostAuctionEvent> > earlyCampaignEvents;

    std::string serializeToString() const;
    void reconstituteFromString(const std::string & str);
};


/*****************************************************************************/
/* FINISHED INFO                                                             */
/*****************************************************************************/

/** Information we track (persistently) about an auction that has finished
    (either won or lost).  We keep this around for an hour waiting for
    impressions, clicks or conversions; this structure contains the
    information necessary to join them up.
*/

struct FinishedInfo {
    FinishedInfo()
        : fromOldRouter(false)
    {
    }

    Date auctionTime;            ///< Time at which the auction started
    Id auctionId;       ///< Auction ID from host
    Id adSpotId;          ///< Spot ID from host
    int spotIndex;
    std::shared_ptr<BidRequest> bidRequest;  ///< What we bid on
    Datacratic::UnicodeString bidRequestStr;
    std::string bidRequestStrFormat;
    JsonHolder augmentations;
    std::set<Id> uids;                ///< All UIDs for this user

    /** The set of channels that are associated with this request.  They
        are copied here from the winning agent's configuration so that
        we know how to filter and route the visits.
    */
    SegmentList visitChannels;

    /** Add all of the given UIDs to the set.
    */
    void addUids(const UserIds & toAdd)
    {
        for (auto it = toAdd.begin(), end = toAdd.end();  it != end;  ++it) {
            auto jt = uids.find(it->second);
            if (jt != uids.end())
                return;
            uids.insert(it->second);
        }
    }

    Date bidTime;                ///< Time at which we bid
    Auction::Response bid;       ///< Bid response
    Json::Value bidToJson() const;

    bool hasWin() const { return winTime != Date(); }
    void setWin(Date winTime, BidStatus status, Amount winPrice,
                const std::string & winMeta)
    {
        ExcCheck(!hasWin(), "already has win");

        this->winTime = winTime;
        this->reportedStatus = status;
        this->winPrice = winPrice;
        this->winMeta = winMeta;
    }

    void forceWin(Date winTime, Amount winPrice, const std::string & winMeta)
    {
        ExcCheck(!hasWin() || (reportedStatus == BS_LOSS),
                "only losses can be overriden");

        this->winTime = winTime;
        this->reportedStatus = BS_WIN;
        this->winPrice = winPrice;
        this->winMeta = winMeta;
    }

    Date winTime;                ///< Time at which win received
    BidStatus reportedStatus;    ///< Whether we think we won it or lost it
    Amount winPrice;             ///< Win price
    std::string winMeta;         ///< Metadata from win
    Json::Value winToJson() const;

    CampaignEvents campaignEvents;

    struct Visit {
        Date visitTime;           ///< Time at which visit received
        SegmentList channels;     ///< Channel(s) associated with visit
        std::string meta;         ///< Visit metadata

        Json::Value toJson() const;
        void serialize(ML::DB::Store_Writer & store) const;
        void reconstitute(ML::DB::Store_Reader & store);
    };

    std::vector<Visit> visits;

    /** Add a visit to the visits array. */
    void addVisit(Date visitTime,
                  const std::string & visitMeta,
                  const SegmentList & channels);

    Json::Value visitsToJson() const;

    Json::Value toJson() const;

    bool fromOldRouter;

    std::string serializeToString() const;
    void reconstituteFromString(const std::string & str);
};

} // namespace RTBKIT


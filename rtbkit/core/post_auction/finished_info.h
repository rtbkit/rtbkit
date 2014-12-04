/** finished_info.h                                 -*- C++ -*-
    Rémi Attab, 18 Apr 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    Information related to bids that were associated with a win.

*/

#pragma once

#include "rtbkit/common/auction.h"
#include "rtbkit/common/auction_events.h"
#include "soa/types/string.h"

#include <memory>

namespace RTBKIT {

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

    Date auctionTime;

    Id auctionId;       ///< Auction ID from host
    Id adSpotId;          ///< Spot ID from host
    int spotIndex;
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
    void setWin(
            Date winTime,
            BidStatus status,
            Amount winPrice,
            Amount rawWinPrice,
            const std::string & winMeta)
    {
        ExcCheck(!hasWin(), "already has win");

        this->winTime = winTime;
        this->reportedStatus = status;
        this->winPrice = winPrice;
        this->rawWinPrice = rawWinPrice;
        this->winMeta = winMeta;
    }

    void forceWin(
            Date winTime,
            Amount winPrice,
            Amount rawWinPrice,
            const std::string & winMeta)
    {
        ExcCheck(!hasWin() || (reportedStatus == BS_LOSS),
                "only losses can be overriden");

        this->winTime = winTime;
        this->reportedStatus = BS_WIN;
        this->winPrice = winPrice;
        this->rawWinPrice = rawWinPrice;
        this->winMeta = winMeta;
    }

    Date winTime;                ///< Time at which win received
    BidStatus reportedStatus;    ///< Whether we think we won it or lost it
    Amount winPrice;             ///< Win price Post-WinPriceCostModel
    Amount rawWinPrice;          ///< Win price Pre-WinPriceCostModel
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
};


} // namespace RTBKIT

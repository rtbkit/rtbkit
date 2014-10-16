/* simple_event_matcher.h                                 -*- C++ -*-
   Rémi Attab, 18 Apr 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.

   Event matching logic.
*/

#pragma once

#include "timeout_map.h"
#include "event_matcher.h"
#include "finished_info.h"
#include "submission_info.h"
#include "rtbkit/common/auction.h"
// #include "soa/service/pending_list.h"
#include "soa/service/logs.h"

#include <utility>


/******************************************************************************/
/* HASH                                                                       */
/******************************************************************************/

namespace std {

template<>
struct hash< std::pair<Datacratic::Id, Datacratic::Id> >
{
    size_t operator() (const std::pair<Datacratic::Id, Datacratic::Id>&) const;
};

} // namespace std


namespace RTBKIT {

/******************************************************************************/
/* SIMPLE EVENT MATCHER                                                       */
/******************************************************************************/

struct SimpleEventMatcher : public EventMatcher
{
    SimpleEventMatcher(std::string prefix, std::shared_ptr<EventService> events);
    SimpleEventMatcher(std::string prefix, std::shared_ptr<ServiceProxies> proxies);


    /************************************************************************/
    /* EVENT MATCHING                                                       */
    /************************************************************************/

    /** Handle a new auction that came in. */
    virtual void doAuction(std::shared_ptr<SubmittedAuctionEvent> event);

    /** Handle a post-auction event that came in. */
    virtual void doEvent(std::shared_ptr<PostAuctionEvent> event);

    /** Periodic auction expiry. */
    virtual void checkExpiredAuctions();


    /************************************************************************/
    /* PERSISTENCE                                                          */
    /************************************************************************/

    // virtual void initStatePersistence(const std::string & path);

    static Logging::Category print;
    static Logging::Category error;
    static Logging::Category trace;

private:

    void throwException(const std::string & key, const std::string & msg)
        __attribute__((__noreturn__))
    {
        doError(key, msg);
        THROW(error) << msg;
    }

    void doReallyLateWin(const std::shared_ptr<PostAuctionEvent>& event);

    /** We got a win/loss.  Match it up with its bid and pass on to the
        winning bidder.
    */
    void doWinLoss(std::shared_ptr<PostAuctionEvent> event, bool isReplay);

    /** We got an impression or click on the control socket */
    void doCampaignEvent(std::shared_ptr<PostAuctionEvent> event);

    /** Communicate the result of a bid message to an agent. */
    void doBidResult(
            const Id & auctionId,
            const Id & adSpotId,
            const SubmissionInfo & submission,
            Amount price,
            Date timestamp,
            BidStatus status,
            MatchedWinLoss::Confidence,
            const std::string & winLossMeta,
            const UserIds & uids);

    Date expireSubmitted(
            Date start, const std::pair<Id, Id> & key, const SubmissionInfo & info);

    Date expireFinished(const std::pair<Id, Id> & key, const FinishedInfo & info);


    /** List of auctions we're currently tracking as submitted.  Note that an
        auction may be both submitted and in flight (if we had submitted a bid
        from one agent but were waiting on bids for another agent).

        The key is the (auction id, spot id) pair since after submission,
        the result from every auction comes back separately.
    */
    typedef TimeoutMap<std::pair<Id, Id>, SubmissionInfo> Submitted;
    Submitted submitted;

    /** List of auctions we've won and we're waiting for a campaign event
        from, or otherwise we're keeping around in case a duplicate WIN or a
        campaign event message comes through, or otherwise we're looking for a
        late WIN message for.

        We keep this list around for 5 minutes for those that were lost,
        and one hour for those that were won.
    */
    typedef TimeoutMap<std::pair<Id, Id>, FinishedInfo> Finished;
    Finished finished;

    /** Maintains a map of auction id with the most recently seen spot id. Used
        to associate an event that doesn't have a spot id with an entry within
        submitted or finished.

        Note that while we could maintain the list of all spot ids associated
        with an auction id, this doesn't give us anything since we don't know
        which entry is the real entry. So instead we keep an arbitrarily chosen
        entry.
     */
    std::unordered_map<Id, Id> spotIdMap;
};

} // RTBKIT

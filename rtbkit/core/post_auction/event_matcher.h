/* event_matcher.h                                 -*- C++ -*-
   Rémi Attab (remi.attab@gmail.com), 18 Apr 2014
   FreeBSD-style copyright and disclaimer apply

   Event matching logic.
*/

#pragma once

#include "finished_info.h"
#include "submitted_info.h"

namespace RTBKIT {


/******************************************************************************/
/* EVENT MATCHER                                                              */
/******************************************************************************/

struct EventMatcher : public EventRecoder
{

    EventMatcher(
            PostAuctionService& service,
            std::shared_ptr<EventService> events);


    /************************************************************************/
    /* CALLBACKS                                                            */
    /************************************************************************/

    std::function<void(MatchedWinLoss)> onMatchedWinLoss;

    std::function<void(MatchedCampaignEvent)> onMatchedCampaignEvent;

    std::function<void(UnmatchedEvent)> onUnmatchedEvent;

    std::function<void(PostAuctionErrorEvent)> onError;


    void doError(std::string key, std::string message)
    {
        if (!onError) return;
        recordHit("error.%s", key);
        onError(PostAuctionerrorEvent(std::mmove(key), std::move(message)));
    }


    /************************************************************************/
    /* BANKER                                                               */
    /************************************************************************/

    std::shared_ptr<Banker> getBanker() const
    {
        return banker;
    }

    void setBanker(const std::shared_ptr<Banker> & newBanker)
    {
        matcher.setBanker(banker = newBanker);
    }


    /**************************************************************************/
    /* TIMEOUTS                                                               */
    /**************************************************************************/

    void setWinTimeout(const float & timeOut) {

        if (timeOut < 0.0)
            throw ML::Exception("Invalid timeout for Win timeout");

        matcher.setWinTimeout(winTimeout = timeOut);
    }

    void setAuctionTimeout(const float & timeOut) {

        if (timeOut < 0.0)
            throw ML::Exception("Invalid timeout for Win timeout");

        matcher.setWinTimeout(auctionTimeout = timeOut);
    }


    /************************************************************************/
    /* EVENT MATCHING                                                       */
    /************************************************************************/

    /** Handle a new auction that came in. */
    void doAuction(const SubmittedAuctionEvent & event);

    /** Handle a post-auction event that came in. */
    void doEvent(const std::shared_ptr<PostAuctionEvent> & event);

    /** An auction was submitted... record that */
    void doSubmitted(const std::shared_ptr<PostAuctionEvent> & event);

    /** We got an impression or click on the control socket */
    void doCampaignEvent(const std::shared_ptr<PostAuctionEvent> & event);

    /** Periodic auction expiry. */
    void checkExpiredAuctions();


private:

    /** We got a win/loss.  Match it up with its bid and pass on to the
        winning bidder.
    */
    void doWinLoss(const std::shared_ptr<PostAuctionEvent> & event,
                   bool isReplay);

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


    std::shared_ptr<Banker> banker;

    /** List of auctions we're currently tracking as submitted.  Note that an
        auction may be both submitted and in flight (if we had submitted a bid
        from one agent but were waiting on bids for another agent).

        The key is the (auction id, spot id) pair since after submission,
        the result from every auction comes back separately.
    */
    typedef PendingList<std::pair<Id, Id>, SubmissionInfo> Submitted;
    Submitted submitted;

    /** List of auctions we've won and we're waiting for a campaign event
        from, or otherwise we're keeping around in case a duplicate WIN or a
        campaign event message comes through, or otherwise we're looking for a
        late WIN message for.

        We keep this list around for 5 minutes for those that were lost,
        and one hour for those that were won.
    */
    typedef PendingList<std::pair<Id, Id>, FinishedInfo> Finished;
    Finished finished;

    std::shared_ptr<Banker> banker;
    PostAuctionService& service;

    float auctionTimeout;
    float winTimeout;

};

} // RTBKIT

/** event_matcher.h                                 -*- C++ -*-
    Rémi Attab, 24 Apr 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    Event matching interface.

*/

#pragma once

#include "events.h"
#include "submission_info.h"
#include "rtbkit/core/banker/banker.h"
#include "rtbkit/common/auction_events.h"
#include "soa/service/service_base.h"

#include <utility>


namespace RTBKIT {

/******************************************************************************/
/* EVENT MATCHER                                                              */
/******************************************************************************/

struct EventMatcher : public EventRecorder
{

    EventMatcher(std::string prefix, std::shared_ptr<EventService> events) :
        EventRecorder(prefix, std::move(events))
    {}

    EventMatcher(std::string prefix, std::shared_ptr<ServiceProxies> proxies) :
        EventRecorder(prefix, std::move(proxies))
    {}

    virtual void start() {}
    virtual void shutdown() {}


    /************************************************************************/
    /* CALLBACKS                                                            */
    /************************************************************************/

    std::function<void(std::shared_ptr<MatchedWinLoss>)> onMatchedWinLoss;
    std::function<void(std::shared_ptr<MatchedCampaignEvent>)> onMatchedCampaignEvent;
    std::function<void(std::shared_ptr<UnmatchedEvent>)> onUnmatchedEvent;
    std::function<void(std::shared_ptr<PostAuctionErrorEvent>)> onError;


    /************************************************************************/
    /* BANKER                                                               */
    /************************************************************************/

    std::shared_ptr<Banker> getBanker() const
    {
        return banker;
    }

    virtual void setBanker(const std::shared_ptr<Banker> & newBanker)
    {
        banker = newBanker;
    }


    /**************************************************************************/
    /* TIMEOUTS                                                               */
    /**************************************************************************/

    virtual void setWinTimeout(const float & timeOut)
    {
        if (timeOut < 0.0)
            throw ML::Exception("Invalid timeout for Win timeout");

        winTimeout = timeOut;
    }

    virtual void setAuctionTimeout(const float & timeOut)
    {
        if (timeOut < 0.0)
            throw ML::Exception("Invalid timeout for Win timeout");

        auctionTimeout = timeOut;
    }


    /************************************************************************/
    /* EVENT MATCHING                                                       */
    /************************************************************************/

    /** Handle a new auction that came in. */
    virtual void doAuction(std::shared_ptr<SubmittedAuctionEvent> event) = 0;

    /** Handle a post-auction event that came in. */
    virtual void doEvent(std::shared_ptr<PostAuctionEvent> event) = 0;

    /** Periodic auction expiry. */
    virtual void checkExpiredAuctions() = 0;


    /************************************************************************/
    /* PERSISTENCE                                                          */
    /************************************************************************/

    virtual void initStatePersistence(const std::string & path) {}


protected:

    void doMatchedWinLoss(std::shared_ptr<MatchedWinLoss> event)
    {
        if (onMatchedWinLoss) onMatchedWinLoss(std::move(event));
    }

    void doMatchedCampaignEvent(std::shared_ptr<MatchedCampaignEvent> event)
    {
        if (onMatchedCampaignEvent) onMatchedCampaignEvent(std::move(event));
    }

    void doUnmatchedEvent(std::shared_ptr<UnmatchedEvent> event)
    {
        if (onUnmatchedEvent) onUnmatchedEvent(std::move(event));
    }

    void doError(std::shared_ptr<PostAuctionErrorEvent> event)
    {
        if (onError) onError(std::move(event));
    }

    void doError(std::string key, std::string message)
    {
        recordHit("error.%s", key);
        doError(std::make_shared<PostAuctionErrorEvent>(
                        std::move(key), std::move(message)));
    }


    float auctionTimeout;
    float winTimeout;

    std::shared_ptr<Banker> banker;

};


} // namespace RTBKIT

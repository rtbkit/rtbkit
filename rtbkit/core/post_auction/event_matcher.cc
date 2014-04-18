/* event_matcher.cc                                 -*- C++ -*-
   Rémi Attab (remi.attab@gmail.com), 18 Apr 2014
   FreeBSD-style copyright and disclaimer apply

   Event matching implementation.
*/

#pragma once

namespace RTBKIT {


/******************************************************************************/
/* UTILS                                                                      */
/******************************************************************************/

template<typename Value>
bool findAuction(PendingList<pair<Id,Id>, Value> & pending,
                 const Id & auctionId)
{
    auto key = make_pair(auctionId, Id());
    auto key2 = pending.completePrefix(key, IsPrefixPair());
    return key2.first == auctionId;
}

template<typename Value>
bool findAuction(PendingList<pair<Id,Id>, Value> & pending,
                 const Id & auctionId,
                 Id & adSpotId, Value & val)
{
    auto key = make_pair(auctionId, adSpotId);
    if (!adSpotId) {
        auto key2 = pending.completePrefix(key, IsPrefixPair());
        if (key2.first == auctionId) {
            //cerr << "found info for " << make_pair(auctionId, adSpotId)
            //     << " under " << key << endl;
            adSpotId = key2.second;
            key = key2;
        }
        else return false;
    }

    if (!pending.count(key)) return false;
    val = pending.get(key);

    return true;
}


/******************************************************************************/
/* EVENT MATCHER                                                              */
/******************************************************************************/

void
EventMatcher::
checkExpiredAuctions()
{
    Date start = Date::now();

    {
        cerr << " checking " << submitted.size()
             << " submitted auctions for inferred loss" << endl;


        auto onExpiredSubmitted = [&] (const pair<Id, Id> & key,
                                       const SubmissionInfo & info)
            {
                const Id & auctionId = key.first;
                const Id & adSpotId = key.second;

                recordHit("submittedAuctionExpiry");

                if (!info.bidRequest) {
                    recordHit("submittedAuctionExpiryWithoutBid");
                    return Date();
                }

                try {
                    this->doBidResult(auctionId, adSpotId, info, Amount() /* price */,
                                      start /* date */, BS_LOSS, "inferred",
                                      "null", UserIds());
                } catch (const std::exception & exc) {
                    cerr << "error handling expired loss auction: " << exc.what()
                    << endl;
                    this->logPAError("checkExpiredAuctions.loss", exc.what());
                }

                return Date();
            };

        submitted.expire(onExpiredSubmitted, start);
    }

    {
        cerr << " checking " << finished.size()
             << " finished auctions for expiry" << endl;

        auto onExpiredFinished = [&] (const pair<Id, Id> & key,
                                      const FinishedInfo & info)
            {
                recordHit("finishedAuctionExpiry");
                return Date();
            };

        finished.expire(onExpiredFinished);
    }

    banker->logBidEvents(*this);
}


void
EventMatcher::
doAuction(const SubmittedAuctionEvent & event)
{
    try {
        recordHit("processedAuction");

        const Id & auctionId = event.auctionId;

        //cerr << "doAuction for " << auctionId << endl;

        Date lossTimeout = event.lossTimeout;

        // move the auction over to the submitted bid pipeline...
        auto key = make_pair(auctionId, event.adSpotId);

        SubmissionInfo submission;
        vector<std::shared_ptr<PostAuctionEvent> > earlyWinEvents;
        if (submitted.count(key)) {
            submission = submitted.pop(key);
            earlyWinEvents.swap(submission.earlyWinEvents);
            recordHit("auctionAlreadySubmitted");
        }

        submission.bidRequest = std::move(event.bidRequest);
        submission.bidRequestStrFormat = std::move(event.bidRequestStrFormat);
        submission.bidRequestStr = std::move(event.bidRequestStr);
        submission.augmentations = std::move(event.augmentations);
        submission.bid = std::move(event.bidResponse);

        submitted.insert(key, submission, lossTimeout);

        string transId = makeBidId(auctionId, event.adSpotId, event.bidResponse.agent);
        banker->attachBid(event.bidResponse.account,
                          transId,
                          event.bidResponse.price.maxPrice);

#if 0
        //cerr << "submitted " << auctionId << "; now " << submitted.size()
        //     << " auctions submitted" << endl;

        // Add to awaiting result list
        if (!agents.count(submission.bid.agent)) {
            logPAError("doSubmitted.unknownAgentWonAuction",
                       "unknown agent won auction");
            continue;
        }
        agents[submission.bid.agent].awaitingResult
            .insert(make_pair(auctionId, adSpotId));
#endif

        /* Replay any early win/loss events. */
        for (auto it = earlyWinEvents.begin(),
                 end = earlyWinEvents.end();
             it != end;  ++it) {
            recordHit("replayedEarlyWinEvent");
            //cerr << "replaying early win message" << endl;
            doWinLoss(*it, true /* is_replay */);
        }
    } catch (const std::exception & exc) {
        cerr << "doAuction ignored error handling auction: "
             << exc.what() << endl;
    }
}


void
EventMatcher::
doEvent(const std::shared_ptr<PostAuctionEvent> & event)
{
    //cerr << "!!!EventMatcher::doEvent:got post auction event " <<
    //print(event->type) << endl;

    try {
        switch (event->type) {
        case PAE_WIN:
        case PAE_LOSS:
            doWinLoss(event, false);
            break;
        case PAE_CAMPAIGN_EVENT:
            doCampaignEvent(event);
            break;
        default:
            throw Exception("postAuctionLoop.unknownEventType",
                            "unknown event type (%d)", event->type);
        }
    } catch (const std::exception & exc) {
        cerr << "doEvent " << print(event->type) << " threw: "
             << exc.what() << endl;
    }

    //cerr << "finished with event " << print(event->type) << endl;
}

void
EventMatcher::
doWinLoss(const std::shared_ptr<PostAuctionEvent> & event, bool isReplay)
{
    lastWinLoss = Date::now();

#if 0
    static Date dbg_ts;

    if (!dbg_ts.secondsSinceEpoch()) dbg_ts = lastWinLoss;

    if (lastWinLoss > dbg_ts.plusSeconds(0.2)) {
      cerr << "WIN_RECEIVED: " << dbg_ts.printClassic() << endl;
      dbg_ts = Date::now();
    }
#endif

    BidStatus status;
    if (event->type == PAE_WIN) {
        ML::atomic_inc(numWins);
        status = BS_WIN;
        recordHit("processedWin");
    }
    else {
        status = BS_LOSS;
        ML::atomic_inc(numLosses);
        recordHit("processedLoss");
    }

    const char * typeStr = print(event->type);

    if (!isReplay)
        recordHit("bidResult.%s.messagesReceived", typeStr);
    else
        recordHit("bidResult.%s.messagesReplayed", typeStr);

    //cerr << "doWinLoss 1" << endl;

    // cerr << "doWin" << message << endl;
    const Id & auctionId = event->auctionId;
    const Id & adSpotId = event->adSpotId;
    Amount winPrice = event->winPrice;
    Date timestamp = event->timestamp;
    const JsonHolder & meta = event->metadata;
    const UserIds & uids = event->uids;

    Date bidTimestamp = event->bidTimestamp;

    // debugSpot(auctionId, adSpotId, typeStr);

    auto getTimeGapMs = [&] ()
        {
            return 1000.0 * Date::now().secondsSince(bidTimestamp);
        };

    /*cerr << "doWinLoss for " << auctionId << "-" << adSpotId
         << " " << typeStr
         << " submitted.size() = " << submitted.size()
         << endl;
         */
    //cerr << "  key = (" << auctionId << "," << adSpotId << ")" << endl;
    auto key = make_pair(auctionId, adSpotId);
    /* In this case, the auction is finished which means we've already either:
       a) received a WIN message (and this one is a duplicate);
       b) received no WIN message, timed out, and inferred a loss

       Note that an auction is only removed when the last bidder has bid or
       timed out, and so an auction may be both inFlight and submitted or
       finished.
    */
    if (finished.count(key)) {

        //cerr << "doWinLoss in finished" << endl;

        FinishedInfo info = finished.get(key);
        if (info.hasWin() && status == info.reportedStatus) {
            if (winPrice == info.winPrice) {
                recordHit("bidResult.%s.duplicate", typeStr);
                return;
            }
            else {
                recordHit("bidResult.%s.duplicateWithDifferentPrice",
                          typeStr);
                return;
            }
        }
        else recordHit("bidResult.%s.auctionAlreadyFinished",
                       typeStr);
        double timeGapMs = getTimeGapMs();
        recordOutcome(timeGapMs,
                      "bidResult.%s.alreadyFinishedTimeSinceBidSubmittedMs",
                      typeStr);

        if (event->type == PAE_WIN) {
            // Late win with auction still around
            banker->forceWinBid(info.bid.account, winPrice, LineItems());

            info.forceWin(timestamp, winPrice, meta.toString());

            finished.update(key, info);

            logMessage("MATCHEDWIN",
                    info.auctionId,
                    to_string(info.spotIndex),
                    info.bid.agent,
                    info.bid.account.at(1, ""),
                    info.winPrice.toString(),
                    info.bid.price.maxPrice.toString(),
                    to_string(info.bid.price.priority),
                    info.bidRequestStr,
                    info.bid.bidData,
                    info.bid.meta,
                    to_string(info.bid.creativeId),
                    info.bid.creativeName,
                    info.bid.account.at(0, ""),
                    uids.toJsonStr(),
                    info.winMeta,
                    info.bid.account.at(0, ""),
                    info.adSpotId,
                    info.bid.account.toString(),
                    info.bidRequestStrFormat);


            sendAgentMessage(info.bid.agent, "LATEWIN", timestamp,
                     "guaranteed", info.auctionId,
                     to_string(info.bidRequest->findAdSpotIndex(adSpotId)),
                     info.winPrice.toString(),
                     info.bidRequestStrFormat,
                     info.bidRequestStr,
                     info.bid.bidData,
                     info.bid.meta,
                     info.augmentations);

            recordHit("bidResult.%s.winAfterLossAssumed", typeStr);
            recordOutcome(winPrice.value,
                          "bidResult.%s.winAfterLossAssumedAmount.%s",
                          typeStr, winPrice.getCurrencyStr());
        }

        /*
          cerr << "doWinLoss: auction " << key
          << " was in submitted auctions and also in finished auctions"
          << endl;
        */
        return;
    }

    //cerr << "doWinLoss not in finished" << endl;

    double lossTimeout = 15.0;
    /* If the auction wasn't finished, then it should be submitted.  The only
       time this won't happen is:
       a) when the WIN message raced and got in before we noticed the auction
          timeout.  In that case we will find the auction in inFlight and we
          can store that message there.
       b) when we were more than an hour late, which means that the auction
          is completely unknown.
    */
#if 0
    cerr << fName << " number of elements in submitted " << submitted.size() << endl;
    for (auto it = submitted.begin() ; it != submitted.end() ;++it)
        cerr << it->first << endl;
#endif
    if (!submitted.count(key)) {
        double timeGapMs = getTimeGapMs();
        if (timeGapMs < lossTimeout * 1000) {
            recordHit("bidResult.%s.noBidSubmitted", typeStr);
            //cerr << "WIN for active auction: " << meta
            //     << " timeGapMs = " << timeGapMs << endl;

            /* We record the win message here and play it back once we submit
               the auction.
            */
            SubmissionInfo info;
            info.earlyWinEvents.push_back(event);
            submitted.insert(key, info, Date::now().plusSeconds(lossTimeout));

            return;
        }
        else {
            auto & account = event->account;

            cerr << "REALLY REALLY LATE WIN event='" << *event
                 << "' timeGapMs = " << timeGapMs << endl;
            cerr << "message = " << meta << endl;
            cerr << "bidTimestamp = " << bidTimestamp.print(6) << endl;
            cerr << "now = " << Date::now().print(6) << endl;
            cerr << "account = " << account << endl;

            recordHit("bidResult.%s.notInSubmitted", typeStr);
            recordOutcome(timeGapMs,
                          "bidResult.%s.notInSubmittedTimeSinceBidSubmittedMs",
                          typeStr);

            if(!account.empty()) {
                banker->forceWinBid(account, winPrice, LineItems());
            }

            return;
        }
    }
    SubmissionInfo info = submitted.pop(key);
    if (!info.bidRequest) {
        //cerr << "doWinLoss doubled bid request" << endl;

        // We doubled up on a WIN without having got the auction yet
        info.earlyWinEvents.push_back(event);
        submitted.insert(key, info, Date::now().plusSeconds(lossTimeout));
        return;
    }

    recordHit("bidResult.%s.delivered", typeStr);

    //cerr << "event.metadata = " << event->metadata << endl;
    //cerr << "event.winPrice = " << event->winPrice << endl;

    doBidResult(auctionId, adSpotId, info,
                winPrice, timestamp, status,
                status == BS_WIN ? "guaranteed" : "inferred",
                meta.toString(), uids);
    std::for_each(info.earlyCampaignEvents.begin(),
                  info.earlyCampaignEvents.end(),
                  std::bind(&EventMatcher::doCampaignEvent, this,
                            std::placeholders::_1));

    //cerr << "doWinLoss done" << endl;
}


void
EventMatcher::
doCampaignEvent(const std::shared_ptr<PostAuctionEvent> & event)
{
    //RouterProfiler profiler(this, dutyCycleCurrent.nsImpression);
    //static const char* fName = "EventMatcher::doCampaignEvent:";
    const string & label = event->label;
    const Id & auctionId = event->auctionId;
    Id adSpotId = event->adSpotId;
    Date timestamp = event->timestamp;
    const JsonHolder & meta = event->metadata;
    const UserIds & uids = event->uids;

    SubmissionInfo submissionInfo;
    FinishedInfo finishedInfo;

    if (event->type != PAE_CAMPAIGN_EVENT) {
        throw ML::Exception("event type must be PAE_CAMPAIGN_EVENT: "
                            + string(print(event->type)));
    }

    lastCampaignEvent = Date::now();

    recordHit("delivery.EVENT.%s.messagesReceived", label);

    //cerr << fName << typeStr << " " << auctionId << "-" << adSpotId << endl;
    //cerr <<"The number of elements in submitted " << submitted.size() << endl;
    auto recordUnmatched = [&] (const std::string & why)
        {
            this->logMessage(string("UNMATCHED") + label, why,
                             auctionId.toString(), adSpotId.toString(),
                             to_string(timestamp.secondsSinceEpoch()),
                             meta);
        };

    if (findAuction(submitted, auctionId, adSpotId, submissionInfo)) {
        // Record the impression or click in the submission info.  This will
        // then be passed on once the win comes in.
        //
        // TODO: for now we just ignore the event; we should eventually
        // implement what is written above
        //cerr << "auction " << auctionId << "-" << adSpotId
        //     << " in flight but got " << type << endl;
        recordHit("delivery.%s.stillInFlight", label);
        logPAError(string("doCampaignEvent.auctionNotWon") + label,
                   "message for auction that's not won");
        recordUnmatched("inFlight");

        submissionInfo.earlyCampaignEvents.push_back(event);

        submitted.update(make_pair(auctionId, adSpotId), submissionInfo);
        return;
    }
    else if (findAuction(finished, auctionId, adSpotId, finishedInfo)) {
        // Update the info
        if (finishedInfo.campaignEvents.hasEvent(label)) {
            recordHit("delivery.%s.duplicate", label);
            logPAError(string("doCampaignEvent.duplicate")
                       + label, "message duplicated");
            recordUnmatched("duplicate");
            return;
        }

        finishedInfo.campaignEvents.setEvent(label, timestamp, meta);
        ML::atomic_inc(numCampaignEvents);

        recordHit("delivery.%s.account.%s.matched",
                  label,
                  finishedInfo.bid.account.toString().c_str());

        pair<Id, Id> key(auctionId, adSpotId);
        //cerr << "key = " << key << endl;
        if (!key.second)
            throw ML::Exception("updating null entry in finished map");

        // Add in the user IDs to the index so we can route any visits
        // properly
        finishedInfo.addUids(uids);

        finished.update(key, finishedInfo);

        routePostAuctionEvent(label, finishedInfo,
                              SegmentList(), false /* filterChannels */);
    }
    else {
        /* We get here if we got an IMPRESSION or a CLICK before we got
           notification that an auction had been submitted.

           Normally this should happen rarely.  However, in some cases
           (for example a transient failure in the router to post auction
           loop link which is rectified and allows buffered messages to
           be replayed) we may still want to match things up.

           What we should do here is to keep these messages around in a
           buffer (like the early win messages) and replay them when the
           auction event comes in.
        */

        recordHit("delivery.%s.auctionNotFound", label);
        //cerr << "delivery " << typeStr << ": auction "
        //     << auctionId << "-" << adSpotId << " not found"
        //     << endl;
        logPAError(string("doCampaignEvent.auctionNotFound") + label,
                   "auction not found for delivery message");
        recordUnmatched("auctionNotFound");
    }
}


void
EventMatcher::
doBidResult(const Id & auctionId,
            const Id & adSpotId,
            const SubmissionInfo & submission,
            Amount winPrice,
            Date timestamp,
            BidStatus status,
            const std::string & confidence,
            const std::string & winLossMeta,
            const UserIds & uids)
{
    string msg;

    if (status == BS_WIN) msg = "WIN";
    else if (status == BS_LOSS) msg = "LOSS";
    else throwException("doBidResult.nonWinLoss", "submitted non win/loss");

    if (!adSpotId)
        throw ML::Exception("inserting null entry in finished map");

    string agent = submission.bid.agent;

    // Find the adspot ID
    int adspot_num = submission.bidRequest->findAdSpotIndex(adSpotId);

    if (adspot_num == -1) {
        logPAError("doBidResult.adSpotIdNotFound",
                   "adspot ID ", adSpotId, " not found in auction ",
                   submission.bidRequestStr);
    }

    const Auction::Response & response = submission.bid;

    const AccountKey & account = response.account;
    if (account.size() == 0) {
        throw ML::Exception("invalid account key");
    }

    Amount bidPrice = response.price.maxPrice;

    if (winPrice > bidPrice) {
        logPAError("doBidResult.winPriceExceedsBidPrice",
                   ML::format("win price %s exceeds bid price %s",
                              winPrice.toString(),
                                  bidPrice.toString()));
    }

    // Make sure we account for the bid no matter what
    ML::Call_Guard guard
        ([&] ()
         {
             banker->cancelBid(account, makeBidId(auctionId, adSpotId, agent));
         });

    // No bid
    if (bidPrice == Amount() && response.price.priority == 0) {
        throwException("doBidResult.responseadNoBidPrice",
                       "bid response had no bid price");
    }

    Amount price = winPrice;

    if (status == BS_WIN) {
        WinCostModel wcm = response.wcm;
        wcm.data["win"] = winLossMeta;
        Bids bids = Bids::fromJson(response.bidData);
        price = wcm.evaluate(bids.bidForSpot(adspot_num), winPrice);

        recordOutcome(winPrice.value, "accounts.%s.winPrice.%s",
                      account.toString('.'),
                      winPrice.getCurrencyStr());

        recordOutcome(price.value, "accounts.%s.winCostPrice.%s",
                      account.toString('.'),
                      price.getCurrencyStr());

        // This is a real win
        guard.clear();
        banker->winBid(account, makeBidId(auctionId, adSpotId, agent), price,
                       LineItems());

        // local win; send it back
    }
    else if (status == BS_LOSS) {
        // local loss; send it back
    }
    else throwException("doBidResult.nonWinLoss", "submitted non win/loss");

    logMessage("MATCHED" + msg,
               auctionId,
               to_string(adspot_num),
               response.agent,
               account.at(1, ""),
               price.toString(),
               response.price.maxPrice.toString(),
               to_string(response.price.priority),
               submission.bidRequestStr,
               response.bidData,
               response.meta,
               to_string(response.creativeId),
               response.creativeName,
               account.at(0, ""),
               uids.toJsonStr(),
               winLossMeta,
               account.at(0, ""),
               adSpotId,
               account.toString(),
               submission.bidRequestStrFormat,
               winPrice.toString());

    sendAgentMessage(response.agent, msg, timestamp,
                     confidence, auctionId,
                     to_string(adspot_num),
                     price.toString(),
                     submission.bidRequestStrFormat,
                     submission.bidRequestStr,
                     response.bidData,
                     response.meta,
                     submission.augmentations);

    // Finally, place it in the finished queue
    FinishedInfo i;
    i.auctionId = auctionId;
    i.adSpotId = adSpotId;
    i.spotIndex = adspot_num;
    i.bidRequest = submission.bidRequest;
    i.bidRequestStr = submission.bidRequestStr;
    i.bidRequestStrFormat = submission.bidRequestStrFormat ;
    i.bid = response;
    i.reportedStatus = status;
    //i.auctionTime = auction.start;
    i.setWin(timestamp, status, price, winPrice, winLossMeta);

    // Copy the configuration into the finished info so that we can
    // know which visits to route back
    i.visitChannels = response.visitChannels;

    i.addUids(uids);

    double expiryInterval = winTimeout;
    if (status == BS_LOSS)
        expiryInterval = auctionTimeout;

    Date expiryTime = Date::now().plusSeconds(expiryInterval);

    finished.insert(make_pair(auctionId, adSpotId), i, expiryTime);
}
} // RTBKIT

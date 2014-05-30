/* simple_event_matcher.cc                                 -*- C++ -*-
   Rémi Attab, 18 Apr 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.

   Event matching implementation.
*/

#include "events.h"
#include "simple_event_matcher.h"
#include "jml/utils/guard.h"

#include <iostream>

using namespace std;
using namespace Datacratic;
using namespace ML;

namespace RTBKIT {

/******************************************************************************/
/* UTILS                                                                      */
/******************************************************************************/

namespace {

template<typename Value>
bool findAuction(
        TimeoutMap<pair<Id,Id>, Value> & pending,
        const std::unordered_map<Id, Id>& spotIdMap,
        const Id & auctionId, Id & adSpotId, Value & val)
{
    if (!adSpotId) {
        auto it = spotIdMap.find(auctionId);
        if (it == spotIdMap.end()) return false;

        adSpotId = it->second;
    }

    auto key = make_pair(auctionId, adSpotId);
    if (!pending.count(key)) return false;

    val = pending.get(key);
    return true;
}

std::string makeBidId(Id auctionId, Id spotId, const std::string & agent)
{
    return auctionId.toString() + "-" + spotId.toString() + "-" + agent;
}


} // namespace anonymous


/******************************************************************************/
/* EVENT MATCHER                                                              */
/******************************************************************************/

Logging::Category SimpleEventMatcher::print("SimpleEventMatcher");
Logging::Category SimpleEventMatcher::error("SimpleEventMatcher Error", SimpleEventMatcher::print);
Logging::Category SimpleEventMatcher::trace("SimpleEventMatcher Trace", SimpleEventMatcher::print);


SimpleEventMatcher::
SimpleEventMatcher(std::string prefix, std::shared_ptr<EventService> events) :
    EventMatcher(std::move(prefix), std::move(events))
{}

SimpleEventMatcher::
SimpleEventMatcher(std::string prefix, std::shared_ptr<ServiceProxies> proxies) :
    EventMatcher(std::move(prefix), std::move(proxies))
{}


Date
SimpleEventMatcher::
expireSubmitted(Date start, const pair<Id, Id> & key, const SubmissionInfo & info)
{
    const Id & auctionId = key.first;
    const Id & adSpotId = key.second;

    // Just making sure it doesn't leak if doBidResult throws.
    spotIdMap.erase(key.first);

    recordHit("submittedAuctionExpiry");

    if (!info.bidRequest) {
        recordHit("submittedAuctionExpiryWithoutBid");
        return Date();
    }

    try {
        this->doBidResult(
                auctionId,
                adSpotId,
                info,
                Amount() /* price */,
                start /* date */,
                BS_LOSS,
                MatchedWinLoss::Inferred,
                "null",
                UserIds());
    } catch (const std::exception & exc) {
        LOG(print) << "error handling expired loss auction: "
            << exc.what() << endl;
        doError("checkExpiredAuctions.loss", exc.what());
    }

    return Date();
}


Date
SimpleEventMatcher::
expireFinished(const pair<Id, Id> & key, const FinishedInfo & info)
{
    spotIdMap.erase(key.first);

    recordHit("finishedAuctionExpiry");
    return Date();
}

void
SimpleEventMatcher::
checkExpiredAuctions()
{
    Date now = Date::now();

    using std::placeholders::_1;
    using std::placeholders::_2;

    recordLevel(submitted.size(), "submittedSize");
    submitted.expire(
            std::bind(&SimpleEventMatcher::expireSubmitted, this, now, _1, _2),
            now);

    recordLevel(finished.size(), "finishedSize");
    finished.expire(
            std::bind(&SimpleEventMatcher::expireFinished, this, _1, _2),
            now);

    banker->logBidEvents(*this);
}



void
SimpleEventMatcher::
doEvent(std::shared_ptr<PostAuctionEvent> event)
{
    auto type = event->type;

    try {
        switch (type) {
        case PAE_WIN:
        case PAE_LOSS:
            doWinLoss(std::move(event), false);
            break;
        case PAE_CAMPAIGN_EVENT:
            doCampaignEvent(std::move(event));
            break;
        default:
            THROW(error) << "postAuctionLoop.unknownEventType"
                << "unknown event type " << type;
        }
    } catch (const std::exception & exc) {
        LOG(error) << "doEvent " << RTBKIT::print(type)
            << " threw: " << exc.what() << endl;
    }
}


void
SimpleEventMatcher::
doAuction(std::shared_ptr<SubmittedAuctionEvent> event)
{
    try {
        recordHit("processedAuction");

        const Id & auctionId = event->auctionId;

        Date lossTimeout = event->lossTimeout;

        // move the auction over to the submitted bid pipeline...
        auto key = make_pair(auctionId, event->adSpotId);

        SubmissionInfo submission;
        vector<std::shared_ptr<PostAuctionEvent> > earlyWinEvents;
        if (submitted.count(key)) {
            submission = submitted.pop(key);
            spotIdMap.erase(key.first);

            earlyWinEvents.swap(submission.earlyWinEvents);
            recordHit("auctionAlreadySubmitted");
        }

        submission.bidRequest = event->bidRequest();
        submission.bidRequestStrFormat = std::move(event->bidRequestStrFormat);
        submission.bidRequestStr = std::move(event->bidRequestStr);
        submission.augmentations = std::move(event->augmentations);
        submission.bid = std::move(event->bidResponse);

        submitted.emplace(key, submission, lossTimeout);
        spotIdMap[key.first] = key.second;

        string transId =
            makeBidId(auctionId, event->adSpotId, submission.bid.agent);

        banker->attachBid(
                submission.bid.account, transId, submission.bid.price.maxPrice);

        /* Replay any early win/loss events. */
        for (auto it = earlyWinEvents.begin(), end = earlyWinEvents.end();
             it != end;  ++it)
        {
            recordHit("replayedEarlyWinEvent");
            doWinLoss(*it, true /* is_replay */);
        }

    } catch (const std::exception & exc) {
        LOG(error) << "doAuction ignored error handling auction: "
             << exc.what() << endl;
    }
}

void
SimpleEventMatcher::
doWinLoss(std::shared_ptr<PostAuctionEvent> event, bool isReplay)
{
    BidStatus status;
    if (event->type == PAE_WIN) {
        status = BS_WIN;
        recordHit("processedWin");
    }
    else {
        status = BS_LOSS;
        recordHit("processedLoss");
    }

    const char * typeStr = RTBKIT::print(event->type);

    if (!isReplay)
        recordHit("bidResult.%s.messagesReceived", typeStr);
    else
        recordHit("bidResult.%s.messagesReplayed", typeStr);

    const Id & auctionId = event->auctionId;
    const Id & adSpotId = event->adSpotId;
    Amount winPrice = event->winPrice;
    Date timestamp = event->timestamp;
    const JsonHolder & meta = event->metadata;
    const UserIds & uids = event->uids;

    Date bidTimestamp = event->bidTimestamp;

    auto getTimeGapMs = [&] ()
        {
            return 1000.0 * Date::now().secondsSince(bidTimestamp);
        };

    auto key = make_pair(auctionId, adSpotId);

    /* In this case, the auction is finished which means we've already either:
       a) received a WIN message (and this one is a duplicate);
       b) received no WIN message, timed out, and inferred a loss

       Note that an auction is only removed when the last bidder has bid or
       timed out, and so an auction may be both inFlight and submitted or
       finished.
    */
    if (finished.count(key)) {

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

            info.forceWin(timestamp, winPrice, winPrice, meta.toString());

            finished.get(key) = info;

            doMatchedWinLoss(std::make_shared<MatchedWinLoss>(
                            MatchedWinLoss::LateWin,
                            MatchedWinLoss::Guaranteed,
                            *event, info));


            recordHit("bidResult.%s.winAfterLossAssumed", typeStr);
            recordOutcome(winPrice.value,
                          "bidResult.%s.winAfterLossAssumedAmount.%s",
                          typeStr, winPrice.getCurrencyStr());
        }

        return;
    }

    double lossTimeout = 15.0;

    /* If the auction wasn't finished, then it should be submitted.  The only
       time this won't happen is:
       a) when the WIN message raced and got in before we noticed the auction
          timeout.  In that case we will find the auction in inFlight and we
          can store that message there.
       b) when we were more than an hour late, which means that the auction
          is completely unknown.
    */
    if (!submitted.count(key)) {
        double timeGapMs = getTimeGapMs();
        if (timeGapMs < lossTimeout * 1000) {
            recordHit("bidResult.%s.noBidSubmitted", typeStr);

            /* We record the win message here and play it back once we submit
               the auction.
            */
            SubmissionInfo info;
            info.earlyWinEvents.push_back(event);
            submitted.emplace(key, info, Date::now().plusSeconds(lossTimeout));
            spotIdMap[key.first] = key.second;

            return;
        }
        else {
            auto & account = event->account;

            LOG(print) << "REALLY REALLY LATE WIN event='" << *event
                << "' timeGapMs = " << timeGapMs << endl;
            LOG(print) << "message = " << meta << endl;
            LOG(print) << "bidTimestamp = " << bidTimestamp.print(6) << endl;
            LOG(print) << "now = " << Date::now().print(6) << endl;
            LOG(print) << "account = " << account << endl;

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
    spotIdMap.erase(key.first);

    if (!info.bidRequest) {
        // We doubled up on a WIN without having got the auction yet
        info.earlyWinEvents.push_back(event);
        submitted.emplace(key, info, Date::now().plusSeconds(lossTimeout));
        spotIdMap[key.first] = key.second;
        return;
    }

    recordHit("bidResult.%s.delivered", typeStr);

    auto confidence = status == BS_WIN ?
        MatchedWinLoss::Guaranteed : MatchedWinLoss::Inferred;

    doBidResult(auctionId, adSpotId, info,
                winPrice, timestamp, status, confidence,
                meta.toString(), uids);

    using namespace std::placeholders;
    std::for_each(
            info.earlyCampaignEvents.begin(),
            info.earlyCampaignEvents.end(),
            std::bind(&SimpleEventMatcher::doCampaignEvent, this, _1));
}


void
SimpleEventMatcher::
doCampaignEvent(std::shared_ptr<PostAuctionEvent> event)
{
    const string & label = event->label;
    const Id & auctionId = event->auctionId;
    Id adSpotId = event->adSpotId;
    Date timestamp = event->timestamp;
    const JsonHolder & meta = event->metadata;
    const UserIds & uids = event->uids;

    SubmissionInfo submissionInfo;
    FinishedInfo finishedInfo;

    if (event->type != PAE_CAMPAIGN_EVENT) {
        THROW(error) << "event type must be PAE_CAMPAIGN_EVENT: "
            << RTBKIT::print(event->type);
    }

    recordHit("delivery.EVENT.%s.messagesReceived", label);

    auto recordUnmatched = [&] (const std::string & why) {
        doUnmatchedEvent(std::make_shared<UnmatchedEvent>(why, *event));
    };

    if (findAuction(submitted, spotIdMap, auctionId, adSpotId, submissionInfo)) {
        // Record the impression or click in the submission info.  This will
        // then be passed on once the win comes in.
        //
        // TODO: for now we just ignore the event; we should eventually
        // implement what is written above
        recordHit("delivery.%s.stillInFlight", label);
        doError("doCampaignEvent.auctionNotWon" + label,
                "message for auction that's not won");

        recordUnmatched("inFlight");

        submissionInfo.earlyCampaignEvents.push_back(event);
        submitted.get(make_pair(auctionId, adSpotId)) = submissionInfo;
        spotIdMap[auctionId] = adSpotId;
        return;
    }

    else if (findAuction(finished, spotIdMap, auctionId, adSpotId, finishedInfo)) {
        // Update the info
        if (finishedInfo.campaignEvents.hasEvent(label)) {
            recordHit("delivery.%s.duplicate", label);
            doError("doCampaignEvent.duplicate" + label,
                    "message duplicated");
            recordUnmatched("duplicate");
            return;
        }

        finishedInfo.campaignEvents.setEvent(label, timestamp, meta);

        recordHit("delivery.%s.account.%s.matched",
                  label,
                  finishedInfo.bid.account.toString().c_str());

        pair<Id, Id> key(auctionId, adSpotId);
        if (!key.second)
            THROW(error) << "updating null entry in finished map";

        // Add in the user IDs to the index so we can route any visits
        // properly
        finishedInfo.addUids(uids);

        finished.get(key) = finishedInfo;

        doMatchedCampaignEvent(
                std::make_shared<MatchedCampaignEvent>(label, finishedInfo));
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
        doError("doCampaignEvent.auctionNotFound" + label,
                   "auction not found for delivery message");
        recordUnmatched("auctionNotFound");
    }
}


void
SimpleEventMatcher::
doBidResult(
        const Id & auctionId,
        const Id & adSpotId,
        const SubmissionInfo & submission,
        Amount winPrice,
        Date timestamp,
        BidStatus status,
        MatchedWinLoss::Confidence confidence,
        const std::string & winLossMeta,
        const UserIds & uids)
{
    string msg;

    if (status == BS_WIN) msg = "WIN";
    else if (status == BS_LOSS) msg = "LOSS";
    else THROW(error) << "submitted non win/loss";

    if (!adSpotId)
        THROW(error) << "inserting null entry in finished map";

    string agent = submission.bid.agent;

    // Find the adspot ID
    int adspot_num = submission.bidRequest->findAdSpotIndex(adSpotId);
    if (adspot_num == -1) {
        doError("doBidResult.adSpotIdNotFound",
                "adspot ID " + adSpotId.toString() +
                " not found in auction " +
                submission.bidRequestStr.utf8String());
    }

    const Auction::Response & response = submission.bid;

    const AccountKey & account = response.account;
    if (account.size() == 0)
        THROW(error) << "invalid account key";

    Amount bidPrice = response.price.maxPrice;

    if (winPrice > bidPrice) {
        doError("doBidResult.winPriceExceedsBidPrice",
                ML::format("win price %s exceeds bid price %s",
                        winPrice.toString(), bidPrice.toString()));
    }

    // Make sure we account for the bid no matter what
    ML::Call_Guard guard ([&] () {
                auto transId = makeBidId(auctionId, adSpotId, agent);
                banker->cancelBid(account, transId);
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

        auto transId = makeBidId(auctionId, adSpotId, agent);
        banker->winBid(account, transId, price, LineItems());
    }

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
    i.setWin(timestamp, status, price, winPrice, winLossMeta);
    i.addUids(uids);

    // Copy the configuration into the finished info so that we can
    // know which visits to route back
    i.visitChannels = response.visitChannels;

    auto matchedType =
        status == BS_WIN ? MatchedWinLoss::Win : MatchedWinLoss::Loss;
    doMatchedWinLoss(std::make_shared<MatchedWinLoss>(
                    matchedType, confidence, i, timestamp, uids));

    double expiryInterval = winTimeout;
    if (status == BS_LOSS)
        expiryInterval = auctionTimeout;

    Date expiryTime = Date::now().plusSeconds(expiryInterval);
    finished.emplace(make_pair(auctionId, adSpotId), i, expiryTime);
    spotIdMap[auctionId] = adSpotId;
}



/******************************************************************************/
/* PERSISTENCE                                                                */
/******************************************************************************/
// Needs to be properly tested before enabling.

namespace {

std::pair<Id, Id>
unstringifyPair(const std::string & str)
{
    istringstream stream(str);
    DB::Store_Reader store(stream);
    pair<Id, Id> result;
    store >> result.first >> result.second;
    return result;
}

std::string stringifyPair(const std::pair<Id, Id> & vals)
{
    if (!vals.second || vals.second.type == Id::NULLID)
        throw ML::Exception("attempt to store null ID");

    ostringstream stream;
    {
        DB::Store_Writer store(stream);
        store << vals.first << vals.second;
    }

    return stream.str();
}

} // file scope


#if 0 // Persistence layer that needs to be reworked.

void
SimpleEventMatcher::
initStatePersistence(const std::string & path)
{
    typedef PendingPersistenceT<pair<Id, Id>, SubmissionInfo>
        SubmittedPending;

    auto submittedDb = std::make_shared<LeveldbPendingPersistence>();
    submittedDb->open(path + "/submitted");

    auto submittedPersistence
        = std::make_shared<SubmittedPending>();
    submittedPersistence->store = submittedDb;

    auto stringifySubmissionInfo = [] (const SubmissionInfo & info)
        {
            return info.serializeToString();
        };

    auto unstringifySubmissionInfo = [] (const std::string & str)
        {
            SubmissionInfo info;
            info.reconstituteFromString(str);
            return info;
        };

    submittedPersistence->stringifyKey = stringifyPair;
    submittedPersistence->unstringifyKey = unstringifyPair;
    submittedPersistence->stringifyValue = stringifySubmissionInfo;
    submittedPersistence->unstringifyValue = unstringifySubmissionInfo;

    Date newTimeout = Date::now().plusSeconds(15);

    auto acceptSubmitted = [&] (pair<Id, Id> & key,
                                SubmissionInfo & info,
                                Date & timeout) -> bool
        {
            info.fromOldRouter = true;
            newTimeout.addSeconds(0.001);
            timeout = newTimeout;
            // this->debugSpot(key.first, key.second, "RECONST SUBMITTED");
            return true;
        };

    submitted.initFromStore(submittedPersistence,
                            acceptSubmitted,
                            Date::now().plusSeconds(15));

    typedef PendingPersistenceT<pair<Id, Id>, FinishedInfo>
        FinishedPending;

    auto finishedDb = std::make_shared<LeveldbPendingPersistence>();
    finishedDb->open(path + "/finished");

    auto finishedPersistence
        = std::make_shared<FinishedPending>();
    finishedPersistence->store = finishedDb;

    auto stringifyFinishedInfo = [] (const FinishedInfo & info)
        {
            return info.serializeToString();
        };

    auto unstringifyFinishedInfo = [] (const std::string & str)
        {
            FinishedInfo info;
            info.reconstituteFromString(str);
            return info;
        };

    finishedPersistence->stringifyKey = stringifyPair;
    finishedPersistence->unstringifyKey = unstringifyPair;
    finishedPersistence->stringifyValue = stringifyFinishedInfo;
    finishedPersistence->unstringifyValue = unstringifyFinishedInfo;

    newTimeout = Date::now().plusSeconds(auctionTimeout);

    auto acceptFinished = [&] (pair<Id, Id> & key,
                               FinishedInfo & info,
                               Date & timeout) -> bool
        {
            info.fromOldRouter = true;
            newTimeout.addSeconds(0.001);
            timeout = newTimeout;
            // this->debugSpot(key.first, key.second, "RECONST FINISHED");

            return true;
        };

    finished.initFromStore(finishedPersistence,
                           acceptFinished,
                           Date::now().plusSeconds(auctionTimeout));

    auto backgroundWork = [=] (volatile int & shutdown, int64_t threadId)
        {
            while (!shutdown) {
                futex_wait(const_cast<int &>(shutdown), 0, 600.0);
                if (shutdown) break;
                //continue;

                {
                    Date start = Date::now();
                    submittedDb->compact();
                    Date end = Date::now();
                    this->recordEvent("persistentData.submitted.compactTimeMs",
                                  ET_OUTCOME,
                                  1000.0 * (end.secondsSince(start)));
                    uint64_t size = submittedDb->getDbSize();
                    //cerr << "submitted db is " << size / 1024.0 / 1024.0
                    //     << "MB" << endl;
                    this->recordEvent("persistentData.submitted.dbSizeMb",
                                  ET_LEVEL, size / 1024.0 / 1024.0);
                }

                {
                    Date start = Date::now();
                    finishedDb->compact();
                    Date end = Date::now();
                    this->recordEvent("persistentData.finished.compactTimeMs",
                                  ET_OUTCOME,
                                  1000.0 * (end.secondsSince(start)));
                    uint64_t size = finishedDb->getDbSize();
                    //cerr << "finished db is " << size / 1024.0 / 1024.0
                    //     << "MB" << endl;
                    this->recordEvent("persistentData.finished.dbSizeMb",
                                  ET_LEVEL, size / 1024.0 / 1024.0);
                }
            }

            cerr << "exiting background work thread" << endl;
        };

    // loop.startSubordinateThread(backgroundWork);
}

#endif

} // RTBKIT

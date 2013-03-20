/* post_auction_loop.h                                             -*- C++ -*-
   Jeremy Barnes, 31 May 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Loop for post-auction processing.
*/

#include <string>
#include "post_auction_loop.h"
#include <sstream>
#include <iostream>
#include <boost/make_shared.hpp>
#include "rtbkit/core/agent_configuration/agent_config.h"
#include "jml/utils/pair_utils.h"
#include "jml/arch/futex.h"
#include "jml/db/persistent.h"
#include "rtbkit/core/banker/banker.h"
#include "rtbkit/common/messages.h"

using namespace std;
using namespace ML;


namespace RTBKIT {


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

const char * print(PostAuctionEventType type)
{
    switch (type) {
    case PAE_INVALID: return "INVALID";
    case PAE_WIN: return "WIN";
    case PAE_LOSS: return "LOSS";
    case PAE_IMPRESSION: return "IMPRESSION";
    case PAE_CLICK: return "CLICK";
    case PAE_VISIT: return "VISIT";
    default:
        return "UNKNOWN";
    }
}

COMPACT_PERSISTENT_ENUM_IMPL(PostAuctionEventType);


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
    store << version << type << auctionId << adSpotId << timestamp
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
        store >> type >> auctionId >> adSpotId >> timestamp
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
operator << (std::ostream & stream, const PostAuctionEvent & event)
{
    return stream << event.print();
}


/*****************************************************************************/
/* SUBMISSION INFO                                                           */
/*****************************************************************************/

std::string
SubmissionInfo::
serializeToString() const
{
    ostringstream stream;
    ML::DB::Store_Writer writer(stream);
    int version = 5;
    writer << version
           << bidRequestStr
           << bidRequestStrFormat
           << augmentations.toString()
           << earlyWinEvents
           << earlyImpressionClickEvents;
    bid.serialize(writer);
    return stream.str();
}

DB::Store_Writer & operator << (DB::Store_Writer & store,
                                std::shared_ptr<PostAuctionEvent> event)
{
    event->serialize(store);
    return store;
}

DB::Store_Reader & operator >> (DB::Store_Reader & store,
                                std::shared_ptr<PostAuctionEvent> & event)
{
    event.reset(new PostAuctionEvent());
    event->reconstitute(store);
    return store;
}

void
SubmissionInfo::
reconstituteFromString(const std::string & str)
{
    istringstream stream(str);
    ML::DB::Store_Reader store(stream);
    int version;
    store >> version;
    if (version < 1 || version > 5)
        throw ML::Exception("bad version %d", version);
    store >> bidRequestStr;
    if (version == 5)
    {
        store >> bidRequestStrFormat ;
    }
    if (version > 1) {
        string s;
        store >> s;
        augmentations = s;
    }
    else augmentations.clear();
    if (version == 3) {
        vector<vector<string> > msg1, msg2;
        store >> msg1 >> msg2;
        if (!msg1.empty() || !msg2.empty())
            cerr << "warning: discarding early events from old format"
                 << endl;
        earlyWinEvents.clear();
        earlyImpressionClickEvents.clear();
    }
    else if (version > 3) {
        store >> earlyWinEvents >> earlyImpressionClickEvents;
    }
    else {
        earlyWinEvents.clear();
        earlyImpressionClickEvents.clear();
    }
    bid.reconstitute(store);

    if (bidRequestStr != "")
        bidRequest.reset(BidRequest::parse(bidRequestStrFormat, bidRequestStr));
    else bidRequest.reset();
}


/*****************************************************************************/
/* FINISHED INFO                                                             */
/*****************************************************************************/

Json::Value
FinishedInfo::
bidToJson() const
{
    Json::Value result = bid.toJson();
    result["timestamp"] = bidTime.secondsSinceEpoch();
    return result;
}

Json::Value
FinishedInfo::
winToJson() const
{
    Json::Value result;
    if (!hasWin()) return result;

    result["timestamp"] = winTime.secondsSinceEpoch();
    result["reportedStatus"] = (reportedStatus == BS_WIN ? "WIN" : "LOSS");
    result["winPrice"] = winPrice.toJson();
    result["meta"] = winMeta;

    return result;
}

Json::Value
FinishedInfo::
impressionToJson() const
{
    Json::Value result;
    if (!hasImpression()) return result;

    result["timestamp"] = impressionTime.secondsSinceEpoch();
    result["meta"] = impressionMeta;

    return result;
}

Json::Value
FinishedInfo::
clickToJson() const
{
    Json::Value result;
    if (!hasClick()) return result;

    result["timestamp"] = clickTime.secondsSinceEpoch();
    result["meta"] = clickMeta;

    return result;
}

void
FinishedInfo::
addVisit(Date visitTime,
         const std::string & visitMeta,
         const SegmentList & channels)
{
    Visit visit;
    visit.visitTime = visitTime;
    visit.channels = channels;
    visit.meta = visitMeta;
    visits.push_back(visit);
}

Json::Value
FinishedInfo::
visitsToJson() const
{
    Json::Value result;
    for (unsigned i = 0;  i < visits.size();  ++i) {
        Json::Value & v = result[i];
        const Visit & visit = visits[i];
        v["timestamp"] = visit.visitTime.secondsSinceEpoch();
        v["meta"] = visit.meta;
        v["channels"] = visit.channels.toJson();
    }
    return result;
}

Json::Value
FinishedInfo::
toJson() const
{
    throw ML::Exception("FinishedInfo::toJson()");
    Json::Value result;
    return result;
}

void
FinishedInfo::Visit::
serialize(DB::Store_Writer & store) const
{
    unsigned char version = 1;
    store << version << visitTime << channels << meta;
}

void
FinishedInfo::Visit::
reconstitute(DB::Store_Reader & store)
{
    unsigned char version;
    store >> version;
    if (version != 1)
        throw ML::Exception("invalid version");
    store >> visitTime >> channels >> meta;
}

IMPL_SERIALIZE_RECONSTITUTE(FinishedInfo::Visit);

std::string
FinishedInfo::
serializeToString() const
{
    ostringstream stream;
    ML::DB::Store_Writer writer(stream);
    int version = 5;
    writer << version
           << auctionTime << auctionId << adSpotId
           << bidRequestStr << bidTime <<bidRequestStrFormat;
    bid.serialize(writer);
    writer << winTime
           << reportedStatus << winPrice << winMeta
           << impressionTime << impressionMeta
           << clickTime << clickMeta << fromOldRouter
           << augmentations.toString();
    writer << visitChannels << uids << visits;

    return stream.str();
}

void
FinishedInfo::
reconstituteFromString(const std::string & str)
{
    istringstream stream(str);
    ML::DB::Store_Reader store(stream);
    int version, istatus;
    store >> version;
    if (version > 5)
        throw ML::Exception("bad version %d", version);

    string auctionIdStr, adSpotIdStr;

    store >> auctionTime >> auctionId >> adSpotId
          >> bidRequestStr >> bidTime;
    if(version == 5)
        store >> bidRequestStrFormat;
    bid.reconstitute(store);

    store >> winTime >> istatus;
    if (version == 3) {
        int winPriceMicros;
        store >> winPriceMicros;
        winPrice = MicroUSD(winPriceMicros);
    }
    else store >> winPrice;
    store >> winMeta
          >> impressionTime >> impressionMeta
          >> clickTime >> clickMeta >> fromOldRouter;

    if (version > 1) {
        string s;
        store >> s;
        augmentations = s;
    }
    else augmentations.clear();

    if (version > 2) {
        store >> visitChannels >> uids >> visits;
    }

    reportedStatus = (BidStatus)istatus;

    bidRequest.reset(BidRequest::parse(bidRequestStrFormat, bidRequestStr));
}


/*****************************************************************************/
/* POST AUCTION LOOP                                                         */
/*****************************************************************************/

PostAuctionLoop::
PostAuctionLoop(std::shared_ptr<ServiceProxies> proxies,
                const std::string & serviceName)
    : ServiceBase(serviceName, proxies),
      logger(getZmqContext()),
      monitorProviderClient(getZmqContext(), *this),
      auctions(65536),
      events(65536),
      endpoint(getZmqContext()),
      router(!!getZmqContext()),
      toAgents(getZmqContext()),
      configListener(getZmqContext())
{
}

PostAuctionLoop::
PostAuctionLoop(ServiceBase & parent,
                const std::string & serviceName)
    : ServiceBase(serviceName, parent),
      logger(getZmqContext()),
      monitorProviderClient(getZmqContext(), *this),
      auctions(16386),
      events(1024),
      endpoint(getZmqContext()),
      router(!!getZmqContext()),
      toAgents(getZmqContext()),
      configListener(getZmqContext())
{
}

void
PostAuctionLoop::
init()
{
    initConnections();
    monitorProviderClient.init(getServices()->config);
}

void
PostAuctionLoop::
initConnections()
{
    registerServiceProvider(serviceName(), { "rtbPostAuctionService" });

    cerr << "post auction logger on " << serviceName() + "/logger" << endl;
    logger.init(getServices()->config, serviceName() + "/logger");

    auctions.onEvent = std::bind<void>(&PostAuctionLoop::doAuction, this,
                                       std::placeholders::_1);
    events.onEvent   = std::bind<void>(&PostAuctionLoop::doEvent, this,
                                       std::placeholders::_1);
    toAgents.clientMessageHandler = [&] (const std::vector<std::string> & msg)
        {
            // Clients should never send the post auction service anything,
            // but we catch it here just in case
            cerr << "PostAuctionLoop got agent message " << msg << endl;
        };

    router.bind("AUCTION",
                std::bind(&PostAuctionLoop::doAuctionMessage, this,
                          std::placeholders::_1));
    router.bind("WIN",
                std::bind(&PostAuctionLoop::doWinMessage, this,
                          std::placeholders::_1));
    router.bind("LOSS",
                std::bind(&PostAuctionLoop::doLossMessage, this,
                          std::placeholders::_1));
    router.bind("IMPRESSION",
                std::bind(&PostAuctionLoop::doImpressionMessage, this,
                          std::placeholders::_1));
    router.bind("CLICK",
                std::bind(&PostAuctionLoop::doClickMessage, this,
                          std::placeholders::_1));
    router.bind("VISIT",
                std::bind(&PostAuctionLoop::doVisitMessage, this,
                          std::placeholders::_1));

    // Every second we check for expired auctions
    loop.addPeriodic("PostAuctionLoop::checkExpiredAuctions", 1.0,
                     std::bind<void>(&PostAuctionLoop::checkExpiredAuctions,
                                     this));

    // Initialize zeromq endpoints
    endpoint.init(getServices()->config, ZMQ_XREP, serviceName() + "/events");
    toAgents.init(getServices()->config, serviceName() + "/agents");
    configListener.init(getServices()->config);
    endpoint.messageHandler
        = std::bind(&ZmqMessageRouter::handleMessage,
                    &router,
                    std::placeholders::_1);

    loop.addSource("PostAuctionLoop::auctions", auctions);
    loop.addSource("PostAuctionLoop::events", events);

    loop.addSource("PostAuctionLoop::endpoint", endpoint);

    loop.addSource("PostAuctionLoop::toAgents", toAgents);
    loop.addSource("PostAuctionLoop::configListener", configListener);
    loop.addSource("PostAuctionLoop::logger", logger);
}

void
PostAuctionLoop::
bindTcp()
{
    logger.bindTcp(getServices()->ports->getRange("logs"));
    endpoint.bindTcp(getServices()->ports->getRange("postAuctionLoop"));
    toAgents.bindTcp(getServices()->ports->getRange("postAuctionLoopAgents"));
}

void
PostAuctionLoop::
start(std::function<void ()> onStop)
{
    loop.start(onStop);
    monitorProviderClient.start();
}

void
PostAuctionLoop::
shutdown()
{
    loop.shutdown();
    logger.shutdown();
    toAgents.shutdown();
    endpoint.shutdown();
    configListener.shutdown();
    monitorProviderClient.shutdown();
}

Json::Value
PostAuctionLoop::
getServiceStatus() const
{
    return Json::Value();
}

void
PostAuctionLoop::
throwException(const std::string & key, const std::string & fmt, ...)
{
    recordHit("error.exception");
    recordHit("error.exception.%s", key);

    string message;
    va_list ap;
    va_start(ap, fmt);
    try {
        message = vformat(fmt.c_str(), ap);
        va_end(ap);
    }
    catch (...) {
        va_end(ap);
        throw;
    }

    logPAError("exception", key, message);
    throw ML::Exception("Router Exception: " + key + ": " + message);
}

void
PostAuctionLoop::
injectWin(const Id & auctionId,
          const Id & adSpotId,
          Amount winPrice,
          Date timestamp,
          const JsonHolder & winMeta,
          const UserIds & uids,
          const AccountKey & account,
          Date bidTimestamp)
{
    auto event = std::make_shared<PostAuctionEvent>();
    event->type = PAE_WIN;
    event->auctionId = auctionId;
    event->adSpotId = adSpotId;
    event->timestamp = timestamp;
    event->winPrice = winPrice;
    event->metadata = winMeta;
    event->uids = uids;
    event->account = account;
    event->bidTimestamp = bidTimestamp;

    events.push(event);
}

void
PostAuctionLoop::
injectLoss(const Id & auctionId,
           const Id & adSpotId,
           Date timestamp,
           const JsonHolder & json,
           const AccountKey & account,
           Date bidTimestamp)
{
    //cerr << "injecting loss for " << auctionId << endl;

    if (timestamp == Date())
        timestamp = Date::now();

    auto event = std::make_shared<PostAuctionEvent>();
    event->type = PAE_LOSS;
    event->auctionId = auctionId;
    event->adSpotId = adSpotId;
    event->timestamp = timestamp;
    event->winPrice = Amount();
    event->account = account;
    event->bidTimestamp = bidTimestamp;

    events.push(event);
}

void
PostAuctionLoop::
injectImpression(const Id & auctionId,
                 const Id & adSpotId,
                 Date timestamp,
                 const JsonHolder & impressionMeta,
                 const UserIds & uids)
{
    auto event = std::make_shared<PostAuctionEvent>();
    event->type = PAE_IMPRESSION;
    event->auctionId = auctionId;
    event->adSpotId = adSpotId;
    event->timestamp = timestamp;
    event->metadata = impressionMeta;
    event->uids = uids;

    events.push(event);
}

void
PostAuctionLoop::
injectClick(const Id & auctionId,
            const Id & adSpotId,
            Date timestamp,
            const JsonHolder & clickMeta,
            const UserIds & uids)
{
    auto event = std::make_shared<PostAuctionEvent>();
    event->type = PAE_CLICK;
    event->auctionId = auctionId;
    event->adSpotId = adSpotId;
    event->timestamp = timestamp;
    event->metadata = clickMeta;
    event->uids = uids;

    events.push(event);
}

void
PostAuctionLoop::
injectVisit(Date timestamp,
            const SegmentList & channels,
            const JsonHolder & visitMeta,
            const UserIds & uids)
{
    auto event = std::make_shared<PostAuctionEvent>();
    event->type = PAE_VISIT;
    event->timestamp = timestamp;
    event->metadata = visitMeta;
    event->uids = uids;
    event->channels = channels;

    events.push(event);
}

void
PostAuctionLoop::
doAuctionMessage(const std::vector<std::string> & message)
{
    recordHit("messages.AUCTION");
    //cerr << "doAuctionMessage " << message << endl;

    SubmittedAuctionEvent event
        = ML::DB::reconstituteFromString<SubmittedAuctionEvent>(message.at(2));
    doAuction(event);
}

void
PostAuctionLoop::
doWinMessage(const std::vector<std::string> & message)
{
    recordHit("messages.WIN");
    auto event = std::make_shared<PostAuctionEvent>
        (ML::DB::reconstituteFromString<PostAuctionEvent>(message.at(2)));
    doWinLoss(event, false /* replay */);
}

void
PostAuctionLoop::
doLossMessage(const std::vector<std::string> & message)
{
    recordHit("messages.LOSS");
    auto event = std::make_shared<PostAuctionEvent>
        (ML::DB::reconstituteFromString<PostAuctionEvent>(message.at(2)));
    doWinLoss(event, false /* replay */);
}

void
PostAuctionLoop::
doImpressionMessage(const std::vector<std::string> & message)
{
    recordHit("messages.IMPRESSION");
    auto event = std::make_shared<PostAuctionEvent>
        (ML::DB::reconstituteFromString<PostAuctionEvent>(message.at(2)));
    doImpressionClick(event);
}

void
PostAuctionLoop::
doClickMessage(const std::vector<std::string> & message)
{
    recordHit("messages.CLICK");
    auto event = std::make_shared<PostAuctionEvent>
        (ML::DB::reconstituteFromString<PostAuctionEvent>(message.at(2)));
    doImpressionClick(event);
}

void
PostAuctionLoop::
doVisitMessage(const std::vector<std::string> & message)
{
    recordHit("messages.VISIT");
    auto event = std::make_shared<PostAuctionEvent>
        (ML::DB::reconstituteFromString<PostAuctionEvent>(message.at(2)));
    doVisit(event);
}

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

void
PostAuctionLoop::
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

    newTimeout = Date::now().plusSeconds(900);

    auto acceptFinished = [&] (pair<Id, Id> & key,
                               FinishedInfo & info,
                               Date & timeout) -> bool
        {
            info.fromOldRouter = true;
            newTimeout.addSeconds(0.001);
            timeout = newTimeout;
            // this->debugSpot(key.first, key.second, "RECONST FINISHED");

            // Index the IDs
            for (auto it = info.uids.begin(), end = info.uids.end();
                 it != end;  ++it) {
                uidIndex[*it][key] = Date::now();
            }

            return true;
        };

    finished.initFromStore(finishedPersistence,
                           acceptFinished,
                           Date::now().plusSeconds(900));

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

    loop.startSubordinateThread(backgroundWork);
}


void
PostAuctionLoop::
checkExpiredAuctions()
{
    Date start = Date::now();

    {
        cerr << " checking " << submitted.size()
             << " submitted auctions for inferred loss" << endl;


        //RouterProfiler profiler(this, dutyCycleCurrent.nsExpireSubmitted);


        auto onExpiredSubmitted = [&] (const pair<Id, Id> & key,
                                       const SubmissionInfo & info)
            {
                //RouterProfiler profiler(this, dutyCycleCurrent.nsOnExpireSubmitted);

                const Id & auctionId = key.first;
                const Id & adSpotId = key.second;

                recordHit("submittedAuctionExpiry");

                if (!info.bidRequest) {
                    recordHit("submittedAuctionExpiryWithoutBid");
                    //cerr << "expired with no bid request" << endl;
                    // this->debugSpot(auctionId, adSpotId, "EXPIRED SPOT NO BR", {});

                    // this->dumpSpot(auctionId, adSpotId);
                    return Date();
                }

                // this->debugSpot(auctionId, adSpotId, "EXPIRED SPOT", {});

                //cerr << "onExpiredSubmitted " << key << endl;
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

        //RouterProfiler profiler(this, dutyCycleCurrent.nsExpireFinished);

        auto onExpiredFinished = [&] (const pair<Id, Id> & key,
                                      const FinishedInfo & info)
            {
                recordHit("finishedAuctionExpiry");

                // this->debugSpot(key.first, key.second, "EXPIRED FINISHED", {});

                // We need to clean up the uid index
                for (auto it = info.uids.begin(), end = info.uids.end();
                     it != end;  ++it) {
                    const Id & uid = *it;
                    auto & entry = uidIndex[uid];
                    entry.erase(key);
                    if (entry.empty())
                        uidIndex.erase(uid);
                }
                return Date();
            };

        finished.expire(onExpiredFinished);
    }

    banker->logBidEvents(*this);
}

void
PostAuctionLoop::
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
PostAuctionLoop::
doEvent(const std::shared_ptr<PostAuctionEvent> & event)
{
    //cerr << "!!!PostAuctionLoop::doEvent:got post auction event " <<
    //print(event->type) << endl;

    try {
        switch (event->type) {
        case PAE_WIN:
        case PAE_LOSS:
            doWinLoss(event, false);
            break;
        case PAE_IMPRESSION:
        case PAE_CLICK:
            doImpressionClick(event);
            break;
        case PAE_VISIT:
            doVisit(event);
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
PostAuctionLoop::
addToUidIndex(const std::string & uidDomain,
              const Id & uid,
              const Id & auctionId,
              const Id & slotId)
{
    if (!uid) return;
    uidIndex[uid][make_pair(auctionId, slotId)] = Date::now();
}

void
PostAuctionLoop::
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
    const AccountKey & account = event->account;
    if (account.size() == 0) {
        throw ML::Exception("invalid account key");
    }

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
        if (info.hasWin()) {
            if (winPrice == info.winPrice
                && status == info.reportedStatus) {
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
        cerr << "WIN for already completed auction: " << meta
             << " timeGapMs = " << timeGapMs << endl;

        cerr << "info win: " << info.winMeta << " time " << info.winTime
             << " info.hasWin() = " << info.hasWin() << endl;

        if (event->type == PAE_WIN) {
            // Late win with auction still around
            banker->forceWinBid(account, winPrice, LineItems());

            info.setWin(timestamp, BS_WIN, winPrice, meta.toString());

            finished.update(key, info);

            recordHit("bidResult.%s.winAfterLossAssumed", typeStr);
            recordOutcome(winPrice.value,
                          "bidResult.%s.winAfterLossAssumedAmount.%s",
                          typeStr, winPrice.getCurrencyStr());

            cerr << "got late win with price " << winPrice
                 << " for account " << account << endl;
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

            banker->forceWinBid(account, winPrice, LineItems());

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
    std::for_each(info.earlyImpressionClickEvents.begin(),
                  info.earlyImpressionClickEvents.end(),
                  std::bind(&PostAuctionLoop::doImpressionClick, this,
                            std::placeholders::_1));

    //cerr << "doWinLoss done" << endl;
}

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

void
PostAuctionLoop::
doImpressionClick(const std::shared_ptr<PostAuctionEvent> & event)
{
    lastImpression = Date::now();

    //RouterProfiler profiler(this, dutyCycleCurrent.nsImpression);
    //static const char* fName = "PostAuctionLoop::doImpressionClick:";
    PostAuctionEventType typeEnum = event->type;
    const char * typeStr = print(event->type);
    const Id & auctionId = event->auctionId;
    Id adSpotId = event->adSpotId;
    Date timestamp = event->timestamp;
    const JsonHolder & meta = event->metadata;
    const UserIds & uids = event->uids;

    SubmissionInfo submissionInfo;
    FinishedInfo finishedInfo;

    recordHit("delivery.%s.messagesReceived", typeStr);

    //cerr << fName << typeStr << " " << auctionId << "-" << adSpotId << endl;
    //cerr <<"The number of elements in submitted " << submitted.size() << endl;
    auto recordUnmatched = [&] (const std::string & why)
        {
            this->logMessage(string("UNMATCHED") + typeStr, why,
                             auctionId.toString(), adSpotId.toString(),
                             to_string(timestamp.secondsSinceEpoch()),
                             meta);
            if (typeEnum == PAE_CLICK)
                cerr << "UNMATCHEDCLICK " << auctionId << "-"
                     << adSpotId << endl;
        };

    if (findAuction(submitted, auctionId, adSpotId, submissionInfo)) {
        // Record the impression or click in the submission info.  This will
        // then be passed on once the win comes in.
        //
        // TODO: for now we just ignore the event; we should eventually
        // implement what is written above
        //cerr << "auction " << auctionId << "-" << adSpotId
        //     << " in flight but got " << type << endl;
        recordHit("delivery.%s.stillInFlight", typeStr);
        logPAError(string("doImpressionClick.auctionNotWon") + typeStr,
                   "message for auction that's not won");
        recordUnmatched("inFlight");

        submissionInfo.earlyImpressionClickEvents.push_back(event);

        submitted.update(make_pair(auctionId, adSpotId), submissionInfo);
        return;
    }
    else if (findAuction(finished, auctionId, adSpotId, finishedInfo)) {
        // Update the info
        if (typeEnum == PAE_IMPRESSION) {
            if (finishedInfo.hasImpression()) {
                recordHit("delivery.%s.duplicate", typeStr);
                logPAError(string("doImpressionClick.duplicate") + typeStr,
                           "message duplicated");
                recordUnmatched("duplicate");
                return;
            }

            finishedInfo.setImpression(timestamp, meta.toString());
            ML::atomic_inc(numImpressions);

            recordHit("delivery.IMPRESSION.account.%s.matched",
                      finishedInfo.bid.account.toString().c_str());

            //Json::Value impInfo = Json::parse(meta);
            //cerr <<
            //cerr << meta << endl;
        }
        else if (typeEnum == PAE_CLICK) {
            if (finishedInfo.hasClick()) {
                recordHit("delivery.CLICK.duplicate");
                logPAError(string("doImpressionClick.duplicate") + typeStr,
                           "message duplicated");
                recordUnmatched("duplicate");
                return;
            }

            finishedInfo.setClick(timestamp, meta.toString());
            ML::atomic_inc(numClicks);

            cerr << "CLICK " << auctionId << "-" << adSpotId
                 << " " << finishedInfo.bid.account
                 << " " << finishedInfo.bid.agent
                 << " " << uids.toJson().toString()
                 << endl;

            recordHit("delivery.CLICK.account.%s.matched",
                      finishedInfo.bid.account.toString().c_str());
        }
        else throw ML::Exception("unknown delivery event");

        pair<Id, Id> key(auctionId, adSpotId);
        //cerr << "key = " << key << endl;
        if (!key.second)
            throw ML::Exception("updating null entry in finished map");

        // Add in the user IDs to the index so we can route any visits
        // properly
        finishedInfo.addUids(uids,
                             [&] (string dom, Id id)
                             {
                                 this->addToUidIndex(dom, id, auctionId,
                                                     adSpotId);
                             });

        finished.update(key, finishedInfo);

        routePostAuctionEvent(typeEnum, finishedInfo,
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



        recordHit("delivery.%s.auctionNotFound", typeStr);
        //cerr << "delivery " << typeStr << ": auction "
        //     << auctionId << "-" << adSpotId << " not found"
        //     << endl;
        logPAError(string("doImpressionClick.auctionNotFound") + typeStr,
                   "auction not found for delivery message");
        recordUnmatched("auctionNotFound");
    }
}

bool
PostAuctionLoop::
routePostAuctionEvent(PostAuctionEventType type,
                      const FinishedInfo & finishedInfo,
                      const SegmentList & channels,
                      bool filterChannels)
{
    // For the moment, send the message to all of the agents that are
    // bidding on this account
    const AccountKey & account = finishedInfo.bid.account;

    const char * typeStr = print(type);

    bool sent = false;
    auto onMatchingAgent = [&] (const AgentConfigEntry & entry)
        {
            if (!entry.config) return;
            if (filterChannels) {
                if (!entry.config->visitChannels.match(channels))
                    return;
            }

            sent = true;

            this->sendAgentMessage(entry.name,
                                   typeStr,
                                   Date::now(),
                                   finishedInfo.auctionId,
                                   finishedInfo.adSpotId,
                                   to_string(finishedInfo.spotIndex),
                                   finishedInfo.bidRequestStr,
                                   finishedInfo.bidToJson(),
                                   finishedInfo.winToJson(),
                                   finishedInfo.impressionToJson(),
                                   finishedInfo.clickToJson(),
                                   finishedInfo.augmentations,
                                   finishedInfo.visitsToJson(),
                                   finishedInfo.bidRequestStrFormat /* bidRequestSource */);
        };

    configListener.forEachAccountAgent(account, onMatchingAgent);

    if (!sent) {
        recordHit("delivery.%s.orphaned", typeStr);
        logPAError(string("doImpressionClick.noListeners") + typeStr,
                   "nothing listening for account " + account.toString());
    }
    else recordHit("delivery.%s.delivered", typeStr);

    // TODO: full account
    this->logMessage
        (string("MATCHED") + typeStr,
         finishedInfo.auctionId,
         finishedInfo.adSpotId,
         finishedInfo.bidRequestStr,
         finishedInfo.bidToJson(),
         finishedInfo.winToJson(),
         finishedInfo.impressionToJson(),
         finishedInfo.clickToJson(),
         finishedInfo.visitsToJson(),
         finishedInfo.bid.account[0],
         finishedInfo.bid.account[1],
         finishedInfo.bid.account.toString(),
         finishedInfo.bidRequestStrFormat);

    return sent;
}

void
PostAuctionLoop::
doVisit(const std::shared_ptr<PostAuctionEvent> & event)
{
    //RouterProfiler profiler(this, dutyCycleCurrent.nsVisit);

    Date timestamp = event->timestamp;
    const JsonHolder & meta = event->metadata;
    const UserIds & uids = event->uids;
    const SegmentList & channels = event->channels;

    FinishedInfo finishedInfo;

    //cerr << "channels = " << channels << endl;

    recordHit("delivery.VISIT.messagesReceived");

    channels.forEach([&] (int, string ch, float wt)
                     {
                         //cerr << "channel " << ch << endl;
                         this->recordHit("delivery.VISIT.channel.%s.messagesReceived", ch);
                     });

    //cerr << type << " " << auctionId << "-" << adSpotId << endl;

    auto recordUnmatched = [&] (const std::string & why)
        {
            this->recordHit("delivery.VISIT.%s", why);
            this->logMessage("UNMATCHEDVISIT", why,
                             to_string(timestamp.secondsSinceEpoch()),
                             meta, channels.toJsonStr());
        };

    // Try to find any of the UIDs in the index

    bool foundUser = false;

    for (auto it = uids.begin(), end = uids.end();  it != end;  ++it) {
        auto uit = uidIndex.find(it->second);
        if (uit == uidIndex.end()) continue;

        cerr << "visit metadata " << meta << endl;
        cerr << "UID " << *it << " matched "
             << uit->second.size() << " users" << endl;

        auto & entries = uit->second;

        foundUser = true;

        // For each auction this UID matches, we notify of the visit
        for (auto jt = entries.begin(), jend = entries.end();
             jt != jend;  ++jt) {
            Id auctionId = jt->first.first;
            Id adSpotId = jt->first.second;

            cerr << "  auction " << auctionId << " spot " << adSpotId
                 << endl;

            // Find if we have a finished auction (answer should be yes)
            std::pair<Id, Id> key(auctionId, adSpotId);

            recordHit("delivery.VISIT.foundMatchingAuction");

            FinishedInfo finishedInfo;
            if (!findAuction(finished, auctionId, adSpotId, finishedInfo)) {
                logPAError("doVisit.inconsistentIndex",
                           "auction in indexed not in finished");
                recordUnmatched("inconsistentIndex");
                return;
            }

            cerr << "  found finished auction for account "
                 << finishedInfo.bid.account << " with channels "
                 << finishedInfo.visitChannels << endl;

            // Check for a channel match
            if (!finishedInfo.visitChannels.match(channels)) {
                cerr << event->print() << endl;
                cerr << "channel mismatch: channels =  ";
                channels.forEach([&] (int, string ch, float wt)
                                 {
                                     cerr << " " << ch;
                                 });
                cerr << endl;
                recordUnmatched("channelMismatch");
                return;
            }

            recordHit("delivery.VISIT.matched");

            cerr << "matched" << endl;

            channels.forEach([&] (int, string ch, float wt)
                             {
                                 this->recordHit("delivery.VISIT.account.%s"
                                                 ".channel.%s.matched",
                                                 finishedInfo.bid.account.toString().c_str(),
                                                 ch);
                             });

            //cerr << "MATCHED VISIT " << message << endl;

            finishedInfo.addVisit(timestamp,
                                  meta.toString(),
                                  channels);
            finished.update(key, finishedInfo);

            routePostAuctionEvent(PAE_VISIT, finishedInfo, channels,
                                  true /* filterChannels */);
        }
    }

    if (foundUser) {
        recordHit("delivery.VISIT.foundUser");

        channels.forEach([&] (int, string ch, float wt)
                         {
                             this->recordHit("delivery.VISIT.channel.%s"
                                             ".foundUser", ch);
                         });
    }
}

void
PostAuctionLoop::
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
    //RouterProfiler profiler(this, dutyCycleCurrent.nsBidResult);
    //static const char *fName = "PostAuctionLoop::doBidResult:";
    string msg;

    if (status == BS_WIN) msg = "WIN";
    else if (status == BS_LOSS) msg = "LOSS";
    else throwException("doBidResult.nonWinLoss", "submitted non win/loss");

#if 0
    cerr << "doBidResult: " << msg
         << " id " << auctionId << " spot " << adSpotId
         << " submission " << submission.bid.agent << " "
         << submission.bid.price.maxPrice
         << " winPrice " << winPrice << "bid request string format<"
         << submission.bidRequestStrFormat << ">" <<  endl;
#endif

    // debugSpot(auctionId, adSpotId, msg);

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

#if 0
    if (doDebug)
        debugSpot(auctionId, adSpotId, "ACCOUNT " + account.toString());
#endif

    Amount bidPrice = response.price.maxPrice;

#if 0
    cerr << fName << "bidPriceMicros = " << bidPriceMicros
         << " winPriceMicros = " << winPriceMicros
         << endl;
#endif

    //cerr << "account = " << account << endl;

    if (winPrice > bidPrice) {
        //cerr << submission.bidRequestStr << endl;
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

    if (status == BS_WIN) {
        //info.stats->totalSpent += winPrice;
        //info.stats->totalBidOnWins += bidPrice;

        // This is a real win
        guard.clear();
        banker->winBid(account, makeBidId(auctionId, adSpotId, agent), winPrice,
                       LineItems());

        //++info.stats->wins;
        // local win; send it back

        //cerr << "MATCHEDWIN " << auctionId << "-" << adSpotId << endl;
    }
    else if (status == BS_LOSS) {
        //++info.stats->losses;
        // local loss; send it back
    }
    else throwException("doBidResult.nonWinLoss", "submitted non win/loss");
    logMessage("MATCHED" + msg,
               auctionId,
               to_string(adspot_num),
               response.agent,
               account[1],
               winPrice.toString(),
               response.price.maxPrice.toString(),
               to_string(response.price.priority),
               submission.bidRequestStr,
               response.bidData,
               response.meta,
               to_string(response.creativeId),
               response.creativeName,
               account[0],
               uids.toJsonStr(),
               winLossMeta,
               account[0],
               adSpotId,
               account.toString(),
               submission.bidRequestStrFormat);

    sendAgentMessage(response.agent, msg, timestamp,
                     confidence, auctionId,
                     to_string(adspot_num),
                     winPrice.toString(),
                     submission.bidRequestStr,
                     response.bidData,
                     "null",
                     response.meta,
                     submission.augmentations,
                     submission.bidRequestStrFormat
                     /* "datacratic" */);

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
    i.setWin(timestamp, status, winPrice, winLossMeta);

    // Copy the configuration into the finished info so that we can
    // know which visits to route back
    i.visitChannels = response.visitChannels;

    i.addUids(uids,
              [&] (string dom, Id id)
              {
                  this->addToUidIndex(dom, id, auctionId, adSpotId);
              });

    double expiryInterval = 3600;
    if (status == BS_LOSS)
        expiryInterval = 900;

    Date expiryTime = Date::now().plusSeconds(expiryInterval);

    finished.insert(make_pair(auctionId, adSpotId), i, expiryTime);
}

void
PostAuctionLoop::
injectSubmittedAuction(const Id & auctionId,
                       const Id & adSpotId,
                       std::shared_ptr<BidRequest> bidRequest,
                       const std::string & bidRequestStr,
                       const std::string & bidRequestStrFormat,
                       const JsonHolder & augmentations,
                       const Auction::Response & bidResponse,
                       Date lossTimeout)
{
    if (bidRequestStr.size() == 0) {
        throw ML::Exception("invalid bidRequestStr");
    }
    if (bidRequestStrFormat.size() == 0) {
        throw ML::Exception("invalid bidRequestStrFormat");
    }

    SubmittedAuctionEvent event;
    event.auctionId = auctionId;
    event.adSpotId = adSpotId;
    event.bidRequest = bidRequest;
    event.bidRequestStr = bidRequestStr;
    event.bidRequestStrFormat = bidRequestStrFormat;
    event.augmentations = augmentations;
    event.bidResponse = bidResponse;
    event.lossTimeout = lossTimeout;

    auctions.push(event);
}

void
PostAuctionLoop::
notifyFinishedSpot(const Id & auctionId, const Id & adSpotId)
{
    throw ML::Exception("notifyFinishedSpot(): not implemented");
}

std::string
PostAuctionLoop::
makeBidId(Id auctionId, Id spotId, const std::string & agent)
{
    return auctionId.toString() + "-" + spotId.toString() + "-" + agent;
}

std::string
PostAuctionLoop::
getProviderName()
    const
{
    return serviceName();
}

Json::Value
PostAuctionLoop::
getProviderIndicators()
    const
{
    Json::Value value;

    /* PA health check:
       - WINs in the last 10 seconds
       - IMPRESSIONs in the last 10 seconds */
    Date now = Date::now();
    bool status(now < lastWinLoss.plusSeconds(30)
                && now < lastImpression.plusSeconds(30));

#if 0
    if (!status)  {
      cerr << "--- WRONGNESS DETECTED:" 
	   << " wins: " << (now - lastWinLoss)
	   << ", imps: " << (now - lastImpression)
	   << endl;
    }
#endif

    value["status"] = status ? "ok" : "failure";

    return value;
}


} // namespace RTBKIT


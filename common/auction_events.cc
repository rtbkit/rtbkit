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


PostAuctionEvent::
PostAuctionEvent(Json::Value const & json)
    : type(PAE_INVALID)
{
    for (auto it = json.begin(), end = json.end(); it != end; ++it) {
        if (it.memberName() == "type")
            type = (PostAuctionEventType) it->asInt();
        else if (it.memberName() == "label")
            label = it->asString();
        else if (it.memberName() == "auctionId")
            auctionId.parse(it->asString());
        else if (it.memberName() == "adSpotId")
            adSpotId.parse(it->asString());
        else if (it.memberName() == "timestamp")
            timestamp = Date::fromSecondsSinceEpoch(it->asDouble());
        else if (it.memberName() == "account")
            account = AccountKey::fromJson(*it);
        else if (it.memberName() == "winPrice")
            winPrice = Amount::fromJson(*it);
        else if (it.memberName() == "uids")
            uids = UserIds::createFromJson(*it);
        else if (it.memberName() == "channels")
            channels = SegmentList::createFromJson(*it);
        else if (it.memberName() == "bidTimestamp")
            bidTimestamp = Date::fromSecondsSinceEpoch(it->asDouble());
        else throw ML::Exception("unknown location field " + it.memberName());
    }
}


Json::Value
PostAuctionEvent::
toJson() const
{
    Json::Value result;
    result["type"] = (int) type;
    result["auctionId"] = auctionId.toString();
    result["adSpotId"] = adSpotId.toString();
    result["timestamp"] = timestamp.secondsSinceEpoch();
    result["account"] = account.toJson();
    result["winPrice"] = winPrice.toJson();
    result["uids"] = uids.toJson();
    result["channels"] = channels.toJson();
    result["bidTimestamp"] = bidTimestamp.secondsSinceEpoch();
    return result;
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


/******************************************************************************/
/* CAMPAIGN EVENTS                                                            */
/******************************************************************************/

CampaignEvent
CampaignEvent::
fromJson(const Json::Value & jsonValue)
{
    double timeSeconds(jsonValue["time"].asDouble());

    CampaignEvent event(
            jsonValue["label"].asString(),
            Date::fromSecondsSinceEpoch(timeSeconds),
            jsonValue["meta"]);

    return event;
}

Json::Value
CampaignEvent::
toJson() const
{
    Json::Value json(Json::ValueType::objectValue);

    json["label"] = label_;
    json["timestamp"] = time_.secondsSinceEpoch();
    json["meta"] = meta_.toJson();

    return json;
}

void
CampaignEvent::
serialize(DB::Store_Writer & writer) const
{
    writer << label_ << time_.secondsSinceEpoch();
    meta_.serialize(writer);
}

void
CampaignEvent::
reconstitute(DB::Store_Reader & store)
{
    double timeSeconds;
    string metaStr;

    store >> label_ >> timeSeconds;
    time_ = Date::fromSecondsSinceEpoch(timeSeconds);
    meta_.reconstitute(store);
}

Json::Value
CampaignEvents::
toJson() const
{
    Json::Value json(Json::ValueType::arrayValue);

    for (const CampaignEvent & history: *this) {
        json.append(history.toJson());
    }

    return json;
}

CampaignEvents
CampaignEvents::
fromJson(const Json::Value& json)
{
    CampaignEvents events;
    ExcCheck(json.isArray(), "invalid format for a campaign events object");

    for (size_t i = 0; i < json.size(); ++i)
        events.push_back(CampaignEvent::fromJson(json[i]));

    return events;
}

bool
CampaignEvents::
hasEvent(const std::string & label) const
{
    for (const CampaignEvent & history: *this) {
        if (history.label_ == label) {
            return true;
        }
    }
    return false;
}

void
CampaignEvents::
setEvent(const std::string & label,
                 Date eventTime,
                 const JsonHolder & eventMeta)
{
    if (hasEvent(label))
        throw ML::Exception("already has event '" + label + "'");
    emplace_back(label, eventTime, eventMeta);
}


/******************************************************************************/
/* DELIVERY EVENTS                                                            */
/******************************************************************************/


DeliveryEvent::Bid
DeliveryEvent::Bid::
fromJson(const Json::Value& json)
{
    Bid bid;
    if (!json) return bid;

    bid.present = true;

    const auto& members = json.getMemberNames();

    for (const auto& m : members) {
        const Json::Value& member = json[m];
        bool invalid = false;

        switch(m[0]) {
        case 'a':
            if (m == "agent") bid.agent = member.asString();
            else if (m == "account") bid.account = AccountKey::fromJson(member);
            else invalid = true;
            break;

        case 'b':
            if (m == "bidData") bid.bids = Bids::fromJson(member.asString());
            else invalid = true;
            break;

        case 'c':
            if (m == "creativeId") bid.creativeId = member.asInt();
            else if (m == "creativeName") bid.creativeName = member.asString();
            else invalid = true;
            break;

        case 'l':
            if (m == "localStatus") {
                string status = member.asString();
                if (status == "PENDING") bid.localStatus = Auction::PENDING;
                else if (status == "WIN") bid.localStatus = Auction::WIN;
                else if (status == "LOSS") bid.localStatus = Auction::LOSS;
                else if (status == "TOOLATE") bid.localStatus = Auction::TOOLATE;
                else if (status == "INVALID") bid.localStatus = Auction::INVALID;
                else throw Exception("invalid localStatus value: " + status);
            }
            else invalid = true;
            break;

        case 'm':
            if (m == "meta") bid.meta = member.asString();
            else invalid = true;
            break;

        case 'p':
            if (m == "price") bid.price = Auction::Price::fromJson(member);
            else invalid = true;
            break;

        case 't':
            if (m == "timestamp")
                bid.time = Date::fromSecondsSinceEpoch(member.asDouble());
            else if (m == "test") bid.test = member.asBool();
            else if (m == "tagId") bid.test = member.asInt();
            else invalid = true;
            break;

        default:
            invalid = true;
            break;
        }

        ExcCheck(!invalid, "Unknown member: " + m);
    }

    return bid;
}

Json::Value
DeliveryEvent::Bid::
toJson() const
{
    Json::Value json;
    if (!present) return json;

    json["timestamp"] = time.secondsSinceEpoch();

    json["price"] = price.toJson();
    json["test"] = test;
    json["tagId"] = tagId;
    json["bidData"] = bids.toJson();

    json["agent"] = agent;
    json["account"] = account.toJson();
    json["meta"] = meta;

    json["creativeId"] = creativeId;
    json["creativeName"] = creativeName;

    json["localStatus"] = Auction::Response::print(localStatus);

    return json;
}



DeliveryEvent::Win
DeliveryEvent::Win::
fromJson(const Json::Value& json)
{
    Win win;
    if (!json) return win;
    win.present = true;

    const auto& members = json.getMemberNames();

    for (const auto& m : members) {
        const Json::Value& member = json[m];
        bool invalid = false;

        switch(m[0]) {
        case 'm':
            if (m == "meta") win.meta = member.asString();
            else invalid = true;
            break;

        case 't':
            if (m == "timestamp")
                win.time = Date::fromSecondsSinceEpoch(member.asDouble());
            else invalid = true;
            break;

        case 'r':
            if (m == "reportedStatus")
                win.reportedStatus = bidStatusFromString(member.asString());
            else invalid = true;
            break;

        case 'w':
            if (m == "winPrice") win.price = Amount::fromJson(member);
            else invalid = true;
            break;

        default:
            invalid = true;
            break;
        }

        ExcCheck(!invalid, "Unknown member: " + m);
    }

    return win;
}

Json::Value
DeliveryEvent::Win::
toJson() const
{
    Json::Value json;
    if (!present) return json;

    json["timestamp"] = time.secondsSinceEpoch();
    json["reportedStatus"] = (reportedStatus == BS_WIN ? "WIN" : "LOSS");
    json["winPrice"] = price.toJson();
    json["meta"] = meta;

    return json;
}

Json::Value
DeliveryEvent::
impressionToJson() const
{
    Json::Value json;

    for (const CampaignEvent& ev : campaignEvents) {
        if (ev.label_ != "IMPRESSION") continue;
        json = ev.toJson();
        break;
    }

    return json;
}

Json::Value
DeliveryEvent::
clickToJson() const
{
    Json::Value json;

    for (const CampaignEvent& ev : campaignEvents) {
        if (ev.label_ != "CLICK") continue;
        json = ev.toJson();
        break;
    }

    return json;
}

DeliveryEvent::Visit
DeliveryEvent::Visit::
fromJson(const Json::Value& json)
{
    Visit visit;
    if (!json) return visit;

    const auto& members = json.getMemberNames();

    for (const auto& m : members) {
        const Json::Value& member = json[m];
        bool invalid = false;

        switch(m[0]) {

        case 'c':
            if (m == "channels")
                visit.channels = SegmentList::createFromJson(member);
            else invalid = true;
            break;

        case 'm':
            if (m == "meta") visit.meta = member.asString();
            else invalid = true;
            break;

        case 't':
            if (m == "timestamp")
                visit.time = Date::fromSecondsSinceEpoch(member.asDouble());
            else invalid = true;
            break;

        default:
            invalid = true;
            break;
        }

        ExcCheck(!invalid, "Unknown member: " + m);
    }

    return visit;
}

Json::Value
DeliveryEvent::Visit::
toJson() const
{
    Json::Value json;

    json["timestamp"] = time.secondsSinceEpoch();
    json["channels"] = channels.toJson();
    json["meta"] = meta;

    return json;
}


Json::Value
DeliveryEvent::
visitsToJson() const
{
    Json::Value json;

    for (const auto& visit : visits)
        json.append(visit.toJson());

    return json;
}



DeliveryEvent
DeliveryEvent::
parse(const std::vector<std::string>& msg)
{
    DeliveryEvent ev;
    ExcCheckGreaterEqual(msg.size(), 12, "Invalid message size");

    using boost::lexical_cast;

    ev.event = msg[0];
    ev.timestamp = Date::fromSecondsSinceEpoch(lexical_cast<double>(msg[1]));
    ev.auctionId = Id(msg[2]);
    ev.spotId = Id(msg[3]);
    ev.spotIndex = lexical_cast<int>(msg[4]);

    string bidRequestSource = msg[5];
    ev.bidRequest.reset(BidRequest::parse(bidRequestSource, msg[6]));

    ev.augmentations = msg[7];

    auto jsonParse = [] (const string& str)
        {
            if (str.empty()) return Json::Value();
            return Json::parse(str);
        };

    ev.bid = Bid::fromJson(jsonParse(msg[8]));
    ev.win = Win::fromJson(jsonParse(msg[9]));
    ev.campaignEvents = CampaignEvents::fromJson(jsonParse(msg[10]));

    const Json::Value& visits = jsonParse(msg[11]);
    for (size_t i = 0; i < visits.size(); ++i)
        ev.visits.push_back(Visit::fromJson(visits[i]));

    return ev;
}


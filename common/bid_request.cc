/* bid_request.cc
   Jeremy Barnes, 1 February 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

*/

#include "rtbkit/common/bid_request.h"
#include "jml/arch/exception.h"
#include "jml/arch/format.h"
#include "jml/arch/backtrace.h"
#include "jml/arch/spinlock.h"

#include "jml/utils/parse_context.h"
#include "jml/utils/less.h"
#include "jml/utils/string_functions.h"

#include <dlfcn.h>
#include <boost/thread/locks.hpp>
#include <boost/algorithm/string.hpp>
#include <unordered_map>

#include "jml/db/persistent.h"


using namespace std;
using namespace ML;
using namespace ML::DB;


namespace RTBKIT {


/*****************************************************************************/
/* FORMAT                                                                    */
/*****************************************************************************/

void
Format::
fromJson(const Json::Value & val)
{
    if (val.isString()) {
        string s = val.asString();
        fromString(s);
#if 0
        int width, height;
        int nchars = -1;
        int res = sscanf(s.c_str(), "%dx%d%n", &width, &height, &nchars);
        if ((res != 2 && res != 3) || nchars != s.length())
            throw ML::Exception("couldn't parse format string "
                                + s);
        this->width = width;
        this->height = height;
#endif
    } else if (val.isArray()) {
        throw ML::Exception("array format parsing not done yet");
    }
    else if (val.isObject()) {
        width = val["width"].asInt();
        height = val["height"].asInt();
    }
    else throw ML::Exception("couldn't parse format string " + val.toString());
}

void
Format::
fromString(const string &s)
{
    int width, height;
    int nchars = -1;
    int res = sscanf(s.c_str(), "%dx%d%n", &width, &height, &nchars);
    if ((res != 2 && res != 3) || nchars != s.length())
        throw ML::Exception("couldn't parse format string "
                            + s);
    this->width = width;
    this->height = height;
}

Json::Value
Format::
toJson() const
{
    return print();
}

std::string
Format::
toJsonStr() const
{
    return print();
}

std::string
Format::
print() const
{
    return ML::format("%dx%d", width, height);
}

void
Format::
serialize(ML::DB::Store_Writer & store) const
{
    store << compact_size_t(width) << compact_size_t(height);
}

void
Format::
reconstitute(ML::DB::Store_Reader & store)
{
    width = compact_size_t(store);
    height = compact_size_t(store);
}


/*****************************************************************************/
/* FORMAT SET                                                                */
/*****************************************************************************/

void
FormatSet::
fromJson(const Json::Value & val)
{
    clear();
    if (val.isString()) {
        Format f;
        f.fromJson(val);
        push_back(f);
        return;
    }
    else if (val.isArray()) {
        for (unsigned i = 0;  i < val.size();  ++i) {
            Format f;
            f.fromJson(val[i]);
            push_back(f);
        }
    }
    else throw ML::Exception("couldn't parse format set from JSON "
                             + val.toString());
}

Json::Value
FormatSet::
toJson() const
{
    Json::Value result;
    if (empty()) return result;
    //if (size() == 1)
    //    return result = at(0).toJson();
    for (unsigned i = 0;  i < size();  ++i)
        result[i] = at(i).toJson();
    return result;
}

std::string
FormatSet::
toJsonStr() const
{
    return boost::trim_copy(toJson().toString());
}

std::string
FormatSet::
print() const
{
    if (empty()) return "[]";
    else if (size() == 1)
        return at(0).print();
    else {
        string result = "[";
        for (unsigned i = 0;  i < size();  ++i) {
            if (i != 0) result += ", ";
            result += at(i).print();
        }
        result += ']';
        return result;
    }
}

void
FormatSet::
sort()
{
    std::sort(begin(), end());
}

bool isEmpty(const std::string & str)
{
    return str.empty();
}

bool isEmpty(const Json::Value & val)
{
    return val.isNull();
}

bool isEmpty(const Utf8String &str)
{
    return (str.rawLength() == 0) ;
}

template<typename T>
void addIfNotEmpty(Json::Value & obj, const std::string & key,
                   const T & val)
{
    if (isEmpty(val)) return;
    obj[key] = val;
}

template<typename T>
void addIfNotEmpty(Json::Value & obj, const std::string & key,
                   const T & val, const T & emptyVal)
{
    if (val == emptyVal) return;
    obj[key] = val;
}

inline void addIfNotEmpty(Json::Value & obj, const std::string & key,
                          const Url & url)
{
    if (url.empty()) return;
    obj[key] = url.toString();
}

void
FormatSet::
serialize(ML::DB::Store_Writer & store) const
{
    store << ML::compact_vector<Format, 3, uint16_t>(*this);
}

void
FormatSet::
reconstitute(ML::DB::Store_Reader & store)
{
    ML::compact_vector<Format, 3, uint16_t> & v = *this;
    store >> v;
}


/*****************************************************************************/
/* LOCATION                                                                  */
/*****************************************************************************/

Utf8String
Location::
fullLocationString() const
{
    Utf8String result(countryCode + ":" + regionCode + ":") ;
    result += cityName ;
    result += (":" + postalCode + ":" + boost::lexical_cast<string>(dma));
    return result;
    //Utf8String result(countryCode +":"+ regionCode +":" +
 //   return ML::format("%s:%s:%s:%s:%d",
 //                     countryCode.c_str(), regionCode.c_str(),
 //                     cityName.c_str(), postalCode.c_str(), dma);
}

Json::Value
Location::
toJson() const
{
    Json::Value result;
    addIfNotEmpty(result, "countryCode",  countryCode);
    addIfNotEmpty(result, "regionCode",   regionCode);
    addIfNotEmpty(result, "cityName",     std::string(cityName.rawData(),cityName.rawLength()));
    addIfNotEmpty(result, "postalCode",   postalCode);
    addIfNotEmpty(result, "dma",          dma, -1);
    addIfNotEmpty(result, "timezoneOffsetMinutes", timezoneOffsetMinutes, -1);
    return result;
}

std::string
Location::
toJsonStr() const
{
    return boost::trim_copy(toJson().toString());
}

Location
Location::
createFromJson(const Json::Value & json)
{
    Location result;

    for (auto it = json.begin(), end = json.end(); it != end;  ++it) {
        if (it.memberName() == "countryCode")
            result.countryCode = it->asString();
        else if (it.memberName() == "regionCode")
            result.regionCode = it->asString();
        else if (it.memberName() == "cityName")
            result.cityName = it->asString();
        else if (it.memberName() == "postalCode")
            result.postalCode = it->asString();
        else if (it.memberName() == "dma")
            result.dma = it->asInt();
        else if (it.memberName() == "timezoneOffsetMinutes")
            result.timezoneOffsetMinutes = it->asInt();
        else throw ML::Exception("unknown location field " + it.memberName());
    }
    return result;
}

void
Location::
serialize(ML::DB::Store_Writer & store) const
{
    unsigned char version = 0;
    store << version << countryCode << regionCode << cityName << postalCode
          << compact_size_t(dma) << compact_size_t(timezoneOffsetMinutes);
}

void
Location::
reconstitute(ML::DB::Store_Reader & store)
{
    unsigned char version;
    store >> version;
    if (version != 0)
        throw ML::Exception("invalid Location version");
    store >> countryCode >> regionCode >> cityName >> postalCode;
    dma = compact_size_t(store);
    timezoneOffsetMinutes = compact_size_t(store);
}


/*****************************************************************************/
/* AD SPOT                                                                   */
/*****************************************************************************/

AdSpot::
AdSpot(const Id & id, int reservePrice)
    : id(id), reservePrice(reservePrice), position(Position::NONE)
{
}

namespace {

SmallIntVector getDims(const Json::Value & val)
{
    SmallIntVector result;

    if (val.isArray()) {
        for (unsigned i = 0;  i < val.size();  ++i)
            result.push_back(val[i].asInt());
    }
    else result.push_back(val.asInt());

    return result;
}

} // file scope

AdSpot::Position
AdSpot::stringToPosition(const std::string &pos)
{
    if (pos == "None" || pos == "NONE" || pos == "none")
        return AdSpot::Position::NONE;
    else if (pos == "ABOVE_FOLD" || pos == "above")
        return AdSpot::Position::ABOVE_FOLD;
    else if (pos == "BELOW_FOLD" || pos == "below")
        return AdSpot::Position::BELOW_FOLD;
    else
        throw ML::Exception(" Unknown value for AdSpot::Position ==>" + pos);
}

void
AdSpot::
fromJson(const Json::Value & val)
{
    try {
        id = Id(val["id"].asString());
        if (val.isMember("formats")) {
            formats.fromJson(val["formats"]);
        }
        else {
            SmallIntVector widths = getDims(val["width"]);
            SmallIntVector heights = getDims(val["height"]);
            formats.clear();
            if (widths.size() != heights.size())
                throw ML::Exception("widths and heights must have same size");
            for (unsigned i = 0;  i < widths.size();  ++i)
                formats.push_back(Format(widths[i], heights[i]));
        }
        reservePrice = val["reservePrice"].asInt();
        if (val.isMember("position"))
            position = stringToPosition(val["position"].asString());
        else
            position = Position::NONE;
    } catch (...) {
        cerr << "parsing AdSpot " << val << endl;
        throw;
    }
}

Json::Value
AdSpot::
toJson() const
{
    Json::Value result;
    result["id"] = id.toString();
    result["formats"] = formats.toJson();
    //result["width"] = formats.getWidthsJson();
    //result["height"] = formats.getHeightsJson();
    result["reservePrice"] = reservePrice;
    result["position"] = positionToStr();
    return result;
}

std::string formatDims(const SmallIntVector & dims)
{
    if (dims.size() == 1)
        return ML::format("%d", (int)dims[0]);

    string result = "[";
    for (unsigned i = 0;  i < dims.size();  ++i) {
        result += ML::format("%d", (int)dims[i]);
        if (i != dims.size() - 1)
            result += ',';
    }
    return result + "]";
}

std::string
AdSpot::
format() const
{
    return formats.print();
}

std::string
AdSpot::
firstFormat() const
{
    return formats[0].print();
}

AdSpot
AdSpot::
createFromJson(const Json::Value & json)
{
    AdSpot result;
    result.fromJson(json);
    return result;
}

std::string
AdSpot::
positionToStr() const
{
    return positionToStr(position);
}

std::string
AdSpot::
positionToStr(Position position)
{
    if (position == Position::ABOVE_FOLD) return "ABOVE_FOLD";
    if (position == Position::ABOVE_FOLD) return "BELOW_FOLD";
    return "NONE";
}

void jsonParse(const Json::Value & value, AdSpot::Position & pos)
{
    pos = AdSpot::stringToPosition(value.asString());
}

Json::Value jsonPrint(const AdSpot::Position & pos)
{
    return AdSpot::positionToStr(pos);
}

COMPACT_PERSISTENT_ENUM_IMPL(AdSpot::Position);

void
AdSpot::
serialize(ML::DB::Store_Writer & store) const
{
    store << id << formats << compact_size_t(reservePrice) << position;
}

void
AdSpot::
reconstitute(ML::DB::Store_Reader & store)
{
    store >> id >> formats;
    reservePrice = compact_size_t(reservePrice);
    store >> position;
}


/*****************************************************************************/
/* USER IDS                                                                  */
/*****************************************************************************/

void
UserIds::
add(const Id & id, IdDomain domain)
{
    if (!insert(make_pair(domainToString(domain), id)).second)
        throw ML::Exception("attempt to double add id %s for %s",
                            id.toString().c_str(), domainToString(domain));
    setStatic(id, domain);
}

void
UserIds::
add(const Id & id, const std::string & domain1, IdDomain domain2)
{
    add(id, domain1);
    add(id, domain2);
}

void
UserIds::
add(const Id & id, const std::string & domain)
{
    if (!insert(make_pair(domain, id)).second)
        throw ML::Exception("attempt to double add id " + id.toString() +" for " + domain);
    setStatic(id, domain);
}

const char *
UserIds::
domainToString(IdDomain domain)
{
    switch (domain) {
    case ID_PROVIDER:   return "prov";
    case ID_EXCHANGE:   return "xchg";
    default:            return "<<<UNKNOWN>>>";
    }
}

void
UserIds::
setStatic(const Id & id, const std::string & domain)
{
    if (domain == "prov")
        providerId = id;
    else if (domain == "xchg")
        exchangeId = id;
}

void
UserIds::
setStatic(const Id & id, IdDomain domain)
{
    switch (domain) {
    case ID_PROVIDER:   providerId = id;  break;
    case ID_EXCHANGE:   exchangeId = id;  break;
    default: break;
    }
}

void
UserIds::
set(const Id & id, const std::string & domain)
{
    (*this)[domain] = id;
}

Json::Value
UserIds::
toJson() const
{
    Json::Value result;
    for (auto it = begin(), end = this->end();  it != end;  ++it)
        result[it->first] = it->second.toString();
    return result;
}

std::string
UserIds::
toJsonStr() const
{
    return boost::trim_copy(toJson().toString());
}

UserIds
UserIds::
createFromJson(const Json::Value & json)
{
    UserIds result;

    for (auto it = json.begin(), end = json.end(); it != end;  ++it) {
        Id id(it->asString());
        result.add(id, it.memberName());
    }

    return result;
}

std::string
UserIds::
serializeToString() const
{
    // TODO: do a better job...
    return toJsonStr();
}

UserIds
UserIds::
createFromString(const std::string & str)
{
    // TODO: do a better job...
    return createFromJson(Json::parse(str));
}

void
UserIds::
serialize(ML::DB::Store_Writer & store) const
{
    unsigned char version = 0;
    store << version << (map<std::string, Id> &)(*this);
}

void
UserIds::
reconstitute(ML::DB::Store_Reader & store)
{
    unsigned char version;
    store >> version;
    if (version != 0)
        throw ML::Exception("invalid UserIds version");
    store >> (map<std::string, Id> &)*this;
}


/*****************************************************************************/
/* BID REQUEST                                                               */
/*****************************************************************************/

void
BidRequest::
sortAll()
{
    for (unsigned i = 0;  i < spots.size();  ++i)
        spots[i].formats.sort();
    restrictions.sortAll();
    segments.sortAll();
}

Json::Value
BidRequest::
toJson() const
{
    Json::Value result;
    result["!!CV"] = "0.1";
    result["id"] = auctionId.toString();
    result["timestamp"] = timestamp;
    addIfNotEmpty(result, "isTest", isTest, false);
    addIfNotEmpty(result, "url", url);
    addIfNotEmpty(result, "ipAddress", ipAddress);
    addIfNotEmpty(result, "userAgent", userAgent);
    addIfNotEmpty(result, "language", language);
    addIfNotEmpty(result, "protocolVersion", protocolVersion);
    addIfNotEmpty(result, "exchange", exchange);
    addIfNotEmpty(result, "provider", provider);
    addIfNotEmpty(result, "meta", meta);
    addIfNotEmpty(result, "creative", creative);

    if (!winSurcharges.empty())
        result["winSurcharges"] = winSurcharges.toJson();

    result["location"] = location.toJson();

    if (!spots.empty()) {
        for (unsigned i = 0;  i < spots.size();  ++i)
            result["spots"][i] = spots[i].toJson();
    }

    if (!segments.empty())
        result["segments"] = segments.toJson();
    if (!restrictions.empty())
        result["restrictions"] = restrictions.toJson();
    if (!userIds.empty())
        result["userIds"] = userIds.toJson();

    return result;
}

std::string
BidRequest::
toJsonStr() const
{
    return boost::trim_copy(toJson().toString());
}

BidRequest
BidRequest::
createFromJson(const Json::Value & json)
{
    BidRequest result;

    string canonicalVersion;

    for (auto it = json.begin(), end = json.end(); it != end;  ++it) {
        if (it.memberName() == "!!CV") {
            canonicalVersion = it->asString();
            if (canonicalVersion > "0.1")
                throw ML::Exception("can't parse BidRequest with CV "
                                    + canonicalVersion);
        }
        else if (it.memberName() == "id")
            result.auctionId.parse(it->asString());
        else if (it.memberName() == "timestamp")
            result.timestamp = it->asDouble();
        else if (it.memberName() == "isTest")
            result.isTest = it->asBool();
        else if (it.memberName() == "url")
            result.url = Url(it->asString());
        else if (it.memberName() == "ipAddress")
            result.ipAddress = it->asString();
        else if (it.memberName() == "userAgent")
            result.userAgent = it->asString();
        else if (it.memberName() == "language")
            result.language = it->asString();
        else if (it.memberName() == "protocolVersion")
            result.protocolVersion = it->asString();
        else if (it.memberName() == "exchange")
            result.exchange = it->asString();
        else if (it.memberName() == "provider")
            result.provider = it->asString();
        else if (it.memberName() == "winSurchageMicros")
            result.winSurcharges["surcharge"] += MicroUSD(it->asInt());
        else if (it.memberName() == "winSurcharges")
            result.winSurcharges += LineItems::fromJson(*it);
        else if (it.memberName() == "meta")
            result.meta = *it;
        else if (it.memberName() == "creative")
            result.creative = *it;
        else if (it.memberName() == "location")
            result.location = Location::createFromJson(*it);
        else if (it.memberName() == "segments")
            result.segments = SegmentsBySource::createFromJson(*it);
        else if (it.memberName() == "restrictions")
            result.restrictions = SegmentsBySource::createFromJson(*it);
        else if (it.memberName() == "userIds")
            result.userIds = UserIds::createFromJson(*it);
        else if (it.memberName() == "spots") {
            const Json::Value & json = *it;
            if (!json.empty() && !json.isArray())
                throw ML::Exception("spots is not an array");
            for (unsigned i = 0;  i < json.size();  ++i) {
                result.spots.push_back(AdSpot::createFromJson(json[i]));
            }
        }
        else throw ML::Exception("unknown canonical bid request field "
                                 + it.memberName());
    }
    return result;
}

namespace {
typedef std::unordered_map<std::string, BidRequest::Parser> Parsers;
static Parsers parsers;
typedef boost::lock_guard<ML::Spinlock> Guard;
static ML::Spinlock lock;

BidRequest::Parser getParser(std::string const & source) {
    // see if it's already existing
    {
        Guard guard(lock);
        auto i = parsers.find(source);
        if (i != parsers.end()) return i->second;
    }

    // else, try to load the parser library
    std::string path = "lib" + source + "_bid_request.so";
    void * handle = dlopen(path.c_str(), RTLD_NOW);
    if (!handle) {
        throw ML::Exception("couldn't find bid request parser library " + path);
    }

    // if it went well, it should be registered now
    Guard guard(lock);
    auto i = parsers.find(source);
    if (i != parsers.end()) return i->second;

    throw ML::Exception("couldn't find bid request parser for source " + source);
}

} // file scope

void
BidRequest::
registerParser(const std::string & source, Parser parser)
{
    Guard guard(lock);
    if (!parsers.insert(make_pair(source, parser)).second)
        throw ML::Exception("already had a bid request parser registered");
}

namespace {

struct CanonicalParser {
    static BidRequest * parse(const std::string & str)
    {
        auto json = Json::parse(str);
        auto_ptr<BidRequest> result(new BidRequest());
        *result = BidRequest::createFromJson(json);
        return result.release();
    }
};

struct AtInit {
    AtInit()
    {
        BidRequest::registerParser("recoset", CanonicalParser::parse);
        BidRequest::registerParser("datacratic", CanonicalParser::parse);
    }
} atInit;
} // file scope

BidRequest *
BidRequest::
parse(const std::string & source, const std::string & bidRequest)
{
    if (source.empty()) {
        throw ML::Exception("'source' parameter cannot be empty");
    }

    if (source == "datacratic" || strncmp(bidRequest.c_str(), "{\"!!CV\":", 8) == 0)
        return CanonicalParser::parse(bidRequest);

    Parser parser = getParser(source);
    return parser(bidRequest);
}

BidRequest *
BidRequest::
parse(const std::string & source, const Utf8String & bidRequest)
{
    return BidRequest::parse(source, string(bidRequest.rawData(), bidRequest.rawLength()));
}

SegmentResult
BidRequest::
segmentPresent(const std::string & source,
               const std::string & segment) const
{
    auto it = segments.find(source);
    if (it == segments.end())
        return SEG_MISSING;
    return (it->second->contains(segment)
            ? SEG_PRESENT : SEG_NOT_PRESENT);
}

SegmentResult
BidRequest::
segmentPresent(const std::string & source, int segment) const
{
    auto it = segments.find(source);
    if (it == segments.end())
        return SEG_MISSING;
    return (it->second->contains(segment)
            ? SEG_PRESENT : SEG_NOT_PRESENT);
}

Id
BidRequest::
getUserId(IdDomain domain) const
{
    switch (domain) {
    case ID_PROVIDER:   return userIds.providerId;
    case ID_EXCHANGE:   return userIds.exchangeId;
    default:            throw ML::Exception("unknown ID for getUserId");
    }
}

Id
BidRequest::
getUserId(const std::string & domain) const
{
    auto it = userIds.find(domain);
    if (it == userIds.end())
        return Id();
    return it->second;
}

std::string
BidRequest::
serializeToString() const
{
    ostringstream stream;
    DB::Store_Writer store(stream);
    serialize(store);
    return stream.str();
}

BidRequest
BidRequest::
createFromString(const std::string & str)
{
    DB::Store_Reader store(str.c_str(), str.size());
    BidRequest result;
    result.reconstitute(store);
    return result;
}

inline ML::DB::Store_Writer &
operator << (ML::DB::Store_Writer & store, const Json::Value & val)
{
    return store << val.toString();
}

inline ML::DB::Store_Reader &
operator >> (ML::DB::Store_Reader & store, Json::Value & val)
{
    string s;
    store >> s;
    val = Json::parse(s);
    return store;
}


void
BidRequest::
serialize(ML::DB::Store_Writer & store) const
{
    using namespace ML::DB;
    unsigned char version = 1;
    store << version << auctionId << language << protocolVersion
          << exchange << provider << timestamp << isTest
          << location << userIds << spots << url << ipAddress << userAgent
          << restrictions << segments << meta << creative
          << winSurcharges;
}

void
BidRequest::
reconstitute(ML::DB::Store_Reader & store)
{
    using namespace ML::DB;

    unsigned char version;

    store >> version;

    if (version > 1)
        throw ML::Exception("problem reconstituting BidRequest: "
                            "invalid version");

    store >> auctionId >> language >> protocolVersion
          >> exchange >> provider >> timestamp >> isTest
          >> location >> userIds >> spots >> url >> ipAddress >> userAgent
          >> restrictions >> segments >> meta >> creative;
    if (version == 0) {
        uint64_t winSurchargeMicros = compact_size_t(store);
        winSurcharges.clear();
        winSurcharges["surcharge"] += MicroUSD(winSurchargeMicros);
    }
    else store >> winSurcharges;
}

} // namespace RTBKIT

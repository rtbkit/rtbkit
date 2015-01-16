/* bid_request.cc
   Jeremy Barnes, 1 February 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

*/

#include "rtbkit/common/bid_request.h"
#include "jml/arch/exception.h"
#include "jml/arch/format.h"

#include <boost/thread/locks.hpp>
#include <boost/algorithm/string.hpp>
#include <unordered_map>

#include "jml/db/persistent.h"
#include "rtbkit/openrtb/openrtb_parsing.h"
#include "soa/types/json_printing.h"
#include "soa/service/json_codec.h"


using namespace std;
using namespace ML;
using namespace ML::DB;

namespace Datacratic {

using namespace RTBKIT;

DefaultDescription<Location>::
DefaultDescription()
{
    this->nullAccepted = true;
    addField("countryCode", &Location::countryCode,
             "Country code of location");
    addField("regionCode", &Location::regionCode,
             "Region code of location");
    addField("cityName", &Location::cityName,
             "City name of location");
    addField("postalCode", &Location::postalCode,
             "Postal code of location");
    addField("dma", &Location::dma,
             "DMA code of location");
    addField("metro", &Location::metro,
             "Metro code of location");
    addField("timezoneOffsetMinutes", &Location::timezoneOffsetMinutes,
             "Timezone offset of location in minutes");
}

DefaultDescription<Format>::
DefaultDescription()
{
}

struct HistoricalPositionDescriptor
    : public ValueDescriptionT<OpenRTB::AdPosition> {
    
    virtual void parseJsonTyped(OpenRTB::AdPosition * val,
                                JsonParsingContext & context) const
    {
        if (context.isString()) {
            string s = context.expectStringAscii();
            if (s == "NONE" || s == "none") {
                val->val = OpenRTB::AdPosition::UNKNOWN;
            }
            else if (s == "ABOVE_FOLD" || s == "above") {
                val->val = OpenRTB::AdPosition::ABOVE;
            }
            else if (s == "BELOW_FOLD" || s == "below") {
                val->val = OpenRTB::AdPosition::BELOW;
            }
            else throw ML::Exception("invalid ad position " + s);
        }
        else if (context.isNumber()) {
            int i = context.expectInt();
            val->val = i;
        }
        else throw ML::Exception("can't parse historical ad position " 
                                 + context.expectJson().toString());
    }

    virtual void printJsonTyped(const OpenRTB::AdPosition * val,
                                JsonPrintingContext & context) const
    {
        context.writeInt(val->val);
    }

    virtual bool isDefaultTyped(const OpenRTB::AdPosition * val) const
    {
        return val->val == -1;
    }
};

DefaultDescription<AdSpot>::
DefaultDescription()
{
    addParent<OpenRTB::Impression>();
        
    addField("formats", &AdSpot::formats, "Impression formats");
    addField<OpenRTB::AdPosition>("position", &AdSpot::position, "Impression fold position",
                                  new HistoricalPositionDescriptor());
    addField("reservePrice", &AdSpot::reservePrice, "Impression reserve price");

    //throw ML::Exception("Need to sync formats with the bid request");
}

DefaultDescription<BidRequest>::
DefaultDescription()
{
    onUnknownField = [=] (BidRequest * br, JsonParsingContext & context)
        {
            //context.skip();
            if(context.printPath().find("!!CV") != std::string::npos)
            {
               context.skip();
            }
            else
            {
               cerr << "(default description)got unknown field " << context.printPath() << endl;
            std::function<Json::Value & (int, Json::Value &)> getEntry
            = [&] (int n, Json::Value & curr) -> Json::Value &
            {
                if (n == context.path.size())
                    return curr;
                else if (context.path[n].index != -1)
                    return getEntry(n + 1, curr[context.path[n].index]);
                else return getEntry(n + 1, curr[context.path[n].fieldName()]);
            };

            getEntry(0, br->unparseable)
            = context.expectJson();
            }
        };
    addField("id", &BidRequest::auctionId, "Exchange auction ID");
    addField("timestamp", &BidRequest::timestamp, "Bid request timestamp");
    addField("isTest", &BidRequest::isTest, "Is bid request a test?");
    addField("url", &BidRequest::url, "Site URL");
    addField("ipAddress", &BidRequest::ipAddress, "IP address of user");
    addField("userAgent", &BidRequest::userAgent, "User agent of device");
    addField("language", &BidRequest::language, "User language code");
    addField("protocolVersion", &BidRequest::protocolVersion,
             "bid request protocol version");
    addField("exchange", &BidRequest::exchange, "Original bid request exchagne");
    addField("provider", &BidRequest::provider, "Bid request provider");
    addField("winSurcharges", &BidRequest::winSurcharges,
             "extra amounts paid on win");
    addField("winSurchageMicros", &BidRequest::winSurcharges,
             "extra amounts paid on win");
    addField("meta", &BidRequest::meta,
             "extra metadata about the bid request");
    addField("location", &BidRequest::location,
             "location of user");
    addField("segments", &BidRequest::segments,
             "segments active for user");
    addField("restrictions", &BidRequest::restrictions,
             "restrictions active for bid request");
    addField("userIds", &BidRequest::userIds, "User IDs for this user");
    addField("imp", &BidRequest::imp, "Ad imp in this request");
    addField("spots", &BidRequest::imp, "Ad imp in this request");
    addField("site", &BidRequest::site, "OpenRTB site object");
    addField("app", &BidRequest::app, "OpenRTB app object");
    addField("device", &BidRequest::device, "OpenRTB device object");
    addField("user", &BidRequest::user, "OpenRTB user object");
    addField("unparseable", &BidRequest::unparseable, "Unparseable fields are stored here");
    addField("bidCurrency", &BidRequest::bidCurrency, "Currency we're bidding in");
    addField("ext", &BidRequest::ext, "OpenRTB ext object");
}

} // namespace Datacratic


namespace RTBKIT {

using Datacratic::jsonDecode;
using Datacratic::jsonEncode;

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

bool isEmpty(const Datacratic::UnicodeString &str)
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

inline void addIfNotEmpty(Json::Value & obj, const std::string & key,
                          const std::vector<CurrencyCode> & curr)
{
    if (curr.empty())
        return;

    for (unsigned i = 0;  i < curr.size();  ++i)
        obj[key][i] = Amount::getCurrencyStr(curr[i]);
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

struct FormatSetDescription
    : public ValueDescriptionT<FormatSet> {

    virtual void parseJsonTyped(FormatSet * val,
                                JsonParsingContext & context) const
    {
        val->fromJson(context.expectJson());
    }

    virtual void printJsonTyped(const FormatSet * val,
                                JsonPrintingContext & context) const
    {
        context.writeJson(val->toJson());
    }

    virtual bool isDefaultTyped(const FormatSet * val) const
    {
        return val->empty();
    }
};

ValueDescriptionT<RTBKIT::FormatSet> *
getDefaultDescription(RTBKIT::FormatSet *)
{
    return new FormatSetDescription();
}



/*****************************************************************************/
/* LOCATION                                                                  */
/*****************************************************************************/

Datacratic::UnicodeString
Location::
fullLocationString() const
{
    Datacratic::UnicodeString result(countryCode + ":" + regionCode + ":") ;
    result += cityName ;
    result += ":";
    result += postalCode;
    result += ":";
    result += boost::lexical_cast<string>(dma);
    result += ":";
    result += boost::lexical_cast<string>(metro);
    return result;
    //Datacratic::UnicodeString result(countryCode +":"+ regionCode +":" +
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
    addIfNotEmpty(result, "cityName",     cityName.utf8String());
    addIfNotEmpty(result, "postalCode",   postalCode.utf8String());
    addIfNotEmpty(result, "dma",          dma, -1);
    addIfNotEmpty(result, "metro",        metro, -1);
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
        else if (it.memberName() == "metro")
            result.metro = it->asInt();
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

#if 0
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
#endif

void
AdSpot::
fromJson(const Json::Value & val)
{
    *this = AdSpot();

    // Parse openrtb
    static DefaultDescription<AdSpot> desc;
    StructuredJsonParsingContext context(val);

    // Rather than barf on unknown fields, for forwards compatibility we put them
    // in the unparseable array via this function
    auto onUnknownField = [&] (const ValueDescription*)
        {
            cerr << "(adspot)got unknown field " << context.printPath()
            << context.expectJson() << endl;
            
#if 0
            std::function<Json::Value & (int, Json::Value &)> getEntry
            = [&] (int n, Json::Value & curr) -> Json::Value &
            {
                if (n == context.path.size())
                    return curr;
                else if (context.path[n].index != -1)
                    return getEntry(n + 1, curr[context.path[n].index]);
                else return getEntry(n + 1, curr[context.path[n].key]);
            };

            getEntry(0, unparseable[key]) = context.expectJson();
#endif
        };

    context.onUnknownFieldHandlers.push_back(onUnknownField);

    desc.parseJsonTyped(this, context);
    
    return;

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
        const Json::Value & rpj = val["reservePrice"];
        if (rpj.isNumeric()) {
            reservePrice = USD_CPM(rpj.asDouble());
        }
        else if (!rpj.isNull()) {
            reservePrice = Amount::fromJson(rpj);
        }
        if (val.isMember("position")) {
            auto & pj = val["position"];
            if (pj.isString()) {
                string s = pj.asString();
                if (s == "NONE" || s == "none") {
                    position.val = OpenRTB::AdPosition::UNKNOWN;
                }
                else if (s == "ABOVE_FOLD" || s == "above") {
                    position.val = OpenRTB::AdPosition::ABOVE;
                }
                else if (s == "BELOW_FOLD" || s == "below") {
                    position.val = OpenRTB::AdPosition::BELOW;
                }
            }
            else if (pj.isIntegral()) {
                position.val = pj.asInt();
            }
            else throw ML::Exception("can't parse position " + val["position"].toString());
            //position = stringToPosition(val["position"].asString());
        }
        else
            position.val = -1;//OpenRTB::AdPosition::UNKNOWN;
    } catch (const std::exception & exc) {
        cerr << "parsing AdSpot " << val << ": " << exc.what() << endl;
        throw;
    }
}

Json::Value
AdSpot::
toJson() const
{
    static DefaultDescription<AdSpot> desc;
    StructuredJsonPrintingContext context;
    desc.printJsonTyped(this, context);
    return std::move(context.output);
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

void
AdSpot::
serialize(ML::DB::Store_Writer & store) const
{
    unsigned char version = 2;
    store << version << toJson().toString();
}

void
AdSpot::
reconstitute(ML::DB::Store_Reader & store)
{
    unsigned char version;
    store >> version;
    if (version != 2)
        throw ML::Exception("unknown AdSpot serialization version");
    string s;
    store >> s;
    fromJson(Json::parse(s));
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

Json::Value
UserIds::
toJsonArray() const
{
    Json::Value result(Json::arrayValue);
    for (auto it = begin(), end = this->end();  it != end;  ++it)
        result.append(it->second.toString());
    return result;
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

struct UserIdsDescription
    : public ValueDescriptionT<UserIds> {

    virtual void parseJsonTyped(UserIds * val,
                                JsonParsingContext & context) const
    {
        auto onMember = [&] ()
            {
                string key = context.path.fieldName();
                Id value(context.expectStringAscii());
                val->add(value, key);
            };
        
        context.forEachMember(onMember);
    }

    virtual void printJsonTyped(const UserIds * val,
                                JsonPrintingContext & context) const
    {
        context.startObject();

        for (auto & id: *val) {
            context.startMember(id.first);
            context.writeString(id.second.toString());
        }

        context.endObject();
    }

    virtual bool isDefaultTyped(const UserIds * val) const
    {
        return val->empty();
    }

};

ValueDescriptionT<RTBKIT::UserIds> *
getDefaultDescription(RTBKIT::UserIds *)
{
    return new UserIdsDescription();
}


/*****************************************************************************/
/* BID REQUEST                                                               */
/*****************************************************************************/

void
BidRequest::
sortAll()
{
    restrictions.sortAll();
    segments.sortAll();
}

template<typename T>
void toJsonValue(Json::Value & v, const T & val)
{
    static DefaultDescription<T> desc;
    StructuredJsonPrintingContext printContext;
    desc.printJson(&val, printContext);
    v = std::move(printContext.output);
}

Json::Value
BidRequest::
toJson() const
{
    Json::Value result;
    result["!!CV"] = "RTBKIT-JSON-1.0";
    result["id"] = auctionId.toString();
    result["timestamp"] = timestamp.secondsSinceEpoch();
    addIfNotEmpty(result, "isTest", isTest, false);
    addIfNotEmpty(result, "url", url);
    addIfNotEmpty(result, "ipAddress", ipAddress);
    addIfNotEmpty(result, "userAgent", userAgent.utf8String());
    addIfNotEmpty(result, "language", language.utf8String());
    addIfNotEmpty(result, "protocolVersion", protocolVersion);
    addIfNotEmpty(result, "exchange", exchange);
    addIfNotEmpty(result, "provider", provider);
    addIfNotEmpty(result, "meta", meta);
    addIfNotEmpty(result, "unparseable", unparseable);
    if (!bidCurrency.empty())
        result["bidCurrency"] = jsonEncode(bidCurrency);
    if (blockedCategories.size())
    	result["blockedCategories"] = jsonEncode(blockedCategories);
    if (site) {
        toJsonValue(result["site"], *site);
    }
    if (app)
        toJsonValue(result["app"], *app);
    if (device)
        toJsonValue(result["device"], *device);
    if (user)
        toJsonValue(result["user"], *user);

    if (!winSurcharges.empty())
        result["winSurcharges"] = winSurcharges.toJson();

    result["location"] = location.toJson();

    if (!imp.empty()) {
        for (unsigned i = 0;  i < imp.size();  ++i) {
            result["imp"][i] = imp[i].toJson();
            result["spots"][i] = imp[i].toJson();
        }
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
    static const DefaultDescription<BidRequest> BidRequestDesc;
    
    std::ostringstream stream;
    StreamJsonPrintingContext context(stream);
    BidRequestDesc.printJson(this, context);
    return stream.str();

    //return boost::trim_copy(toJson().toString());
}

template<typename T>
void fromJsonOptional(const Json::Value & val,
                      std::unique_ptr<T> & ptr,
                      Json::Value & unparseable,
                      std::string key)
{
    static DefaultDescription<T> desc;
    StructuredJsonParsingContext context(val);
    std::unique_ptr<T> res(new T());

    // Rather than barf on unknown fields, for forwards compatibility we put them
    // in the unparseable array via this function
    auto onUnknownField = [&] (const ValueDescription*)
        {
            std::function<Json::Value & (int, Json::Value &)> getEntry
            = [&] (int n, Json::Value & curr) -> Json::Value &
            {
                if (n == context.path.size())
                    return curr;
                else if (context.path[n].index != -1)
                    return getEntry(n + 1, curr[context.path[n].index]);
                else return getEntry(n + 1, curr[context.path[n].fieldName()]);
            };

            getEntry(0, unparseable[key]) = context.expectJson();
        };

    context.onUnknownFieldHandlers.push_back(onUnknownField);

    desc.parseJson(res.get(), context);

    ptr.swap(res);
}

BidRequest
BidRequest::
createFromJson(const Json::Value & json)
{
    BidRequest result;

    string canonicalVersion;

    //cerr << "parsing " << json << endl;

    for (auto it = json.begin(), end = json.end(); it != end;  ++it) {

        //cerr << "got member " << it.memberName() << endl;

        if (it.memberName() == "!!CV") {
            canonicalVersion = it->asString();
            if (canonicalVersion != "0.1" && canonicalVersion != "RTBKIT-JSON-1.0")
                throw ML::Exception("can't parse BidRequest with CV "
                                    + canonicalVersion);
        }
        else if (it.memberName() == "id")
            result.auctionId.parse(it->asString());
        else if (it.memberName() == "timestamp")
            result.timestamp = Date::fromSecondsSinceEpoch(it->asDouble());
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
            result.meta["creative"] = *it;
        else if (it.memberName() == "location")
            result.location = Location::createFromJson(*it);
        else if (it.memberName() == "segments")
            result.segments = SegmentsBySource::createFromJson(*it);
        else if (it.memberName() == "restrictions")
            result.restrictions = SegmentsBySource::createFromJson(*it);
        else if (it.memberName() == "userIds")
            result.userIds = UserIds::createFromJson(*it);
        else if (it.memberName() == "imp" || it.memberName() == "spots") {
            const Json::Value & json = *it;
            if (!json.empty() && !json.isArray())
                throw ML::Exception("imp is not an array");
            result.imp.clear();
            for (unsigned i = 0;  i < json.size();  ++i) {
                result.imp.push_back(AdSpot::createFromJson(json[i]));
            }
        }
        else if (it.memberName() == "site") {
            fromJsonOptional(*it, result.site, result.unparseable, "site");
        }
        else if (it.memberName() == "app") {
            fromJsonOptional(*it, result.app, result.unparseable, "app");
        }
        else if (it.memberName() == "device") {
            fromJsonOptional(*it, result.device, result.unparseable, "device");
        }
        else if (it.memberName() == "user") {
            fromJsonOptional(*it, result.user, result.unparseable, "user");
        }
        else if (it.memberName() == "unparseable")
            result.unparseable = *it;
        else if (it.memberName() == "bidCurrency")
            jsonDecode(*it, result.bidCurrency);
        else throw ML::Exception("unknown canonical bid request field "
                                 + it.memberName());
    }

    return result;
}

namespace {

static const DefaultDescription<BidRequest> BidRequestDesc;

struct CanonicalParser {
    
    static BidRequest * parse(const std::string & str)
    {
#if 0 // old and slow
        auto j = Json::parse(str);
        return new BidRequest(BidRequest::createFromJson(j));
#endif

        //cerr << "parsing " << str << endl;
        StreamingJsonParsingContext context;
        context.init("bid request", str.c_str(), str.size());
        auto_ptr<BidRequest> result(new BidRequest());
        BidRequestDesc.parseJsonTyped(result.get(), context);

        //cerr << "result->url = " << result->url << endl;
        //cerr << "result->userAgent = " << result->userAgent << endl;

        return result.release();
    }
};

struct AtInit {
    AtInit()
    {
        PluginInterface<BidRequest>::registerPlugin("recoset", CanonicalParser::parse);
        PluginInterface<BidRequest>::registerPlugin("datacratic", CanonicalParser::parse);
        PluginInterface<BidRequest>::registerPlugin("rtbkit", CanonicalParser::parse);
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
    {
        return CanonicalParser::parse(bidRequest);
    }
    Parser parser = PluginInterface<BidRequest>::getPlugin(source);

    //cerr << "got parser for source " << source << endl;

    auto result = parser(bidRequest);

    if (false) {
        cerr << bidRequest << endl;
        StreamJsonPrintingContext context(cerr);
        BidRequestDesc.printJsonTyped(result, context);
    }

    return result;
}

BidRequest *
BidRequest::
parse(const std::string & source, const Datacratic::UnicodeString & bidRequest)
{
    return BidRequest::parse(source, bidRequest.utf8String());
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
    unsigned char version = 2;
    store << version << auctionId << language << protocolVersion
          << exchange << provider << timestamp << isTest
          << location << userIds << imp << url << ipAddress << userAgent
          << restrictions << segments << meta
          << winSurcharges;
}

void
BidRequest::
reconstitute(ML::DB::Store_Reader & store)
{
    using namespace ML::DB;

    unsigned char version;

    store >> version;

    if (version != 2)
        throw ML::Exception("problem reconstituting BidRequest: "
                            "invalid version");

    store >> auctionId >> language >> protocolVersion
          >> exchange >> provider >> timestamp >> isTest
          >> location >> userIds >> imp >> url >> ipAddress >> userAgent
          >> restrictions >> segments >> meta >> winSurcharges;
}

} // namespace RTBKIT


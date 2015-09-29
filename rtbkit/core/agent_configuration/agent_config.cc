/* rtb_agent_config.cc
   Jeremy Barnes, 24 March 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

*/

#include "agent_config.h"
#include "jml/arch/exception.h"
#include "jml/utils/string_functions.h"
#include <boost/lexical_cast.hpp>
#include "rtbkit/common/auction.h"
#include "rtbkit/core/router/router_types.h"
#include "rtbkit/common/exchange_connector.h"

#define CRYPTOPP_ENABLE_NAMESPACE_WEAK 1
#include "crypto++/md5.h"


using namespace std;
using namespace ML;

namespace RTBKIT {


/*****************************************************************************/
/* CREATIVE                                                                  */
/*****************************************************************************/

Creative::
Creative(int width, int height, std::string name, int id, const std::string dealId)
    : format(width, height), name(name), id(id), dealId(dealId),
      fees(NullFees::createNullFees()), type(Type::Image)
{
}

Creative
Creative::
image(int width, int height, std::string name, int id, std::string dealId)
{
    Creative creative(width, height, name, id, dealId);
    creative.type = Creative::Type::Image;
    return creative;
}

Creative
Creative::
video(int width, int height, uint32_t duration, uint64_t bitrate, std::string name, int id, std::string dealId)
{
    Creative creative(width, height, name, id, dealId);
    creative.duration = duration;
    creative.bitrate = bitrate;
    creative.type = Creative::Type::Video;
    return creative;
}

void
Creative::
fromJson(const Json::Value & val)
{
    if (val.isMember("format")) {
        format.fromJson(val["format"]);
    }
    else {
        format.width = val["width"].asInt();
        format.height = val["height"].asInt();
    }
    name = val["name"].asString();

    id = -1;
    if (val.isMember("id"))
        id = val["id"].asInt();
    if (id == -1)
        throw ML::Exception("creatives require an ID to be specified");

    if (val.isMember("dealId"))
        dealId = val["dealId"].asString();

    providerConfig = val["providerConfig"];

    languageFilter.fromJson(val["languageFilter"], "languageFilter");
    locationFilter.fromJson(val["locationFilter"], "locationFilter");
    exchangeFilter.fromJson(val["exchangeFilter"], "exchangeFilter");

    if (val.isMember("fees")) {
        fees = Fees::createFees(val["fees"]);
    } else {
        fees = NullFees::createNullFees();
    }

    if (val.isMember("segmentFilter")){
        Json::Value segs = val["segmentFilter"];
        for (auto jt = segs.begin(), jend = segs.end(); jt != jend;  ++jt) {
            string source = jt.memberName();
            segments[source].fromJson(*jt);
        }
    }

    if (val.isMember("type")) {
        const std::string type_ = val["type"].asString();
        if (type_ == "video") {
            duration = val["duration"].asUInt();
            bitrate = val["bitrate"].asUInt();
            type = Type::Video;
        } else if (type_ == "image") {
            type = Type::Image;
        }
        else {
            throw ML::Exception("Unknown type '%s'", type_.c_str());
        }
    } else {
        // For backward compatibility, take 'Image' by default
        type = Type::Image;
    }

}

Json::Value
Creative::
toJson() const
{
    Json::Value result;
    result["type"] = typeString();
    result["format"] = format.toJson();
    result["name"] = name;
    if (id != -1)
        result["id"] = id;
    if (!languageFilter.empty())
        result["languageFilter"] = languageFilter.toJson();
    if (!locationFilter.empty())
        result["locationFilter"] = locationFilter.toJson();
    if (!exchangeFilter.empty())
        result["exchangeFilter"] = exchangeFilter.toJson();
    if (!providerConfig.isNull())
        result["providerConfig"] = providerConfig;
    if (!segments.empty()) {
        Json::Value segmentInfo;
        for (auto it = segments.begin(), end = segments.end();
             it != end;  ++it) {
            segmentInfo[it->first] = it->second.toJson();
        }
        result["segmentFilter"] = segmentInfo;
    }
    if (!dealId.empty())
        result["dealId"] = dealId;

    if (fees) {
        result["fees"] = fees->toJson();
    }

    if (type == Type::Video) {
        result["duration"] = duration;
        result["bitrate"] = bitrate;
    }

    return result;
}

const Creative Creative::sampleWS
    (160, 600, "LeaderBoard", 0);
const Creative Creative::sampleBB
    (300, 250, "BigBox", 1);
const Creative Creative::sampleLB
    (728, 90,  "LeaderBoard", 2);

bool
Creative::
compatible(const AdSpot & adspot) const
{
    return ((format.width == 0 && format.height == 0)
            || adspot.formats.empty() // if no format was specified in bid request
            || adspot.formats.compatible(format));
}

bool
Creative::
biddable(const std::string & exchange,
         const std::string & protocolVersion) const
{
    return true;
}

bool
Creative::
isImage() const {
    return type == Type::Image;
}

bool
Creative::
isVideo() const {
    return type == Type::Video;
}

std::string
Creative::
typeString() const {
    switch (type) {
    case Type::Image:
        return "image";
    case Type::Video:
        return "video";
    }

    return "unknown";
}

Json::Value jsonPrint(const Creative & c)
{
    return c.toJson();
}

void
Creative::SegmentInfo::
fromJson(const Json::Value & json)
{
    for (auto it = json.begin(), end = json.end(); it != end;  ++it) {
        if (it.memberName() == "excludeIfNotPresent") {
            excludeIfNotPresent = it->asBool();
            continue;
        }
        const Json::Value & val = *it;

        if (it.memberName() == "include") {
            include = SegmentList::createFromJson(val);
            include.sort();
        }
        else if (it.memberName() == "exclude") {
            exclude = SegmentList::createFromJson(val);
            exclude.sort();
        }
        else {
            throw Exception("segmentFilter has invalid key: %s",
                            it.memberName().c_str());
        }
    }
}

Json::Value
Creative::SegmentInfo::
toJson() const
{
    Json::Value result;
    if (!include.empty())
        result["include"] = include.toJson();
    if (!exclude.empty())
        result["exclude"] = exclude.toJson();
    result["excludeIfNotPresent"] = excludeIfNotPresent;

    return result;
}

IncludeExcludeResult
Creative::SegmentInfo::
process(const SegmentList & segments) const
{
    if (segments.empty())
        return IE_NO_DATA;

    if (!include.empty() && !include.match(segments))
        return IE_NOT_INCLUDED;

    if (exclude.match(segments))
        return IE_EXCLUDED;

    return IE_PASSED;
}

/*****************************************************************************/
/* USER PARTITION                                                            */
/*****************************************************************************/

UserPartition::
UserPartition()
    : hashOn(NONE),
      modulus(1),
      includeRanges(1, Interval(0, 1))
{
}

void
UserPartition::
swap(UserPartition & other)
{
    std::swap(hashOn, other.hashOn);
    std::swap(modulus, other.modulus);
    includeRanges.swap(other.includeRanges);
}

void
UserPartition::
clear()
{
    hashOn = NONE;
    modulus = 1;
    includeRanges.clear();
}

Json::Value
UserPartition::Interval::
toJson() const
{
    Json::Value result;
    result[0u] = first;
    result[1] = last;
    return result;
}

void
UserPartition::
fromJson(const Json::Value & json)
{
    UserPartition newPartition;
    newPartition.clear();

    for (auto it = json.begin(), end = json.end(); it != end;  ++it) {
        if (it.memberName() == "hashOn") {
            string name = it->asString();
            if (name == "null") newPartition.hashOn = NONE;
            else if (name == "random") newPartition.hashOn = RANDOM;
            else if (name == "exchangeId") newPartition.hashOn = EXCHANGEID;
            else if (name == "providerId") newPartition.hashOn = PROVIDERID;
            else if (name == "ipua") newPartition.hashOn = IPUA;
            else throw Exception("unknown hashOn value %s", name.c_str());
        }
        else if (it.memberName() == "modulus") {
            newPartition.modulus = it->asInt();
        }
        else if (it.memberName() == "includeRanges") {
            const Json::Value & arr = *it;
            for (unsigned i = 0;  i < arr.size();  ++i) {
                const Json::Value & ival = arr[i];
                if (ival.size() != 2)
                    throw Exception("bad interval");
                int first = ival[0u].asInt();
                int last = ival[1].asInt();
                newPartition.includeRanges.push_back(Interval(first, last));
            }
        }
        else throw Exception("unknown user partition option: %s",
                             it.memberName().c_str());
    }

    swap(newPartition);
}

Json::Value
UserPartition::
toJson() const
{
    Json::Value result;
    string ho;
    switch (hashOn) {
    case NONE: ho = "null";  break;
    case RANDOM: ho = "random";  break;
    case EXCHANGEID: ho = "exchangeId";  break;
    case PROVIDERID: ho = "providerId";  break;
    case IPUA: ho = "ipua"; break;
    default:
        throw ML::Exception("unknown hashOn");
    }
    result["hashOn"] = ho;
    result["modulus"] = modulus;
    for (unsigned i = 0;  i < includeRanges.size();  ++i)
        result["includeRanges"][i] = includeRanges[i].toJson();
    
    return result;
}


/******************************************************************************/
/* AUGMENTATION INFO                                                          */
/******************************************************************************/

Json::Value
AugmentationConfig::
toJson() const
{
    Json::Value result;

    if (!config.isNull()) result["config"] = config;
    if (!filters.empty()) result["filters"] = filters.toJson();
    if (required) result["required"] = true;

    return result;
}

void
AugmentationConfig::
fromJson(const Json::Value& json)
{
    const auto& members = json.getMemberNames();
    for (const auto& m : members) {
        const Json::Value& val = json[m];

        if      (m == "config") config = val;
        else if (m == "filters") filters.fromJson(val, "augmentor.filters");
        else if (m == "required") required = val.asBool();

        else ExcCheck(false, "Unknown AugmentorInfo field: " + m);
    }
}

AugmentationConfig
AugmentationConfig::
createFromJson(const Json::Value& json, const std::string& name)
{
    AugmentationConfig info(name);
    info.fromJson(json);
    return info;
}



/*****************************************************************************/
/* AGENT CONFIG                                                             */
/*****************************************************************************/

AgentConfig::
AgentConfig()
    : externalId(0),
      external(false),
      test(false),
      roundRobinWeight(0),
      bidProbability(1.0), minTimeAvailableMs(5.0),
      maxInFlight(100),
      blacklistType(BL_OFF),
      blacklistScope(BL_ACCOUNT), blacklistTime(15.0),
      bidControlType(BC_RELAY), fixedBidCpmInMicros(0),
      winFormat(BRF_FULL),
      lossFormat(BRF_LIGHTWEIGHT),
      errorFormat(BRF_LIGHTWEIGHT)
{
    addAugmentation("random");
}

void
AgentConfig::
parse(const std::string & jsonStr)
{
    Json::Value val = Json::parse(jsonStr);
    fromJson(val);
}

void
AgentConfig::SegmentInfo::
fromJson(const Json::Value & json)
{
    for (auto it = json.begin(), end = json.end(); it != end;  ++it) {
        if (it.memberName() == "excludeIfNotPresent") {
            excludeIfNotPresent = it->asBool();
            continue;
        }
        
        const Json::Value & val = *it;

        if (it.memberName() == "include") {
            include = SegmentList::createFromJson(val);
            include.sort();
        }
        else if (it.memberName() == "exclude") {
            exclude = SegmentList::createFromJson(val);
            exclude.sort();
        }
        else if (it.memberName() == "applyToExchanges")
            applyToExchanges.fromJson(val, "segmentFilter applyToExchanges");
        else {
            throw Exception("segmentFilter has invalid key: %s",
                            it.memberName().c_str());
        }
    }
}

IncludeExcludeResult
AgentConfig::SegmentInfo::
process(const SegmentList & segments) const
{
    if (segments.empty()) 
        return IE_NO_DATA;

    if (!include.empty() && !include.match(segments))
        return IE_NOT_INCLUDED;
    
    if (exclude.match(segments))
        return IE_EXCLUDED;
    
    return IE_PASSED;
}

AgentConfig::HourOfWeekFilter::
HourOfWeekFilter()
{
    for (unsigned i = 0;  i < 168;  ++i)
        hourBitmap.set(i);
}

bool
AgentConfig::HourOfWeekFilter::
isDefault() const
{
    return hourBitmap.all();
}

void
AgentConfig::HourOfWeekFilter::
fromJson(const Json::Value & val)
{
    string s = val["hourlyBitmapSundayMidnightUtc"].asString();
    if (s.length() != 168)
        throw ML::Exception("Hourly bitmap string needs 168 characters");
    for (unsigned i = 0;  i < 168;  ++i) {
        if (s[i] == '0')
            hourBitmap[i] = 0;
        else if (s[i] == '1')
            hourBitmap[i] = 1;
        else throw ML::Exception("Hourly bitmap must contain only 0 or 1 "
                                 "characters");
    }
}

Json::Value
AgentConfig::HourOfWeekFilter::
toJson() const
{
    string bitmap;
    for (unsigned i = 0;  i < 168;  ++i)
        bitmap += '0' + int(hourBitmap[i] != 0);
    Json::Value result;
    result["hourlyBitmapSundayMidnightUtc"] = bitmap;
    return result;
}

Json::Value toJson(BidResultFormat fmt)
{
    switch (fmt) {
    case BRF_FULL:         return "full";
    case BRF_LIGHTWEIGHT:  return "lightweight";
    case BRF_NONE:         return "none";
    default:
        throw ML::Exception("unknown BidResultFormat");
    }
}

void fromJson(BidResultFormat & fmt, const Json::Value & j)
{
    string s = lowercase(j.asString());
    if (s == "full")
        fmt = BRF_FULL;
    else if (s == "lightweight")
        fmt = BRF_LIGHTWEIGHT;
    else if (s == "none")
        fmt = BRF_NONE;
    else throw ML::Exception("unknown BidResultFormat " + s + ": accepted "
                             "full, lightweight, none");
}

void
AgentConfig::
fromJson(const Json::Value & json)
{
    *this = createFromJson(json);
}

AgentConfig
AgentConfig::
createFromJson(const Json::Value & json)
{
    AgentConfig newConfig;
    newConfig.augmentations.clear();

    for (auto it = json.begin(), end = json.end(); it != end;  ++it) {
        //cerr << "parsing " << it.memberName() << " with value " << *it << endl;

        if (it.memberName() == "account") {
            newConfig.account = AccountKey::fromJson(*it);
        }
        else if (it.memberName() == "test") {
            newConfig.test = it->asBool();
        }
        else if (it.memberName() == "external") {
            newConfig.external = it->asBool();
        }
        else if (it.memberName() == "externalId") {
            newConfig.externalId = it->asUInt();
        }
        else if (it.memberName() == "requiredIds") {
            if (!it->isArray())
                throw Exception("requiredIds must be an array of string");
            for (unsigned i = 0;  i < it->size();  ++i) {
                const Json::Value & val = (*it)[i];
                newConfig.requiredIds.push_back(val.asString());
            }
        }
        else if (it.memberName() == "roundRobin") {
            for (auto jt = it->begin(), jend = it->end();
                 jt != jend;  ++jt) {
                if (jt.memberName() == "group")
                    newConfig.roundRobinGroup = jt->asString();
                else if (jt.memberName() == "weight")
                    newConfig.roundRobinWeight = jt->asInt();
                else throw Exception("roundRobin group had unknown key "
                                     + jt.memberName());
            }
        }
        else if (it.memberName() == "creatives") {
            //cerr << "doing " << it->size() << " creatives" << endl;

            newConfig.creatives.resize(it->size());

            for (unsigned i = 0;
                 i < newConfig.creatives.size();  ++i) {
                try {
                    newConfig.creatives[i].fromJson((*it)[i]);
                } catch (const std::exception & exc) {
                    throw Exception("parsing creative %d: %s",
                                    i, exc.what());
                }
            }

            //cerr << "got " << newConfig.creatives.size() << " creatives" << endl;
        }
        else if (it.memberName() == "bidProbability") {
            newConfig.bidProbability = it->asDouble();
            if (newConfig.bidProbability < 0 || newConfig.bidProbability > 1.0)
                throw Exception("bidProbability %f not between 0 and 1",
                                newConfig.bidProbability);
        }
        else if (it.memberName() == "minTimeAvailableMs") {
            newConfig.minTimeAvailableMs = it->asDouble();
            if (newConfig.minTimeAvailableMs < 0)
                throw Exception("minTimeAvailableMs %f should be not less than 0",
                                newConfig.minTimeAvailableMs);
        }
        else if (it.memberName() == "maxInFlight") {
            newConfig.maxInFlight = it->asInt();
            if (newConfig.maxInFlight < 0)
                throw Exception("maxInFlight has wrong value: %d",
                                newConfig.maxInFlight);
        }
        else if (it.memberName() == "bidderInterface")
            newConfig.bidderInterface = it->asString();
        else if (it.memberName() == "userPartition") {
            newConfig.userPartition.fromJson(*it);
        }
        else if (it.memberName() == "urlFilter")
            newConfig.urlFilter.fromJson(*it, "urlFilter");
        else if (it.memberName() == "hostFilter")
            newConfig.hostFilter.fromJson(*it, "hostFilter");
        else if (it.memberName() == "locationFilter")
            newConfig.locationFilter.fromJson(*it, "locationFilter");
        else if (it.memberName() == "languageFilter")
            newConfig.languageFilter.fromJson(*it, "languageFilter");
        else if (it.memberName() == "exchangeFilter")
            newConfig.exchangeFilter.fromJson(*it, "exchangeFilter");
        else if (it.memberName() == "latLongDevFilter")
            newConfig.latLongDevFilter.fromJson(*it);
        else if (it.memberName() == "segmentFilter") {
            for (auto jt = it->begin(), jend = it->end();
                 jt != jend;  ++jt) {
                string source = jt.memberName();
                newConfig.segments[source].fromJson(*jt);
            }
        }
        else if (it.memberName() == "tagFilter") {
            newConfig.tagFilter.fromJson(*it);
        }
        else if (it.memberName() == "foldPositionFilter") {
            newConfig.foldPositionFilter.fromJson(*it, "foldPositionFilter");
        }
        else if (it.memberName() == "hourOfWeekFilter") {
            newConfig.hourOfWeekFilter.fromJson(*it);
        }
        else if (it.memberName() == "augmentations") {
            ExcCheckEqual(it->type(), Json::objectValue,
                    "augment must be an object of augmentor name to config");

            for (auto jt = (*it).begin(), end = (*it).end(); jt != end; ++jt) {
                newConfig.augmentations.emplace_back(
                        AugmentationConfig::createFromJson(*jt, jt.memberName()));
            }
        }
        else if (it.memberName() == "blacklist") {
            for (auto jt = it->begin(), jend = it->end();
                 jt != jend;  ++jt) {
                const Json::Value & val = *jt;
                if (jt.memberName() == "type") {
                    if (val.isNull())
                        newConfig.blacklistType = BL_OFF;
                    else {
                        string s = ML::lowercase(val.asString());
                        if (s == "off")
                            newConfig.blacklistType = BL_OFF;
                        else if (s == "user")
                            newConfig.blacklistType = BL_USER;
                        else if (s == "user_site")
                            newConfig.blacklistType = BL_USER_SITE;
                        else throw Exception("invalid blacklist type " + s);
                    }
                }
                else if (jt.memberName() == "time") {
                    newConfig.blacklistTime = val.asDouble();
                }
                else if (jt.memberName() == "scope") {
                    string s = ML::lowercase(val.asString());
                    if (s == "agent")
                        newConfig.blacklistScope = BL_AGENT;
                    else if (s == "account")
                        newConfig.blacklistScope = BL_ACCOUNT;
                    else throw Exception("invalid blacklist scope " + s);
                }
                else throw Exception("blacklist has invalid key: %s",
                                     jt.memberName().c_str());
            }
        }
        else if (it.memberName() == "visits") {
            for (auto jt = it->begin(), jend = it->end();
                 jt != jend;  ++jt) {
                const Json::Value & val = *jt;
                if (jt.memberName() == "channels") {
                    newConfig.visitChannels = SegmentList::createFromJson(val);
                }
                else if (jt.memberName() == "includeUnmatched") {
                    newConfig.includeUnmatchedVisits = val.asBool();
                }
                else throw Exception("visits has invalid key: %s",
                                     jt.memberName().c_str());
            }
        }
        else if (it.memberName() == "bidControl") {
            for (auto jt = it->begin(), jend = it->end();
                 jt != jend;  ++jt) {
                const Json::Value & val = *jt;
                if (jt.memberName() == "type") {
                    string s = ML::lowercase(val.asString());
                    if (s == "relay")
                        newConfig.bidControlType = BC_RELAY;
                    else if (s == "relay_fixed")
                        newConfig.bidControlType = BC_RELAY_FIXED;
                    else if (s == "fixed")
                        newConfig.bidControlType = BC_FIXED;
                    else throw Exception("invalid bid control value " + s);
                }
                else if (jt.memberName() == "fixedBidCpmInMicros") {
                    newConfig.fixedBidCpmInMicros = val.asInt();
                }
                else throw Exception("bidControl has invalid key: %s",
                                     jt.memberName().c_str());
            }
        }
        else if (it.memberName() == "providerConfig") {
            newConfig.providerConfig = *it;
        }
        else if (it.memberName() == "winFormat") {
            RTBKIT::fromJson(newConfig.winFormat, *it);
        }
        else if (it.memberName() == "lossFormat") {
            RTBKIT::fromJson(newConfig.lossFormat, *it);
        }
        else if (it.memberName() == "errorFormat") {
            RTBKIT::fromJson(newConfig.errorFormat, *it);
        }
        else if (it.memberName() == "ext") {
            newConfig.ext = *it;
        }
        else throw Exception("unknown config option: %s",
                             it.memberName().c_str());
    }

    if (newConfig.account.empty())
        throw Exception("each agent must have an account specified");
    
    if (newConfig.creatives.empty())
        throw Exception("can't configure a agent with no creatives");

    return newConfig;
}

Json::Value
AgentConfig::SegmentInfo::
toJson() const
{
    Json::Value result;
    if (!include.empty())
        result["include"] = include.toJson();
    if (!exclude.empty())
        result["exclude"] = exclude.toJson();
    result["excludeIfNotPresent"] = excludeIfNotPresent;
    if (!applyToExchanges.empty())
        result["applyToExchanges"] = applyToExchanges.toJson();
    return result;
}

Json::Value
AgentConfig::
toJson(bool includeCreatives) const
{
    Json::Value result;
    result["account"] = account.toJson();
    result["externalId"] = externalId;
    result["external"] = external;
    result["test"] = test;
    if (roundRobinGroup != "") {
        result["roundRobin"]["group"] = roundRobinGroup;
        if (roundRobinWeight != 0)
            result["roundRobin"]["weight"] = roundRobinWeight;
    }
    if (bidProbability != 1.0)
        result["bidProbability"] = bidProbability;
    result["minTimeAvailableMs"] = minTimeAvailableMs;
    if (maxInFlight != 100)
        result["maxInFlight"] = maxInFlight;

    if (!bidderInterface.empty())
        result["bidderInterface"] = bidderInterface;

    if (!urlFilter.empty())
        result["urlFilter"] = urlFilter.toJson();
    if (!hostFilter.empty())
        result["hostFilter"] = hostFilter.toJson();
    if (!locationFilter.empty())
        result["locationFilter"] = locationFilter.toJson();
    if (!languageFilter.empty())
        result["languageFilter"] = languageFilter.toJson();
    if (!exchangeFilter.empty())
        result["exchangeFilter"] = exchangeFilter.toJson();
    if (!latLongDevFilter.empty())
        result["latLongDevFilter"] = latLongDevFilter.toJson();
    if (!requiredIds.empty()) {
        for (unsigned i = 0;  i < requiredIds.size();  ++i)
            result["requiredIds"][i] = requiredIds[i];
    }
    if (!userPartition.empty())
        result["userPartition"] = userPartition.toJson();
    if (!creatives.empty() && includeCreatives)
        result["creatives"] = collectionToJson(creatives, JsonPrint());
    else if (!creatives.empty()) {
        Json::Value creativeInfo;
        for (unsigned i = 0;  i < creatives.size();  ++i)
            creativeInfo[i] = creatives[i].format.print();
        result["creatives"] = creativeInfo;
    }
    if (!segments.empty()) {
        Json::Value segmentInfo;
        for (auto it = segments.begin(), end = segments.end();
             it != end;  ++it) {
            segmentInfo[it->first] = it->second.toJson();
        }
        result["segmentFilter"] = segmentInfo;
    }

    if (!augmentations.empty()) {
        Json::Value aug;
        for (unsigned i = 0;  i < augmentations.size();  ++i)
            aug[augmentations[i].name] = augmentations[i].toJson();
        result["augmentations"] = aug;
    }
    if (!hourOfWeekFilter.isDefault()) {
        result["hourOfWeekFilter"] = hourOfWeekFilter.toJson();
    }

    if (!foldPositionFilter.empty()) {
        result["foldPositionFilter"] = foldPositionFilter.toJson();
    }
    if (hasBlacklist()) {
        Json::Value & bl = result["blacklist"];
        if (blacklistTime != 0.0)
            bl["time"] = blacklistTime;
        switch (blacklistType) {
        case BL_OFF: bl["type"] = "OFF";  break;
        case BL_USER: bl["type"] = "USER";  break;
        case BL_USER_SITE: bl["type"] = "USER_SITE";  break;
        default:
            throw ML::Exception("unknown blacklist type");
        }

        switch (blacklistScope) {
        case BL_AGENT: bl["scope"] = "AGENT";  break;
        case BL_ACCOUNT: bl["scope"] = "ACCOUNT";  break;
        default:
            throw ML::Exception("unknown blacklist scope");
        }
    }

    if (!visitChannels.empty()) {
        Json::Value & v = result["visits"];
        v["channels"] = visitChannels.toJson();
        v["includeUnmatched"] = includeUnmatchedVisits;
    }

    if (true) {
        Json::Value & bc = result["bidControl"];
        switch (bidControlType) {
        case BC_RELAY: bc["type"] = "RELAY";  break;
        case BC_RELAY_FIXED: bc["type"] = "RELAY_FIXED";  break;
        case BC_FIXED: bc["type"] = "FIXED";  break;
        default:
            throw ML::Exception("unknown bid control type");
        }
        bc["fixedBidCpmInMicros"] = fixedBidCpmInMicros;
    }

    if (!providerConfig.isNull()) {
        result["providerConfig"] = providerConfig;
    }

    result["winFormat"] = RTBKIT::toJson(winFormat);
    result["lossFormat"] = RTBKIT::toJson(lossFormat);
    result["errorFormat"] = RTBKIT::toJson(errorFormat);
    
    if (!ext.isNull()) {
        result["ext"] = ext;
    }

    return result;
}

void
AgentConfig::
addAugmentation(const std::string & name, Json::Value config)
{
    
    AugmentationConfig info;
    info.name = name;
    info.config = std::move(config);

    addAugmentation(info);
}

void
AgentConfig::
addAugmentation(AugmentationConfig info)
{
    for (auto & a: augmentations)
        if (a.name == info.name)
            throw ML::Exception("augmentor " + a.name + " is specified twice");

    augmentations.push_back(std::move(info));

    std::sort(augmentations.begin(), augmentations.end());
}

} // namespace RTBKIT


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
Creative(int width, int height,
         std::string name, int id, int tagId)
    : format(width,height), tagId(tagId), name(name), id(id)
{
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
    
    if (val.isMember("tagId"))
        tagId = val["tagId"].asInt();
    else tagId = -1;

    providerConfig = val["providerConfig"];
    
    if (tagId == -1)
        throw Exception("no tag ID in creative: " + val.toString());

    languageFilter.fromJson(val["languageFilter"], "languageFilter");
    locationFilter.fromJson(val["locationFilter"], "locationFilter");
    exchangeFilter.fromJson(val["exchangeFilter"], "exchangeFilter");
}

Json::Value
Creative::
toJson() const
{
    Json::Value result;
    result["format"] = format.toJson();
    result["name"] = name;
    if (id != -1)
        result["id"] = id;
    if (tagId != -1)
        result["tagId"] = tagId;
    if (!languageFilter.empty())
        result["languageFilter"] = languageFilter.toJson();
    if (!locationFilter.empty())
        result["locationFilter"] = locationFilter.toJson();
    if (!exchangeFilter.empty())
        result["exchangeFilter"] = exchangeFilter.toJson();
    if (!providerConfig.isNull())
        result["providerConfig"] = providerConfig;
    return result;
}

const Creative Creative::sampleWS
    (160, 600, "LeaderBoard", 0, 0);
const Creative Creative::sampleBB
    (300, 250, "BigBox", 1, 1);
const Creative Creative::sampleLB
    (728, 90,  "LeaderBoard", 2, 2);

bool
Creative::
compatible(const AdSpot & adspot) const
{
    return ((format.width == 0 && format.height == 0)
            || adspot.formats.compatible(format));
}

bool
Creative::
biddable(const std::string & exchange,
         const std::string & protocolVersion) const
{
    return true;
}

Json::Value jsonPrint(const Creative & c)
{
    return c.toJson();
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

uint64_t calcMd5(const std::string & str)
{
    CryptoPP::Weak::MD5 md5;
    
    union {
        uint64_t result;
        byte bytes[sizeof(uint64_t)];
    };
    
    md5.CalculateTruncatedDigest(bytes, sizeof(uint64_t),
                                 (const byte *)str.c_str(), str.size());
    
    return result;
}

bool
UserPartition::
matches(const UserIds & ids,
        const std::string& ip,
        const Utf8String& userAgent) const
{
    if (hashOn == NONE)
        return true;
    
    //cerr << "matches: hashOn " << hashOn << " modulus "
    //     << modulus << endl;

    uint32_t value;
    if (hashOn == NONE)
        value = 0;
    else if (hashOn == RANDOM)
        value = random();
    else {
        string str;
        
        switch (hashOn) {
        case EXCHANGEID:   str = ids.exchangeId.toString();   break;
        case PROVIDERID:   str = ids.providerId.toString();   break;
        case IPUA:         str = ip + userAgent.rawString();  break;
        default:
            throw Exception("unknown hashOn");
        };
        
        if (str.empty() || str == "null") return false;

        // TODO: change this to a better hash once we get the chance
        // (clean break in campaigns)
        value = calcMd5(str);

        //cerr << "s = " << s << " value = " << value << endl;
    }

    value %= modulus;

    //cerr << "mod = " << value << endl;

    for (unsigned i = 0;  i < includeRanges.size();  ++i) {
        //cerr << "checking in range " << includeRanges[i].first
        //     << "-" << includeRanges[i].last << endl;
        if (includeRanges[i].in(value)) return true;
        //cerr << "  ... not in" << endl;
    }
    
    //cerr << "not in anything" << endl;

    return false;
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
    default:
        throw ML::Exception("unknown hashOn");
    }
    result["hashOn"] = ho;
    result["modulus"] = modulus;
    for (unsigned i = 0;  i < includeRanges.size();  ++i)
        result["includeRanges"][i] = includeRanges[i].toJson();
    
    return result;
}


/*****************************************************************************/
/* AGENT CONFIG                                                             */
/*****************************************************************************/

AgentConfig::
AgentConfig()
    : test(false), roundRobinWeight(0),
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

bool
AgentConfig::HourOfWeekFilter::
isIncluded(Date date) const
{
    if (isDefault())
        return true;
    if (date == Date())
        throw ML::Exception("null auction date with hour of week filter on");
    int hour = date.hourOfWeek();
    return hourBitmap[hour];
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

        if (it.memberName() == "account")
            newConfig.account = AccountKey::fromJson(*it);
        else if (it.memberName() == "test") {
            newConfig.test = it->asBool();
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
                    if (newConfig.creatives[i].tagId == -1)
                        throw Exception("invalid tag in creative "
                                        + boost::lexical_cast<std::string>((*it)[i]));
                    ;
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
        else if (it.memberName() == "augmentationFilter") {
            newConfig.augmentationFilter.fromJson(*it, "augmentationFilter");
        }
        else if (it.memberName() == "augment") {
            if (!it->isObject())
                throw Exception("augment must be an object of augmentor name to config");

            for (auto jt = (*it).begin(), end = (*it).end(); jt != end; ++jt)
                newConfig.addAugmentation(jt.memberName(), *jt);
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
    if (!augmentationFilter.empty())
        result["augmentationFilter"] = augmentationFilter.toJson();
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
            aug[augmentations[i].name] = augmentations[i].config;
        result["augment"] = aug;
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
    
    return result;
}

BiddableSpots
AgentConfig::
canBid(const ExchangeConnector * exchangeConnector,
       const BidRequest& request,
       const Utf8String & language,
       const Utf8String & location, uint64_t locationHash,
       ML::Lightweight_Hash<uint64_t, int> & locationCache) const
{
    BiddableSpots result;

    // TODO: do a lookup, not an exhaustive scan
    for (unsigned i = 0;  i < request.imp.size();  ++i) {
        auto & item = request.imp[i];

        // Check that the fold position matches
        if (!foldPositionFilter.isIncluded(item.position))
            continue;

        SmallIntVector matching;
        for (unsigned j = 0;  j < creatives.size();  ++j) {
            auto & creative = creatives[j];

            // check that this it's compatible
            const void * exchangeInfo = 0;
            {
                std::lock_guard<ML::Spinlock> guard(creative.lock);
                auto it = creative.providerData.find(exchangeConnector->exchangeName());
                if (it == creative.providerData.end()) {
                    continue;
                }

                exchangeInfo = it->second.get();
            }

            if (exchangeConnector->bidRequestCreativeFilter(request, *this, exchangeInfo) 
                && creative.compatible(item)
                && creative.biddable(request.exchange, request.protocolVersion)
                && creative.exchangeFilter.isIncluded(request.exchange)
                && creative.languageFilter.isIncluded(language.rawString())
                && creative.locationFilter.isIncluded(location, locationHash, locationCache))
                matching.push_back(j);
        }

        if (!matching.empty())
            result.push_back(make_pair(i, matching));
    }
    
    return result;
}

BiddableSpots
AgentConfig::
isBiddableRequest(const ExchangeConnector * exchangeConnector,
                  const BidRequest& request,
                  AgentStats& stats,
                  RequestFilterCache& cache,
                  const FilterStatFn & doFilterStat) const
{
    const void * exchangeInfo = 0;

    /* First, check that the exchange has blessed this campaign as being
       biddable.  If not, we don't go any further. */

    if (exchangeConnector) {
        //cerr << "exchangeName = " << exchangeName
        //     << " providerData.size() = " << providerData.size()
        //     << endl;

        {
            std::lock_guard<ML::Spinlock> guard(lock);
            auto it = providerData.find(exchangeConnector->exchangeName());
            if (it == providerData.end()) {
                if (doFilterStat)
                    doFilterStat("static.001_exchangeRejectedCampaign");
                return BiddableSpots();
            }

            exchangeInfo = it->second.get();
        }

        /* Now we know the exchange connector likes the campaign, we
           should check that the campaign and bid request are compatible.
           
           First we pre-filter
        */
        if (!exchangeConnector->bidRequestPreFilter(request, *this, exchangeInfo)) {
            if (doFilterStat)
                doFilterStat("static.001_exchangeMismatchBidRequestCampaignPre");
            return BiddableSpots();
        }
    }

    /* Find matching creatives for the agent.  This includes fold position
       filtering.
    */
    BiddableSpots biddableSpots = canBid(
            exchangeConnector,
            request,
            cache.language,
            cache.location,
            cache.locationHash,
            cache.locationFilter);

    //cerr << "agent " << it->first << " imp "
    //     << biddableSpots.size() << endl;

    if (biddableSpots.empty()) {
        //cerr << "no biddable imp" << endl;
        ML::atomic_inc(stats.noSpots);
        if (doFilterStat) doFilterStat("static.010_noSpots");
        return BiddableSpots();
    }

    /* Check for generic ID filtering. */
    if (!requiredIds.empty()) {
        for (unsigned i = 0;  i < requiredIds.size();  ++i) {
            if (!request.userIds.count(requiredIds[i])) {
                ML::atomic_inc(stats.requiredIdMissing);
                if (doFilterStat)
                    doFilterStat(("static.030_missingRequiredId_"
                                  + requiredIds[i]).c_str());
                return BiddableSpots();
            }
        }
    }

    /* Check for hour of week. */
    if (!hourOfWeekFilter.isIncluded(request.timestamp)) {
        ML::atomic_inc(stats.hourOfWeekFiltered);
        if (doFilterStat) doFilterStat("static.040_hourOfWeek");
        return BiddableSpots();
    }

    /* Check for the exchange. */
    if (!exchangeFilter.isIncluded(request.exchange)) {
        ML::atomic_inc(stats.exchangeFiltered);
        if (doFilterStat) doFilterStat("static.050_exchangeFiltered");
        return BiddableSpots();
    }
    
    ML::atomic_inc(stats.passedStaticPhase1);

    /* Check for the location. */
    if (!locationFilter.isIncluded(
                    cache.location, cache.locationHash, cache.locationFilter))
    {
        ML::atomic_inc(stats.locationFiltered);
        if (doFilterStat) doFilterStat("static.060_locationFiltered");
        return BiddableSpots();
    }

    /* Check for language. */
    if (!languageFilter.isIncluded(
            cache.language.rawString(), cache.languageHash, cache.languageFilter))
    {
        ML::atomic_inc(stats.languageFiltered);
        if (doFilterStat) doFilterStat("static.070_languageFiltered");
        return BiddableSpots();
    }

    ML::atomic_inc(stats.passedStaticPhase2);

    /* Check for segment inclusion/exclusion. */
    bool exclude = false;
    int segNum = 0;
    for (auto it = segments.begin(), end = segments.end();
         !exclude && it != end;  ++it, ++segNum)
    {
        // Check if the exchange applies to this segment filter
        if (!it->second.applyToExchanges.isIncluded(request.exchange))
            continue;

        // Look up this segment source in the bid request
        auto segs = request.segments.find(it->first);

        // If not found, then check what the default response is
        if (segs == request.segments.end()) {
            exclude = it->second.excludeIfNotPresent;
            ML::atomic_inc(stats.segmentsMissing);
            if (exclude) {
                if (doFilterStat) doFilterStat(
                        ("static.080_segmentInfoMissing_" + it->first).c_str());
            }
        }
        else {
            const auto & segments = segs->second;

            // Check what the include/exclude list says
            IncludeExcludeResult inc = it->second.process(*segments);

            switch (inc) {
            case IE_NO_DATA:
                exclude = it->second.excludeIfNotPresent;
                if (doFilterStat) doFilterStat(
                             ("static.080_segmentHasNoData_" + it->first).c_str());
                break;
            case IE_NOT_INCLUDED:
            case IE_EXCLUDED:
                if (doFilterStat) doFilterStat(
                             ("static.080_segmentExcluded_" + it->first).c_str());
                exclude = true;
                break;
            case IE_PASSED:
                break;
            }
        }

        if (segNum == 0 && exclude)
            ML::atomic_inc(stats.filter1Excluded);
        else if (segNum == 1 && exclude)
            ML::atomic_inc(stats.filter2Excluded);
        else if (exclude)
            ML::atomic_inc(stats.filternExcluded);
    }

    if (exclude) {
        ML::atomic_inc(stats.segmentFiltered);
        return BiddableSpots();
    }

    ML::atomic_inc(stats.passedStaticPhase3);

    /* Check that the user partition matches. */
    if (!userPartition.matches(request.userIds, request.ipAddress, request.userAgent)) {
        ML::atomic_inc(stats.userPartitionFiltered);
        if (doFilterStat) doFilterStat("static.080_userPartitionFiltered");
        return BiddableSpots();
    }

    /* Check for blacklisted domains. */
    if (!hostFilter.isIncluded(request.url, cache.urlHash, cache.urlFilter)) {
        ML::atomic_inc(stats.urlFiltered);
        if (doFilterStat) doFilterStat("static.085_hostFiltered");
        return BiddableSpots();
    }

    /* Check for blacklisted URLs. */
    if (!urlFilter.isIncluded(
                    request.url.toString(), cache.urlHash, cache.urlFilter))
    {
        ML::atomic_inc(stats.urlFiltered);
        if (doFilterStat) doFilterStat("static.090_urlFiltered");
        return BiddableSpots();
    }

    /* Finally, perform any expensive filtering in the exchange connector. */
    if (exchangeConnector) {

        if (!exchangeConnector->bidRequestPostFilter(request, *this, exchangeInfo)) {
            if (doFilterStat)
                doFilterStat("static.099_exchangeMismatchBidRequestCampaignPost");
            return BiddableSpots();
        }
    }

    return biddableSpots;
}

void
AgentConfig::
addAugmentation(const std::string & name, Json::Value config)
{
    
    AugmentationInfo info;
    info.name = name;
    info.config = std::move(config);

    addAugmentation(info);
}

void
AgentConfig::
addAugmentation(AugmentationInfo info)
{
    for (auto & a: augmentations)
        if (a.name == info.name)
            throw ML::Exception("augmentor " + a.name + " is specified twice");

    augmentations.push_back(std::move(info));

    std::sort(augmentations.begin(), augmentations.end());
}

} // namespace RTBKIT


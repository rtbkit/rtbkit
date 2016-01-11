/** basic_filters.cc                                 -*- C++ -*-
    Rémi Attab, 24 Jul 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.
    Default pool of filters for an BidRequest object.
*/

#include "static_filters.h"

// User partition filter.
#define CRYPTOPP_ENABLE_NAMESPACE_WEAK 1
#include "crypto++/md5.h"


using namespace std;
using namespace ML;

namespace RTBKIT {

/******************************************************************************/
/* SEGMENT FILTER                                                             */
/******************************************************************************/

void
SegmentsFilter::
setConfig(unsigned cfgIndex, const AgentConfig& config, bool value)
{
    for (const auto& entry : config.segments) {
        auto& segment = data[entry.first];

        segment.ie.setInclude(cfgIndex, value, entry.second.include);
        segment.ie.setExclude(cfgIndex, value, entry.second.exclude);

        segment.exchange.setIncludeExclude(
                cfgIndex, value, entry.second.applyToExchanges);

        if (entry.second.excludeIfNotPresent) {
            if (value && segment.excludeIfNotPresent.empty())
                excludeIfNotPresent.insert(entry.first);

            segment.excludeIfNotPresent.set(cfgIndex, value);

            if (!value && segment.excludeIfNotPresent.empty())
                excludeIfNotPresent.erase(entry.first);
        }
    }
}

ConfigSet
SegmentsFilter::SegmentData::
applyExchangeFilter(FilterState& state, const ConfigSet& result) const
{
    ConfigSet current = state.configs();

    /* This is a bit tricky because our filter mechanism doesn't gracefully
       support skipping filters which is required for the exchange IE. So
       instead we'll take the result of the original filter and massage it until
       we get the mask we want.
       First off, let's figure out which configs would change from 1 to 0 if we
       applied result to the state.
    */
    ConfigSet affected = current ^ (current & result);
    if (affected.empty()) return ConfigSet(true);

    /* Out of those configs, let's remove the ones that we should skip. Note
       that the filter will return all the configs that should not be
       skipped.
    */
    ConfigSet leftover = affected & exchange.filter(state.request.exchange);

    /* At this point we have a 1 in our leftover bitfield for each configs
       that would be filtered out and is not marked for skipping. We can
       therefor remove those configs from state by negating the bitfield.
       Magic!
    */
    return leftover.negate();
}


void
SegmentsFilter::
fillFilterReasons(FilterState& state, ConfigSet& beforeFilt,
                  ConfigSet& afterFilt, const std::string & segment) const {

    // Some Magic to get all the filtered out configs by this segment.
    FilterState::FilterReasons& reasons = state.getFilterReasons();
    reasons[segment] = beforeFilt ^ (beforeFilt & afterFilt);

}

void
SegmentsFilter::
filter(FilterState& state) const
{
    unordered_set<string> toCheck = excludeIfNotPresent;

    for (const auto& segment : state.request.segments) {
        toCheck.erase(segment.first);

        auto it = data.find(segment.first);
        if (it == data.end()) continue;

        ConfigSet beforeFilt = state.configs();
        ConfigSet result = it->second.ie.filter(*segment.second);

        ConfigSet result2 = it->second.applyExchangeFilter(state, result);
        state.narrowConfigs(result2);

        fillFilterReasons(state, beforeFilt, result2, segment.first);

        if (state.configs().empty()) return;
    }

    for (const auto& segment : toCheck) {
        auto it = data.find(segment);
        if (it == data.end()) continue;

        ConfigSet result = it->second.excludeIfNotPresent.negate();
        ConfigSet result2 = it->second.applyExchangeFilter(state, result);
        ConfigSet beforeFilt = state.configs();
        state.narrowConfigs(result2);
        fillFilterReasons(state, beforeFilt, result2, segment);
        if (state.configs().empty()) return;
    }
}


/******************************************************************************/
/* USER PARTITION FILTER                                                      */
/******************************************************************************/

void
UserPartitionFilter::
setConfig(unsigned cfgIndex, const AgentConfig& config, bool value)
{
    const auto& part = config.userPartition;

    if (part.hashOn == UserPartition::NONE) {
        defaultSet.set(cfgIndex, value);
        return;
    }

    auto uid = config.externalId;
    auto& entry = data[getKey(part, uid)];
    entry.uid = uid;
    if (entry.hashOn == UserPartition::NONE) {
        entry.modulus = part.modulus;
        entry.hashOn = part.hashOn;
    }
    ExcAssertEqual(entry.modulus, part.modulus);
    ExcAssertEqual(entry.hashOn, part.hashOn);

    entry.excludeIfEmpty.set(cfgIndex);

    if (value) entry.filter.addConfig(cfgIndex, part.includeRanges);
    else entry.filter.removeConfig(cfgIndex, part.includeRanges);
}

void
UserPartitionFilter::
filter(FilterState& state) const
{
    ConfigSet matches = defaultSet;
    ConfigSet excludes;

    for (const auto& entry : data) {
        auto value = getValue(state.request, entry.second);

        if (!value.first) excludes |= entry.second.excludeIfEmpty;
        else matches |= entry.second.filter.filter(value.second);
    }

    matches &= excludes.negate();
    state.narrowConfigs(matches);
}

namespace {

// \todo Currently uses MD5 which is suboptimal.
uint64_t calcHash(const std::string& str)
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

} // namespace anonymous

std::pair<bool, uint64_t>
UserPartitionFilter::
getValue(const BidRequest& br, const FilterEntry& entry) const
{
    if (entry.hashOn == UserPartition::RANDOM)
        return make_pair(true, random() % entry.modulus);

    string str;

    switch (entry.hashOn) {
    case UserPartition::EXCHANGEID:
        str = br.userIds.exchangeId.toString(); break;

    case UserPartition::PROVIDERID:
        str = br.userIds.providerId.toString(); break;

    case UserPartition::IPUA:
        str = br.ipAddress + br.userAgent.utf8String(); break;

    default: ExcAssert(false);
    };

    if (str.empty() || str == "null") return make_pair(false, 0);

    auto h = calcHash(str) + entry.uid;
    return make_pair(true, h % entry.modulus);
}


LatLongDevFilter::Square
LatLongDevFilter::squareFromLatLongRadius(float lat, float lon, float radius)
{
    Square sq;
    sq.y_max = (lat * LATITUDE_1DEGREE_KMS) + radius;
    sq.y_min = (lat * LATITUDE_1DEGREE_KMS) - radius;

    sq.x_max = ( lon * LONGITUDE_1DEGREE_KMS * cosInDegrees(lat) ) + radius;
    sq.x_min = ( lon * LONGITUDE_1DEGREE_KMS * cosInDegrees(lat) ) - radius;

    return sq;
}

void LatLongDevFilter::addConfig(unsigned cfgIndex,
        const std::shared_ptr<RTBKIT::AgentConfig>& config)
{
    SquareList squares;
    for ( const RTBKIT::LatLonRad & llr : config->latLongDevFilter.latlonrads){
        auto sq = squareFromLatLongRadius(llr.lat, llr.lon, llr.radius);
        squares.push_back(sq);
    }
    if ( ! squares.empty()){
        squares_by_confindx[cfgIndex] = squares;
        configs_with_filt.set(cfgIndex);
    }
}

void LatLongDevFilter::removeConfig(unsigned cfgIndex,
        const std::shared_ptr<RTBKIT::AgentConfig>& config)
{
    if ( squares_by_confindx.count(cfgIndex) > 0){
        squares_by_confindx.erase(cfgIndex);
        configs_with_filt.reset(cfgIndex);
    }
}

void LatLongDevFilter::filter(RTBKIT::FilterState& state) const
 {
    RTBKIT::ConfigSet matches = state.configs();

    auto it = squares_by_confindx.begin();
    if ( ! checkLatLongPresent(state.request)){
        // If there is no geo info the filter, then filter out all the
        // agent configs that has this filter present.
        state.narrowConfigs(configs_with_filt.negate());
    } else {
        // Filter using the lat long of the request and from the configs
        // of the agents.
        for ( ; it != squares_by_confindx.end(); ++it ){
            if (pointInsideAnySquare(state.request.device->geo->lat.val,
                    state.request.device->geo->lon.val, it->second)) {
                continue;
            }
            matches.reset(it->first);
        }
        state.narrowConfigs(matches);
    }

 }

bool LatLongDevFilter::checkLatLongPresent(
        const RTBKIT::BidRequest & req) const
{
    if ( ! req.device) return false;
    if ( ! req.device->geo) return false;
    if ( req.device->geo->lat.val == std::numeric_limits<float>::quiet_NaN() ||
         req.device->geo->lon.val == std::numeric_limits<float>::quiet_NaN() )
        return false;
    return true;
}

bool
LatLongDevFilter::pointInsideAnySquare(float lat, float lon,
        const SquareList & squares)
{
    const float y = lat * LATITUDE_1DEGREE_KMS;
    const float x = lon * LONGITUDE_1DEGREE_KMS * cosInDegrees(lat);
    for ( auto & sq : squares){
        if (insideSquare(x, y , sq)) return true;
    }
    return false;
}

} // namespace RTBKIT

/******************************************************************************/
/* INIT FILTERS                                                               */
/******************************************************************************/

namespace {

struct AtInit {
    AtInit()
    {
        RTBKIT::FilterBase::registerFactory<RTBKIT::SegmentsFilter>();
        RTBKIT::FilterBase::registerFactory<RTBKIT::UserPartitionFilter>();
        RTBKIT::FilterBase::registerFactory<RTBKIT::HourOfWeekFilter>();
        RTBKIT::FilterBase::registerFactory<RTBKIT::UrlFilter>();
        RTBKIT::FilterBase::registerFactory<RTBKIT::HostFilter>();
        RTBKIT::FilterBase::registerFactory<RTBKIT::LanguageFilter>();
        RTBKIT::FilterBase::registerFactory<RTBKIT::LocationFilter>();
        RTBKIT::FilterBase::registerFactory<RTBKIT::ExchangePreFilter>();
        RTBKIT::FilterBase::registerFactory<RTBKIT::ExchangePostFilter>();
        RTBKIT::FilterBase::registerFactory<RTBKIT::ExchangeNameFilter>();
        RTBKIT::FilterBase::registerFactory<RTBKIT::FoldPositionFilter>();
        RTBKIT::FilterBase::registerFactory<RTBKIT::RequiredIdsFilter>();
        RTBKIT::FilterBase::registerFactory<RTBKIT::LatLongDevFilter>();
    }

} AtInit;

} // namespace anonymous

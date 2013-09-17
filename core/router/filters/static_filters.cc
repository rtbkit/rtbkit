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
filter(FilterState& state) const
{
    unordered_set<string> toCheck = excludeIfNotPresent;

    for (const auto& segment : state.request.segments) {
        toCheck.erase(segment.first);

        auto it = data.find(segment.first);
        if (it == data.end()) continue;

        ConfigSet result = it->second.ie.filter(*segment.second);
        state.narrowConfigs(it->second.applyExchangeFilter(state, result));
        if (state.configs().empty()) return;
    }

    for (const auto& segment : toCheck) {
        auto it = data.find(segment);
        if (it == data.end()) continue;

        ConfigSet result = it->second.excludeIfNotPresent.negate();
        state.narrowConfigs(it->second.applyExchangeFilter(state, result));
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

    auto& entry = data[getKey(part)];
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
        str = br.ipAddress + br.userAgent.rawString(); break;

    default: ExcAssert(false);
    };

    if (str.empty() || str == "null") return make_pair(false, 0);

    return make_pair(true, calcHash(str) % entry.modulus);
}


} // namespace RTBKIT


/******************************************************************************/
/* INIT FILTERS                                                               */
/******************************************************************************/

namespace {

struct InitFilters
{
    InitFilters()
    {
        RTBKIT::FilterRegistry::registerFilter<RTBKIT::SegmentsFilter>();
        RTBKIT::FilterRegistry::registerFilter<RTBKIT::FoldPositionFilter>();
        RTBKIT::FilterRegistry::registerFilter<RTBKIT::HourOfWeekFilter>();
        RTBKIT::FilterRegistry::registerFilter<RTBKIT::RequiredIdsFilter>();
        RTBKIT::FilterRegistry::registerFilter<RTBKIT::UserPartitionFilter>();

        RTBKIT::FilterRegistry::registerFilter<RTBKIT::UrlFilter>();
        RTBKIT::FilterRegistry::registerFilter<RTBKIT::HostFilter>();
        RTBKIT::FilterRegistry::registerFilter<RTBKIT::LanguageFilter>();
        RTBKIT::FilterRegistry::registerFilter<RTBKIT::LocationFilter>();

        RTBKIT::FilterRegistry::registerFilter<RTBKIT::ExchangePreFilter>();
        RTBKIT::FilterRegistry::registerFilter<RTBKIT::ExchangeNameFilter>();
        RTBKIT::FilterRegistry::registerFilter<RTBKIT::ExchangePostFilter>();
    }

} initFilters;


} // namespace anonymous

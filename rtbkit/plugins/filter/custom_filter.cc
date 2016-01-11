/** custom_filter.cc                                 -*- C++ -*-
    Sirma Cagil Altay, 16 Oct 2015
    Copyright (c) 2015 Datacratic.  All rights reserved.

    Same filters as Segments and Url ones. 
    Only names changed
*/

#include "custom_filter.h"

using namespace std;
using namespace ML;

namespace RTBKIT {

/******************************************************************************/
/* MY FILTER                                                                  */
/******************************************************************************/

void
MyFilter::
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
MyFilter::SegmentData::
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
MyFilter::
fillFilterReasons(FilterState& state, ConfigSet& beforeFilt,
                  ConfigSet& afterFilt, const std::string & segment) const {

    // Some Magic to get all the filtered out configs by this segment.
    FilterState::FilterReasons& reasons = state.getFilterReasons();
    reasons[segment] = beforeFilt ^ (beforeFilt & afterFilt);

}

void
MyFilter::
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

} // namespace RTBKIT

/******************************************************************************/
/* INIT FILTERS                                                               */
/******************************************************************************/

namespace {

struct AtInit {
    AtInit()
    {
        RTBKIT::FilterBase::registerFactory<RTBKIT::MyFilter>();
        RTBKIT::FilterBase::registerFactory<RTBKIT::YourFilter>();
    }

} AtInit;

} // namespace anonymous


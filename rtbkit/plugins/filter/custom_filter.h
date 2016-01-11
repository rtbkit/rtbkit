/** custom_filter.h                                 -*- C++ -*-
    Sirma Cagil Altay, 16 Oct 2015
    Copyright (c) 2015 Datacratic.  All rights reserved.

    Same filters as Segments and Url ones. 
    Only names changed
*/

#pragma once

#include "rtbkit/core/router/filters/generic_filters.h"
#include "rtbkit/core/router/filters/priority.h"
#include "rtbkit/common/exchange_connector.h"
#include "jml/utils/compact_vector.h"

#include <array>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <cmath>

namespace RTBKIT {


/******************************************************************************/
/* MY FILTER                                                                  */
/******************************************************************************/

struct MyFilter : public FilterBaseT<MyFilter>
{
    static constexpr const char* name = "My";
    unsigned priority() const { return Priority::Segments; }

    void setConfig(unsigned configIndex, const AgentConfig& config, bool value);
    void filter(FilterState& state) const;

private:

    void fillFilterReasons(FilterState& state, ConfigSet& beforeFilt,
            ConfigSet& afterFilt, const std::string & segment) const;

    struct SegmentData
    {
        typedef ListFilter<std::string> ExchangeFilterT;
        IncludeExcludeFilter<ExchangeFilterT> exchange;

        IncludeExcludeFilter<SegmentListFilter> ie;
        ConfigSet excludeIfNotPresent;

        ConfigSet applyExchangeFilter(
                FilterState& state, const ConfigSet& result) const;
    };

    std::unordered_map<std::string, SegmentData> data;
    std::unordered_set<std::string> excludeIfNotPresent;
};

/******************************************************************************/
/* YOUR FILTER                                                                */
/******************************************************************************/

struct YourFilter : public FilterBaseT<YourFilter>
{
    static constexpr const char* name = "Your";
    unsigned priority() const { return Priority::Url; }

    void setConfig(unsigned configIndex, const AgentConfig& config, bool value)
    {
        impl.setIncludeExclude(configIndex, value, config.urlFilter);
    }

    void filter(FilterState& state) const
    {
        state.narrowConfigs(impl.filter(state.request.url.toString()));
    }

private:
    typedef RegexFilter<boost::regex, std::string> BaseFilter;
    IncludeExcludeFilter<BaseFilter> impl;
};


} // namespace RTBKIT

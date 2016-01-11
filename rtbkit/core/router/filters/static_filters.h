/** basic_filters.h                                 -*- C++ -*-
    Rémi Attab, 26 Jul 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Default pool of filters for a bid request object.

*/

#pragma once

#include "generic_filters.h"
#include "priority.h"
#include "rtbkit/common/exchange_connector.h"
#include "jml/utils/compact_vector.h"

#include <array>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <cmath>

namespace RTBKIT {


/******************************************************************************/
/* SEGMENTS FILTER                                                            */
/******************************************************************************/

struct SegmentsFilter : public FilterBaseT<SegmentsFilter>
{
    static constexpr const char* name = "Segments";
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
/* USER PARTITION FILTER                                                      */
/******************************************************************************/

struct UserPartitionFilter : public FilterBaseT<UserPartitionFilter>
{
    static constexpr const char* name = "UserPartition";
    unsigned priority() const { return Priority::UserPartition; }

    void setConfig(unsigned cfgIndex, const AgentConfig& config, bool value);
    void filter(FilterState& state) const;

private:

    struct FilterEntry
    {
        FilterEntry() : hashOn(UserPartition::NONE) {}

        IntervalFilter<int> filter;
        ConfigSet excludeIfEmpty;
        int modulus;
        int uid;
        UserPartition::HashOn hashOn;
    };

    ConfigSet defaultSet;
    std::unordered_map<uint64_t, FilterEntry> data;

    uint64_t getKey(const UserPartition& obj, int uid) const
    {
        return uint64_t(uid) << 32 | uint64_t(obj.modulus) << 16 | uint64_t(obj.hashOn);
    }

    std::pair<bool, uint64_t>
    getValue(const BidRequest& br, const FilterEntry& entry) const;

};


/******************************************************************************/
/* HOUR OF WEEK FILTER                                                        */
/******************************************************************************/

struct HourOfWeekFilter : public FilterBaseT<HourOfWeekFilter>
{
    HourOfWeekFilter() { data.fill(ConfigSet()); }

    static constexpr const char* name = "HourOfWeek";
    unsigned priority() const { return Priority::HourOfWeek; }

    void setConfig(unsigned configIndex, const AgentConfig& config, bool value)
    {
        const auto& bitmap = config.hourOfWeekFilter.hourBitmap;
        for (size_t i = 0; i < bitmap.size(); ++i) {
            if (!bitmap[i]) continue;
            data[i].set(configIndex, value);
        }
    }

    void filter(FilterState& state) const
    {
        ExcCheckNotEqual(state.request.timestamp, Date(), "Null auction date");
        state.narrowConfigs(data[state.request.timestamp.hourOfWeek()]);
    }

private:

    std::array<ConfigSet, 24 * 7> data;
};


/******************************************************************************/
/* URL FILTER                                                                 */
/******************************************************************************/

struct UrlFilter : public FilterBaseT<UrlFilter>
{
    static constexpr const char* name = "Url";
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


/******************************************************************************/
/* HOST FILTER                                                                */
/******************************************************************************/

struct HostFilter : public FilterBaseT<HostFilter>
{
    static constexpr const char* name = "Host";
    unsigned priority() const { return Priority::Host; }

    void setConfig(unsigned configIndex, const AgentConfig& config, bool value)
    {
        impl.setIncludeExclude(configIndex, value, config.hostFilter);
    }

    void filter(FilterState& state) const
    {
        state.narrowConfigs(impl.filter(state.request.url));
    }

private:
    IncludeExcludeFilter< DomainFilter<std::string> > impl;
};


/******************************************************************************/
/* LANGUAGE FILTER                                                            */
/******************************************************************************/

struct LanguageFilter : public FilterBaseT<LanguageFilter>
{
    static constexpr const char* name = "Language";
    unsigned priority() const { return Priority::Language; }

    void setConfig(unsigned configIndex, const AgentConfig& config, bool value)
    {
        impl.setIncludeExclude(configIndex, value, config.languageFilter);
    }

    void filter(FilterState& state) const
    {
        state.narrowConfigs(impl.filter(state.request.language.utf8String()));
    }

private:
    typedef RegexFilter<boost::regex, std::string> BaseFilter;
    IncludeExcludeFilter<BaseFilter> impl;
};


/******************************************************************************/
/* LOCATION FILTER                                                            */
/******************************************************************************/

struct LocationFilter : public FilterBaseT<LocationFilter>
{
    static constexpr const char* name = "Location";
    unsigned priority() const { return Priority::Location; }

    void setConfig(unsigned configIndex, const AgentConfig& config, bool value)
    {
        impl.setIncludeExclude(configIndex, value, config.locationFilter);
    }

    void filter(FilterState& state) const
    {
        Datacratic::UnicodeString location = state.request.location.fullLocationString();
        state.narrowConfigs(impl.filter(location));
    }

private:
    typedef RegexFilter<boost::u32regex, Datacratic::UnicodeString> BaseFilter;
    IncludeExcludeFilter<BaseFilter> impl;
};


/******************************************************************************/
/* EXCHANGE PRE/POST FILTER                                                   */
/******************************************************************************/

/** The lock makes it next to impossible to do any kind of pre-processing. */
struct ExchangePreFilter : public IterativeFilter<ExchangePreFilter>
{
    static constexpr const char* name = "ExchangePre";
    unsigned priority() const { return Priority::ExchangePre; }

    bool filterConfig(FilterState& state, const AgentConfig& config) const
    {
        if (!state.exchange) return false;

        auto it = config.providerData.find(state.exchange->exchangeName());
        if (it == config.providerData.end()) return false;

        return state.exchange->bidRequestPreFilter(
                state.request, config, it->second.get());
    }
};

struct ExchangePostFilter : public IterativeFilter<ExchangePostFilter>
{
    static constexpr const char* name = "ExchangePost";
    unsigned priority() const { return Priority::ExchangePost; }

    bool filterConfig(FilterState& state, const AgentConfig& config) const
    {
        if (!state.exchange) return false;

        auto it = config.providerData.find(state.exchange->exchangeName());
        if (it == config.providerData.end()) return false;

        return state.exchange->bidRequestPostFilter(
                state.request, config, it->second.get());
    }
};


/******************************************************************************/
/* EXCHANGE NAME FILTER                                                       */
/******************************************************************************/

struct ExchangeNameFilter : public FilterBaseT<ExchangeNameFilter>
{
    static constexpr const char* name = "ExchangeName";
    unsigned priority() const { return Priority::ExchangeName; }


    void setConfig(unsigned configIndex, const AgentConfig& config, bool value)
    {
        data.setIncludeExclude(configIndex, value, config.exchangeFilter);
    }

    void filter(FilterState& state) const
    {
        state.narrowConfigs(data.filter(state.request.exchange));
    }

private:
    IncludeExcludeFilter< ListFilter<std::string> > data;
};


/******************************************************************************/
/* FOLD POSITION FILTER                                                       */
/******************************************************************************/

struct FoldPositionFilter : public FilterBaseT<FoldPositionFilter>
{
    static constexpr const char* name = "FoldPosition";
    unsigned priority() const { return Priority::FoldPosition; }

    void setConfig(unsigned cfgIndex, const AgentConfig& config, bool value)
    {
        impl.setIncludeExclude(cfgIndex, value, config.foldPositionFilter);
    }

    void filter(FilterState& state) const
    {
        for (const auto& imp : state.request.imp) {
            state.narrowConfigs(impl.filter(imp.position));
            if (state.configs().empty()) break;
        }
    }

private:
    IncludeExcludeFilter< ListFilter<OpenRTB::AdPosition> > impl;
};


/******************************************************************************/
/* REQUIRED IDS FILTER                                                        */
/******************************************************************************/

struct RequiredIdsFilter : public FilterBaseT<RequiredIdsFilter>
{
    static constexpr const char* name = "RequireIds";
    unsigned priority() const { return Priority::RequiredIds; }

    void setConfig(unsigned cfgIndex, const AgentConfig& config, bool value)
    {
        for (const auto& domain : config.requiredIds) {
            domains[domain].set(cfgIndex, value);
            required.insert(domain);
        }
    }

    void filter(FilterState& state) const
    {
        std::unordered_set<std::string> missing = required;

        for (const auto& uid : state.request.userIds)
            missing.erase(uid.first);

        ConfigSet mask;
        for (const auto& domain : missing) {
            auto it = domains.find(domain);
            ExcAssert(it != domains.end());
            mask |= it->second;
        }

        state.narrowConfigs(mask.negate());
    }

private:

    std::unordered_map<std::string, ConfigSet> domains;
    std::unordered_set<std::string> required;
};


struct LatLongDevFilter : public RTBKIT::FilterBaseT<LatLongDevFilter>
{
    static constexpr const char* name = "latLongDevFilter";

    /**
     * To see if a point is inside a required area we aproximate that area (
     * circular) with a square, and check that the point is within the
     * boundaries.
     */
    struct Square{
        float x_max;
        float x_min;
        float y_max;
        float y_min;
    };

    typedef std::vector<Square> SquareList;
    std::unordered_map<unsigned, SquareList> squares_by_confindx;
    ConfigSet configs_with_filt;

    unsigned priority() const { return Priority::LatLong; } //low priority

    static constexpr float LONGITUDE_1DEGREE_KMS = 111.321;
    static constexpr float LATITUDE_1DEGREE_KMS = 111.0;

    /**
     * Convert the lat long in a 2D square of side radius.
     * Make it with the following approximations:
     * 1 degree of latitude = 111 kms
     * 1 degre of longitude = 111.321 * cos(latitude) kms
     */
    static Square squareFromLatLongRadius(float lat, float lon, float radius);

    /**
     * Save the squares for each config index.
     * Important: We only filter by those who has the filter.
     */
    virtual void addConfig(unsigned cfgIndex,
            const std::shared_ptr<RTBKIT::AgentConfig>& config);

    /**
     * Remove the square list for the given config index.
     */
    virtual void removeConfig(unsigned cfgIndex,
            const std::shared_ptr<RTBKIT::AgentConfig>& config);

    /**
     * Particular implementation of virtual method for this filter type.
     */
    virtual void filter(RTBKIT::FilterState& state) const ;

    /**
     * Check if it is present in the bid request:
     * - the device
     * - the the geo in the device
     * - the lat and lon in the geo (from the device)
     * If they are present, return true. Otherwise return false.
     */
    bool checkLatLongPresent(const RTBKIT::BidRequest & req) const ;

    /**
     * Return true is the point is inside in at least one of the given
     * square of the list given. Otherwise false.
     */
    static bool pointInsideAnySquare(float lat, float lon,
            const SquareList & squares);

    /**
     * Check if the given point defined by (x, y) is inside of the square
     * defined by the two edges (x_max,y_max) and (x_min, y_min).
     */
    inline static bool insideSquare(float x, float y, const Square & sq )
    {
        return x < sq.x_max && x > sq.x_min && y < sq.y_max && y > sq.y_min;
    }

    inline static float cosInDegrees(float degrees)
    {
        static const double deeToGrad = 3.14159265 / 180.0;
        return cos(degrees * deeToGrad);
    }

};


} // namespace RTBKIT

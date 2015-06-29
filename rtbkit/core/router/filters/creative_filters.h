/** creative_filter.h                                 -*- C++ -*-
    Rémi Attab, 09 Aug 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Filters related to creatives.

*/

#pragma once

#include "generic_creative_filters.h"
#include "priority.h"
#include "rtbkit/common/exchange_connector.h"

#include <unordered_map>
#include <unordered_set>
#include <mutex>

namespace RTBKIT {


/******************************************************************************/
/* CREATIVE FORTMAT FILTER                                                    */
/******************************************************************************/

struct CreativeFormatFilter : public CreativeFilter<CreativeFormatFilter>
{
    static constexpr const char* name = "CreativeFormat";
    unsigned priority() const { return Priority::CreativeFormat; }

    void addCreative(
            unsigned cfgIndex, unsigned crIndex, const Creative& creative)
    {
        auto formatKey = makeKey(creative.format);
        formatFilter[formatKey].set(crIndex, cfgIndex);
    }

    void removeCreative(
            unsigned cfgIndex, unsigned crIndex, const Creative& creative)
    {
        auto formatKey = makeKey(creative.format);
        formatFilter[formatKey].reset(crIndex, cfgIndex);
    }

    void filterImpression(
            FilterState& state, unsigned impIndex, const AdSpot& imp) const
    {
        if(!(imp.formats.empty()))
        {
            // The 0x0 format means: match anything.
            CreativeMatrix creatives = get(Format(0,0));

            for (const auto& format : imp.formats)
                creatives |= get(format);

            state.narrowCreativesForImp(impIndex, creatives);
        }
    }


private:

    typedef uint32_t FormatKey;
    static_assert(sizeof(FormatKey) == sizeof(Format),
            "Conversion of FormatKey depends on size of Format");

    FormatKey makeKey(const Format& format) const
    {
        return uint32_t(format.width << 16 | format.height);
    }

    CreativeMatrix get(const Format& format) const
    {
        auto it = formatFilter.find(makeKey(format));
        return it == formatFilter.end() ? CreativeMatrix() : it->second;
    }

    std::unordered_map<uint32_t, CreativeMatrix> formatFilter;
};


/******************************************************************************/
/* CREATIVE LANGUAGE FILTER                                                   */
/******************************************************************************/

struct CreativeLanguageFilter : public CreativeFilter<CreativeLanguageFilter>
{
    static constexpr const char* name = "CreativeLanguage";
    unsigned priority() const { return Priority::CreativeLanguage; }

    void addCreative(
            unsigned cfgIndex, unsigned crIndex, const Creative& creative)
    {
        impl.addIncludeExclude(cfgIndex, crIndex, creative.languageFilter);
    }

    void removeCreative(
            unsigned cfgIndex, unsigned crIndex, const Creative& creative)
    {
        impl.removeIncludeExclude(cfgIndex, crIndex, creative.languageFilter);
    }

    void filter(FilterState& state) const
    {
        state.narrowAllCreatives(
                impl.filter(state.request.language.utf8String()));
    }

private:
    CreativeIncludeExcludeFilter< CreativeListFilter<std::string> > impl;
};


/******************************************************************************/
/* CREATIVE LOCATION FILTER                                                   */
/******************************************************************************/

struct CreativeLocationFilter : public CreativeFilter<CreativeLocationFilter>
{
    static constexpr const char* name = "CreativeLocation";
    unsigned priority() const { return Priority::CreativeLocation; }

    void addCreative(
            unsigned cfgIndex, unsigned crIndex, const Creative& creative)
    {
        impl.addIncludeExclude(cfgIndex, crIndex, creative.locationFilter);
    }

    void removeCreative(
            unsigned cfgIndex, unsigned crIndex, const Creative& creative)
    {
        impl.removeIncludeExclude(cfgIndex, crIndex, creative.locationFilter);
    }

    void filter(FilterState& state) const
    {
        state.narrowAllCreatives(
                impl.filter(state.request.location.fullLocationString()));
    }

private:
    typedef CreativeRegexFilter<boost::u32regex, Datacratic::UnicodeString> BaseFilter;
    CreativeIncludeExcludeFilter<BaseFilter> impl;
};


/******************************************************************************/
/* CREATIVE EXCHANGE NAME FILTER                                              */
/******************************************************************************/

struct CreativeExchangeNameFilter :
        public CreativeFilter<CreativeExchangeNameFilter>
{
    static constexpr const char* name = "CreativeExchangeName";
    unsigned priority() const { return Priority::CreativeExchangeName; }


    void addCreative(
            unsigned cfgIndex, unsigned crIndex, const Creative& creative)
    {
        impl.addIncludeExclude(cfgIndex, crIndex, creative.exchangeFilter);
    }

    void removeCreative(
            unsigned cfgIndex, unsigned crIndex, const Creative& creative)
    {
        impl.removeIncludeExclude(cfgIndex, crIndex, creative.exchangeFilter);
    }

    void filter(FilterState& state) const
    {
        state.narrowAllCreatives(impl.filter(state.request.exchange));
    }

private:
    CreativeIncludeExcludeFilter< CreativeListFilter<std::string> > impl;
};


/******************************************************************************/
/* CREATIVE EXCHANGE FILTER                                                   */
/******************************************************************************/

/** \todo Find a way to write an efficient version of this. */
struct CreativeExchangeFilter : public IterativeFilter<CreativeExchangeFilter>
{
    static constexpr const char* name = "CreativeExchange";
    unsigned priority() const { return Priority::CreativeExchange; }

    void filter(FilterState& state) const
    {
        // no exchange connector means evertyhing gets filtered out.
        if (!state.exchange) {
            state.narrowAllCreatives(CreativeMatrix());
            return;
        }

        CreativeMatrix creatives;

        for (size_t cfgId = state.configs().next();
             cfgId < state.configs().size();
             cfgId = state.configs().next(cfgId+1))
        {
            const auto& config = *configs[cfgId];

            for (size_t crId = 0; crId < config.creatives.size(); ++crId) {
                const auto& creative = config.creatives[crId];

                auto exchangeInfo = getExchangeInfo(state, creative);
                if (!exchangeInfo.first) continue;

                bool ret = state.exchange->bidRequestCreativeFilter(
                        state.request, config, exchangeInfo.second);

                if (ret) creatives.set(crId, cfgId);

            }
        }

        state.narrowAllCreatives(creatives);
    }

private:

    const std::pair<bool, void*>
    getExchangeInfo(const FilterState& state, const Creative& creative) const
    {
        auto it = creative.providerData.find(state.exchange->exchangeName());

        if (it == creative.providerData.end())
            return std::make_pair(false, nullptr);
        return std::make_pair(true, it->second.get());
    }
};


/******************************************************************************/
/* CREATIVE SEGEMENTS FILTER                                                   */
/******************************************************************************/

struct CreativeSegmentsFilter : public CreativeFilter<CreativeSegmentsFilter>
{
    static constexpr const char* name = "CreativeSegments";

    unsigned priority() const { return Priority::CreativeSegments; }

    void addCreative(unsigned cfgIndex, unsigned crIndex,
                       const Creative& creative)
    {
        setCreative(cfgIndex, crIndex, creative, true);
    }

    void removeCreative(unsigned cfgIndex, unsigned crIndex,
                           const Creative& creative)
    {
        setCreative(cfgIndex, crIndex, creative, false);
    }

    virtual void setCreative(unsigned configIndex, unsigned crIndex,
                     const Creative& creative, bool value);

    void filter(FilterState& state) const ;

private:

    struct SegmentData
    {
        CreativeIncludeExcludeFilter<CreativeSegmentListFilter> ie;
        CreativeMatrix excludeIfNotPresent;
    };

    std::unordered_map<std::string, SegmentData> data;
    std::unordered_set<std::string> excludeIfNotPresent;

};


/******************************************************************************/
/* CREATIVE PMP FILTER                                                   */
/******************************************************************************/

struct CreativePMPFilter : public CreativeFilter<CreativePMPFilter>
{
    static constexpr const char* name = "CreativePMP";
    unsigned priority() const { return Priority::CreativePMP; }


    void addCreative(
            unsigned cfgIndex, unsigned crIndex, const Creative& creative)
    {
        dealFilter[Datacratic::Id(creative.dealId)].set(crIndex, cfgIndex);
    }

    void removeCreative(
            unsigned cfgIndex, unsigned crIndex, const Creative& creative)
    {
        dealFilter[Datacratic::Id(creative.dealId)].reset(crIndex, cfgIndex);
    }

    void filterImpression(
            FilterState& state, unsigned impIndex, const AdSpot& imp) const
    {
        CreativeMatrix creatives;
        if (imp.pmp && imp.pmp->privateAuction.val == 1){
            const auto pmp = *imp.pmp;
            for (const auto& deal : pmp.deals) 
                creatives |= get(deal.id);
        }else {
            creatives |= get(Datacratic::Id("")); //If filter is not set its a No-Deal agent
        }

        state.narrowCreativesForImp(impIndex, creatives);
    }

private:
    std::unordered_map<Datacratic::Id, CreativeMatrix> dealFilter;

    CreativeMatrix get(const Datacratic::Id dealId) const
    {
        auto it = dealFilter.find(dealId);
        return it == dealFilter.end() ? CreativeMatrix() : it->second;
    }
};

} // namespace RTBKIT

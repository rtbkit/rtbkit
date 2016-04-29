/** utils.h                                 -*- C++ -*-
    Rémi Attab, 22 Aug 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Random utilities to write filter tests.

*/

#pragma once

#include "rtbkit/core/agent_configuration/include_exclude.h"
#include "rtbkit/common/exchange_connector.h"
#include "rtbkit/common/segments.h"
#include "rtbkit/common/filter.h"
#include "jml/utils/smart_ptr_utils.h"

#include <boost/test/unit_test.hpp>
#include <string>
#include <iostream>

namespace RTBKIT {

namespace Test {

/******************************************************************************/
/* FILTER EXCHANGE CONNECTOR                                                  */
/******************************************************************************/

struct FilterExchangeConnector : public ExchangeConnector
{
    FilterExchangeConnector(const std::string& name) :
        ExchangeConnector(name), name(name)
    {}

    std::string exchangeName() const { return name; }

    void configure(const Json::Value& parameters) {}
    void enableUntil(Date date) {}

private:
    std::string name;
};

/******************************************************************************/
/* TITLE                                                                      */
/******************************************************************************/

void title(const std::string& title)
{
    std::string padding(80 - 4 - title.size(), '-');
    std::cerr << "[ " << title << " ]" << padding << std::endl;
}


/******************************************************************************/
/* CHECK                                                                      */
/******************************************************************************/

void check(
        const ConfigSet& configs,
        const std::initializer_list<size_t>& expected)
{
    ConfigSet ex;
    for (size_t cfg : expected) ex.set(cfg);

    ConfigSet diff = configs;
    diff ^= ex;

    if (diff.empty()) return;

    std::cerr << "val=" << configs.print() << std::endl
        << "exp=" << ex.print() << std::endl
        << "dif=" << diff.print() << std::endl;
    BOOST_CHECK(false);
}

void check(
        const CreativeMatrix& creatives,
        const std::vector< std::vector<size_t> >& expected)
{
    CreativeMatrix exp;
    for (size_t cr = 0; cr < expected.size(); ++cr)
        for (size_t cfg : expected[cr])
            exp.set(cr, cfg);

    CreativeMatrix diff = creatives;
    diff ^= exp;

    if (diff.empty()) return;

    std::cerr << "val=" << creatives.print() << std::endl
        << "exp=" << exp.print() << std::endl
        << "dif=" << diff.print() << std::endl;
    BOOST_CHECK(false);
}

// Runs the filter on request and checks the result against expected for
// impression imp. Expected follows the same conventions as CreativeMatrix is
// therefor a creative major, list of config ids
// (eg. expected[creativeId][configId]).
void check(
        const FilterBase& filter,
        const BidRequest& request,
        const CreativeMatrix& creatives,
        unsigned imp,
        const std::vector< std::vector<size_t> >& expected)
{
    FilterExchangeConnector conn("bob");

    FilterState state(request, &conn, creatives);
    filter.filter(state);

    check(state.creatives(imp), expected);
}

// Check function used for static filters
void check(
        const FilterBase& filter,
        BidRequest& request,
        const string exchangeName,
        const ConfigSet& mask,
        const initializer_list<size_t>& exp)
{
    FilterExchangeConnector conn(exchangeName);

    /* Note that some filters depends on the bid request's exchange field while
       others depend on the exchange connector's name. Now you might think that
       they're always the same but you'd be wrong. To alleviate my endless pain
       on this subject, let's just fudge it here and call it a day.
     */
    request.exchange = exchangeName;

    // A bid request without ad spots doesn't really make any sense and will
    // accidently make state.configs() to return an empty set.
    request.imp.emplace_back();

    CreativeMatrix activeConfigs;
    for (size_t i = mask.next(); i < mask.size(); i = mask.next(i+1))
        activeConfigs.setConfig(i, 1);

    FilterState state(request, &conn, activeConfigs);

    filter.filter(state);
    check(state.configs() & mask, exp);
}


/******************************************************************************/
/* IE                                                                         */
/******************************************************************************/

template<typename T, typename List = std::vector<T> >
IncludeExclude<T, List>
ie(     const std::initializer_list<T>& includes,
        const std::initializer_list<T>& excludes)
{
    IncludeExclude<T, List> ie;
    for (const auto& v : includes) ie.include.push_back(v);
    for (const auto& v : excludes) ie.exclude.push_back(v);
    return ie;
}

template<typename T, typename List = std::vector<T> >
IncludeExclude<T, List> ie()
{
    return IncludeExclude<T, List>();
}

/******************************************************************************/
/* SEGMENT                                                                    */
/******************************************************************************/

void segmentImpl(SegmentList& seg) {}

template<typename Arg>
void segmentImpl(SegmentList& seg, Arg&& arg)
{
    seg.add(std::forward<Arg>(arg));
}

template<typename Arg, typename... Args>
void segmentImpl(SegmentList& seg, Arg&& arg, Args&&... rest)
{
    segmentImpl(seg, std::forward<Arg>(arg));
    segmentImpl(seg, std::forward<Args>(rest)...);
}

template<typename... Args>
SegmentList segment(Args&&... args)
{
    SegmentList seg;
    segmentImpl(seg, std::forward<Args>(args)...);
    seg.sort();
    return seg;
}

void addSegment(
        BidRequest& br,
        const string& name,
        const SegmentList& segment)
{
    br.segments[name] = make_shared<SegmentList>(segment);
};

void addSegment(
        AgentConfig& config,
        const string& name,
        bool excludeIfNotPresent,
        const SegmentList& includes,
        const SegmentList& excludes,
        const IncludeExclude<string>& exchangeIE)
{
    AgentConfig::SegmentInfo seg;

    seg.excludeIfNotPresent = excludeIfNotPresent;
    seg.include = includes;
    seg.exclude = excludes;
    seg.applyToExchanges = exchangeIE;

    config.segments[name] = seg;
};

/******************************************************************************/
/* IMPRESSION                                                                 */
/******************************************************************************/
void addImp(
        BidRequest& request,
        OpenRTB::AdPosition::Vals pos,
        const std::initializer_list<Format>& formats)
{
    AdSpot imp;
    for (const auto& format : formats) imp.formats.push_back(format);
    imp.position.val = pos;
    request.imp.push_back(imp);
}


void addImp(
        BidRequest& request,
        OpenRTB::AdPosition::Vals pos,
        const std::string & ext)
{
    AdSpot imp;
    imp.ext = Json::parse(ext);
    imp.position.val = pos;
    request.imp.push_back(imp);
}

/******************************************************************************/
/* ADD/REMOVE CONFIG                                                          */
/******************************************************************************/

void addConfig(FilterBase& filter, unsigned cfgIndex, AgentConfig& cfg)
{
    filter.addConfig(cfgIndex, ML::make_unowned_sp(cfg));
}

void addConfig(
        FilterBase& filter,
        unsigned cfgIndex, AgentConfig& cfg,
        CreativeMatrix& creatives)
{
    addConfig(filter, cfgIndex, cfg);
    creatives.setConfig(cfgIndex, cfg.creatives.size());
}

void removeConfig(FilterBase& filter, unsigned cfgIndex, AgentConfig& cfg)
{
    filter.removeConfig(cfgIndex, ML::make_unowned_sp(cfg));
}

void removeConfig(
        FilterBase& filter,
        unsigned cfgIndex, AgentConfig& cfg,
        CreativeMatrix& creatives)
{
    removeConfig(filter, cfgIndex, cfg);
    creatives.resetConfig(cfgIndex);
}

} // namespace TEST

} // namespace RTBKIT

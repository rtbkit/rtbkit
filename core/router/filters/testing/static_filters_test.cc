/** static_filters_test.cc                                 -*- C++ -*-
    Rémi Attab, 22 Aug 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Tests for the static filters.

    Note that these tests assume that the generic filters work properly so we
    don't waste time constructing complicated set of include/exclude statements.

    \todo HourOfTheWeek
 */

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "utils.h"
#include "rtbkit/core/router/filters/static_filters.h"
#include "rtbkit/core/agent_configuration/agent_config.h"
#include "rtbkit/common/bid_request.h"
#include "rtbkit/common/exchange_connector.h"
#include "jml/utils/vector_utils.h"

#include <boost/test/unit_test.hpp>

using namespace std;
using namespace ML;
using namespace Datacratic;

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

void
add(    AgentConfig& config,
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


void
add(    BidRequest& br,
        const string& name,
        const SegmentList& segment)
{
    br.segments[name] = make_shared<SegmentList>(segment);
};


/** Simple test to check that all a config will fail if one of its segment
    fails. Nothing to interesting really.
 */
BOOST_AUTO_TEST_CASE( segmentFilter_simple )
{
    SegmentsFilter filter;
    ConfigSet mask;

    auto doCheck = [&] (
            BidRequest& request,
            const string& exchangeName,
            const initializer_list<size_t>& expected)
    {
        check(filter, request, exchangeName, mask, expected);
    };

    AgentConfig c0;
    add(c0, "seg1", false, segment(1), segment(), ie<string>());
    add(c0, "seg2", false, segment(2), segment(), ie<string>());
    add(c0, "seg3", false, segment(3), segment(), ie<string>());

    BidRequest r0;
    add(r0, "seg1", segment(1));
    add(r0, "seg2", segment(2));
    add(r0, "seg3", segment(3));
    add(r0, "seg4", segment(4));

    BidRequest r1;
    add(r1, "seg1", segment(0));
    add(r1, "seg2", segment(2));
    add(r1, "seg3", segment(3));
    add(r1, "seg4", segment(4));

    BidRequest r2;
    add(r2, "seg1", segment(1));
    add(r2, "seg2", segment(0));
    add(r2, "seg3", segment(3));
    add(r2, "seg4", segment(4));

    BidRequest r3;
    add(r3, "seg1", segment(1));
    add(r3, "seg2", segment(2));
    add(r3, "seg3", segment(0));
    add(r3, "seg4", segment(4));

    title("segment-simple-1");
    addConfig(filter, 0, c0); mask.set(0);

    doCheck(r0, "ex0", { 0 });
    doCheck(r1, "ex0", { });
    doCheck(r2, "ex0", { });
    doCheck(r3, "ex0", { });
}

/** Tests for the endlessly confusing excludeIfNotPresent attribute. */
BOOST_AUTO_TEST_CASE( segmentFilter_excludeIfNotPresent )
{
    SegmentsFilter filter;
    ConfigSet mask;

    auto doCheck = [&] (
            BidRequest& request,
            const string& exchangeName,
            const initializer_list<size_t>& expected)
    {
        check(filter, request, exchangeName, mask, expected);
    };

    AgentConfig c0;

    AgentConfig c1;
    add(c1, "seg1", false, segment(), segment(), ie<string>());

    AgentConfig c2;
    add(c2, "seg1", true, segment(), segment(), ie<string>());

    BidRequest r0;

    BidRequest r1;
    add(r1, "seg1", segment("a"));

    BidRequest r2;
    add(r2, "seg1", segment("a"));
    add(r2, "seg2", segment("b"));

    BidRequest r3;
    add(r3, "seg2", segment("b"));
    add(r3, "seg3", segment("c"));

    title("segment-excludeIfNotPresent-1");
    addConfig(filter, 0, c0); mask.set(0);
    addConfig(filter, 1, c1); mask.set(1);
    addConfig(filter, 2, c2); mask.set(2);

    doCheck(r0, "ex0", { 0, 1 });
    doCheck(r1, "ex0", { 0, 1, 2 });
    doCheck(r2, "ex0", { 0, 1, 2 });
    doCheck(r3, "ex0", { 0, 1 });

    title("segment-excludeIfNotPresent-2");
    removeConfig(filter, 0, c0); mask.reset(0);

    doCheck(r0, "ex0", { 1 });
    doCheck(r1, "ex0", { 1, 2 });
    doCheck(r2, "ex0", { 1, 2 });
    doCheck(r3, "ex0", { 1 });

    title("segment-excludeIfNotPresent-3");
    removeConfig(filter, 2, c2); mask.reset(2);

    doCheck(r0, "ex0", { 1 });
    doCheck(r1, "ex0", { 1 });
    doCheck(r2, "ex0", { 1 });
    doCheck(r3, "ex0", { 1 });
}

/** The logic being tested here is a little wonky.

    Short version, the result of a single segment filter should be ignored
    (whether it succeeded or not) if the exchange filter failed for that
    segment.

    This applies on both regular inclue/exclude segment filtering as well as
    excludeIfNotPresent filtering (just to keep you on your toes).
 */
BOOST_AUTO_TEST_CASE( segmentFilter_exchange )
{
    SegmentsFilter filter;
    ConfigSet mask;

    auto doCheck = [&] (
            BidRequest& request,
            const string& exchangeName,
            const initializer_list<size_t>& expected)
    {
        check(filter, request, exchangeName, mask, expected);
    };

    AgentConfig c0;
    add(c0, "seg1", false, segment(1), segment(), ie<string>({ "ex1" }, {}));

    AgentConfig c1;
    add(c1, "seg1", false, segment(1), segment(), ie<string>({ "ex2" }, {}));

    AgentConfig c2;
    add(c2, "seg1", false, segment(1), segment(), ie<string>({ "ex1" }, {}));
    add(c2, "seg2", false, segment(2), segment(), ie<string>({ "ex2" }, {}));

    // Control case with no exchange filters.
    AgentConfig c3;
    add(c3, "seg1", false, segment(1), segment(), ie<string>());

    // excludeIfNotPresent = true && segment does exist
    AgentConfig c4;
    add(c4, "seg1", true, segment(), segment(), ie<string>({ "ex1" }, {}));
    add(c4, "seg2", true, segment(), segment(), ie<string>({ "ex3" }, {}));

    // excludeIfNotPresent = true && segment doesn't exist.
    AgentConfig c5;
    add(c5, "seg3", true, segment(), segment(), ie<string>({ "ex1" }, {}));
    add(c5, "seg4", true, segment(), segment(), ie<string>({ "ex3" }, {}));

    BidRequest r0;
    add(r0, "seg1", segment(1));
    add(r0, "seg2", segment(1));

    BidRequest r1;
    add(r1, "seg1", segment(2));
    add(r1, "seg2", segment(2));

    title("segment-exchange-1");
    addConfig(filter, 0, c0); mask.set(0);
    addConfig(filter, 1, c1); mask.set(1);
    addConfig(filter, 2, c2); mask.set(2);
    addConfig(filter, 3, c3); mask.set(3);
    addConfig(filter, 4, c4); mask.set(4);
    addConfig(filter, 5, c5); mask.set(5);

    doCheck(r0, "ex1", { 0, 1, 2, 3, 4 });
    doCheck(r1, "ex1", { 1, 4 });

    doCheck(r0, "ex2", { 0, 1, 3, 4, 5 });
    doCheck(r1, "ex2", { 0, 2, 4, 5 });

    doCheck(r0, "ex3", { 0, 1, 2, 3, 4 });
    doCheck(r1, "ex3", { 0, 1, 2, 4 });

    title("segment-exchange-2");
    removeConfig(filter, 2, c2); mask.reset(2);

    doCheck(r0, "ex1", { 0, 1, 3, 4 });
    doCheck(r1, "ex1", { 1, 4 });

    doCheck(r0, "ex2", { 0, 1, 3, 4, 5 });
    doCheck(r1, "ex2", { 0, 4, 5 });

    doCheck(r0, "ex3", { 0, 1, 3, 4 });
    doCheck(r1, "ex3", { 0, 1, 4 });

    title("segment-exchange-3");
    removeConfig(filter, 5, c5); mask.reset(5);

    doCheck(r0, "ex1", { 0, 1, 3, 4 });
    doCheck(r1, "ex1", { 1, 4 });

    doCheck(r0, "ex2", { 0, 1, 3, 4 });
    doCheck(r1, "ex2", { 0, 4 });

    doCheck(r0, "ex3", { 0, 1, 3, 4 });
    doCheck(r1, "ex3", { 0, 1, 4 });

    title("segment-exchange-4");
    removeConfig(filter, 0, c0); mask.reset(0);

    doCheck(r0, "ex1", { 1, 3, 4 });
    doCheck(r1, "ex1", { 1, 4 });

    doCheck(r0, "ex2", { 1, 3, 4 });
    doCheck(r1, "ex2", { 4 });

    doCheck(r0, "ex3", { 1, 3, 4 });
    doCheck(r1, "ex3", { 1, 4 });
}

BOOST_AUTO_TEST_CASE( userPartition )
{
    UserPartitionFilter filter;
    ConfigSet mask;

    auto doCheck = [&] (
            BidRequest& request,
            const string& exchangeName,
            const initializer_list<size_t>& expected)
    {
        check(filter, request, exchangeName, mask, expected);
    };

    auto setReq = [] (BidRequest& request, Id eid, string ip, string ua) {
        request.userIds.exchangeId = eid;
        request.userIds.providerId = Id(0);
        request.ipAddress = ip;
        request.userAgent = ua;
    };

    auto setCfg = [] (
            AgentConfig& config, UserPartition::HashOn hashOn, int modulus)
    {
        config.userPartition.hashOn = hashOn;
        config.userPartition.modulus = modulus;
    };

    typedef pair<int, int> PairT;
    auto add = [] (AgentConfig& config, const initializer_list<PairT>& ranges) {

        // This is terible but the UserPartition constructor adds a 0-1 range.
        // While it's tempting to fix it, I'm pretty sure there's some code
        // somewhere that rely on this arcanne behaviour so I have to
        // investigate first.
        config.userPartition.includeRanges.clear();

        for (const auto& range : ranges) {
            config.userPartition.includeRanges.emplace_back(
                    range.first, range.second);
        }
    };

    AgentConfig c0;
    setCfg(c0, UserPartition::NONE, 10);

    AgentConfig c1;
    setCfg(c1, UserPartition::EXCHANGEID, 8);
    add(c1, { {0, 5} });

    AgentConfig c2;
    setCfg(c2, UserPartition::EXCHANGEID, 16);
    add(c2, { {5, 8}, {12, 16} });

    AgentConfig c3;
    setCfg(c3, UserPartition::IPUA, 2);
    add(c3, { {0, 1} });

    AgentConfig c4;
    setCfg(c4, UserPartition::IPUA, 2);
    add(c4, { {1, 2} });


    // Since the following strings are hashed, any changes to them will affect
    // the outcome of the test. I recomend, you don't do that.

    BidRequest r0; setReq(r0, Id(0),  "1.1.1.1", "ua-0"); // eid = 15, ipua = 0
    BidRequest r1; setReq(r1, Id(1),  "2.2.2.2", "ua-0"); // eid = 4,  ipua = 1
    BidRequest r2; setReq(r2, Id(1),  "1.1.1.1", "ua-1"); // eid = 4,  ipua = 0
    BidRequest r3; setReq(r3, Id(0),  "2.2.2.2", "ua-1"); // eid = 15, ipua = 1
    BidRequest r4; setReq(r4, Id(""), "1.1.1.1", "ua-1"); // eid = -,  ipua = 0
    BidRequest r5; setReq(r5, Id(0),  "",        "ua-1"); // eid = 15, ipua = 1
    BidRequest r6; setReq(r6, Id(0),  "1.1.1.1", "");     // eid = 15, ipua = 0
    BidRequest r7; setReq(r7, Id(0),  "",        "");     // eid = 15, ipua = -


    title("userPartition-1");
    addConfig(filter, 0, c0); mask.set(0);
    addConfig(filter, 1, c1); mask.set(1);
    addConfig(filter, 2, c2); mask.set(2);
    addConfig(filter, 3, c3); mask.set(3);
    addConfig(filter, 4, c4); mask.set(4);

    doCheck(r0, "ex1", { 0, 2, 3 });
    doCheck(r1, "ex1", { 0, 1, 4 });
    doCheck(r2, "ex1", { 0, 1, 3 });
    doCheck(r3, "ex1", { 0, 2, 4 });
    doCheck(r4, "ex1", { 0, 3 });
    doCheck(r5, "ex1", { 0, 2, 4 });
    doCheck(r6, "ex1", { 0, 2, 3 });
    doCheck(r7, "ex1", { 0, 2 });

    title("userPartition-2");
    removeConfig(filter, 0, c0); mask.reset(0);

    doCheck(r0, "ex1", { 2, 3 });
    doCheck(r1, "ex1", { 1, 4 });
    doCheck(r2, "ex1", { 1, 3 });
    doCheck(r3, "ex1", { 2, 4 });
    doCheck(r4, "ex1", { 3 });
    doCheck(r5, "ex1", { 2, 4 });
    doCheck(r6, "ex1", { 2, 3 });
    doCheck(r7, "ex1", { 2 });

    title("userPartition-3");
    removeConfig(filter, 3, c3); mask.reset(3);

    doCheck(r0, "ex1", { 2 });
    doCheck(r1, "ex1", { 1, 4 });
    doCheck(r2, "ex1", { 1 });
    doCheck(r3, "ex1", { 2, 4 });
    doCheck(r4, "ex1", { });
    doCheck(r5, "ex1", { 2, 4 });
    doCheck(r6, "ex1", { 2 });
    doCheck(r7, "ex1", { 2 });

    title("userPartition-4");
    removeConfig(filter, 1, c1); mask.reset(1);

    doCheck(r0, "ex1", { 2 });
    doCheck(r1, "ex1", { 4 });
    doCheck(r2, "ex1", { });
    doCheck(r3, "ex1", { 2, 4 });
    doCheck(r4, "ex1", { });
    doCheck(r5, "ex1", { 2, 4 });
    doCheck(r6, "ex1", { 2 });
    doCheck(r7, "ex1", { 2 });
}

BOOST_AUTO_TEST_CASE( exchangeName )
{
    ExchangeNameFilter filter;
    ConfigSet mask;

    auto doCheck = [&] (
            BidRequest& request,
            const string& exchangeName,
            const initializer_list<size_t>& expected)
    {
        check(filter, request, exchangeName, mask, expected);
    };

    AgentConfig c0;
    c0.exchangeFilter = ie<string>({ "appnexus", "context_web", "casale" }, {});

    BidRequest req;

    title("exchangeName-1");
    addConfig(filter, 0, c0); mask.set(0);

    doCheck(req, "appnexus", { 0 });
    doCheck(req, "adx", { });
}

BOOST_AUTO_TEST_CASE( requiredIds )
{
    RequiredIdsFilter filter;
    ConfigSet mask;

    auto doCheck = [&] (
            BidRequest& request,
            const string& exchangeName,
            const initializer_list<size_t>& expected)
    {
        check(filter, request, exchangeName, mask, expected);
    };

    auto set = [] (
            BidRequest& request,
            const std::initializer_list<std::string>& domains)
    {
        for (const auto& domain : domains)
            request.userIds.add(Id(0), domain);
    };

    AgentConfig c0;
    AgentConfig c1; c1.requiredIds = { "d0", "d1" };
    AgentConfig c2; c2.requiredIds = { "d1", "d2" };
    AgentConfig c3; c3.requiredIds = { "d2" };
    AgentConfig c4; c4.requiredIds = { "d3" };

    BidRequest r0;
    BidRequest r1; set(r1, { "d1" });
    BidRequest r2; set(r2, { "d1", "d2" });
    BidRequest r3; set(r3, { "d2", "d3" });
    BidRequest r4; set(r4, { "d0", "d1", "d2" });
    BidRequest r5; set(r5, { "d5" });

    title("requiredIds-1");
    addConfig(filter, 0, c0); mask.set(0);
    addConfig(filter, 1, c1); mask.set(1);
    addConfig(filter, 2, c2); mask.set(2);
    addConfig(filter, 3, c3); mask.set(3);
    addConfig(filter, 4, c4); mask.set(4);

    doCheck(r0, "ex1", { 0 });
    doCheck(r1, "ex1", { 0 });
    doCheck(r2, "ex1", { 0, 2, 3 });
    doCheck(r3, "ex1", { 0, 3, 4 });
    doCheck(r4, "ex1", { 0, 1, 2, 3 });
    doCheck(r5, "ex1", { 0 });

    title("requiredIds-2");
    removeConfig(filter, 2, c1); mask.reset(2);

    doCheck(r0, "ex1", { 0 });
    doCheck(r1, "ex1", { 0 });
    doCheck(r2, "ex1", { 0, 3 });
    doCheck(r3, "ex1", { 0, 3, 4 });
    doCheck(r4, "ex1", { 0, 1, 3 });
    doCheck(r5, "ex1", { 0 });

    title("requiredIds-3");
    removeConfig(filter, 0, c1); mask.reset(0);

    doCheck(r0, "ex1", { });
    doCheck(r1, "ex1", { });
    doCheck(r2, "ex1", { 3 });
    doCheck(r3, "ex1", { 3, 4 });
    doCheck(r4, "ex1", { 1, 3 });
    doCheck(r5, "ex1", { });

    title("requiredIds-4");
    removeConfig(filter, 3, c1); mask.reset(3);

    doCheck(r0, "ex1", { });
    doCheck(r1, "ex1", { });
    doCheck(r2, "ex1", { });
    doCheck(r3, "ex1", { 4 });
    doCheck(r4, "ex1", { 1 });
    doCheck(r5, "ex1", { });
}

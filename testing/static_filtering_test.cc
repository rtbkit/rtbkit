/** static_filtering_test.cc                                 -*- C++ -*-
    Rémi Attab, 12 Jun 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Various tests for the static filters

*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "rtbkit/common/bid_request.h"
#include "rtbkit/core/agent_configuration/agent_config.h"
#include "rtbkit/core/router/router_types.h"
#include "rtbkit/testing/generic_exchange_connector.h"
#include "soa/jsoncpp/value.h"

#include <boost/test/unit_test.hpp>
#include <utility>

using namespace std;
using namespace RTBKIT;


/******************************************************************************/
/* UTILITIES                                                                  */
/******************************************************************************/

void addSpot(BidRequest& request, Id id, size_t height, size_t width)
{
    AdSpot spot;
    spot.id = id;
    spot.formats.emplace_back(height, width);
    request.imp.push_back(spot);
}

BidRequest basicRequest()
{
    BidRequest request;
    addSpot(request, Id(0), 160, 600);
    return request;
}

AgentConfig basicConfig()
{
    AgentConfig config;

    config.account = {"hello", "world"};
    config.creatives.push_back(Creative::sampleLB);
    config.creatives.push_back(Creative::sampleWS);
    config.creatives.push_back(Creative::sampleBB);

    return config;
}

pair<bool, AgentStats>
check(  AgentConfig config,
        const BidRequest& request,
        const ExchangeConnector& exchange = GenericExchangeConnector())
{
    auto cpgCompat = exchange.getCampaignCompatibility(config, true);
    BOOST_CHECK(cpgCompat.isCompatible);
    config.providerData[exchange.exchangeName()] = cpgCompat.info;

    for (auto& creative : config.creatives) {
        auto crCompat = exchange.getCreativeCompatibility(creative, true);
        BOOST_CHECK(crCompat.isCompatible);
        creative.providerData[exchange.exchangeName()] = crCompat.info;
    }

    AgentConfig::RequestFilterCache cache(request);
    AgentStats stats;

    BiddableSpots spots = config.isBiddableRequest(&exchange, request, stats, cache);
    return make_pair(!spots.empty(), stats);
}

void dumpError(
        const string& name,
        const pair<bool, AgentStats>& ret,
        const string& expected)
{
    cerr << name << ": expected=[ " << expected << " ] got=[ ";
    if (ret.first) {
        cerr << "passed ]" << endl;
        return;
    }

    Json::Value json = ret.second.toJson();
    for (auto it = json.begin(), end = json.end(); it != end; ++it) {
        string memberName = it.memberName();

        if (memberName.find("filter_") != 0) continue;
        if (!(*it).asInt()) continue;

        cerr << "filtered on " << memberName << "]" << endl;
        return;
    }

    cerr << "unknown ]" << endl;
    cerr << json.toString();
}

#define OK(_ret_,_name_)                                       \
    do {                                                       \
        auto ret = (_ret_);                                    \
        if (!ret.first) dumpError(_name_, ret, "passed");      \
    } while (false)

#define FILTERED(_ret_,_stat_,_name_)                                   \
    do {                                                                \
        auto ret = (_ret_);                                             \
        if (ret.first)          dumpError(_name_, ret, "filtered on " #_stat_); \
        if (!ret.second._stat_) dumpError(_name_, ret, "filtered on " #_stat_); \
    } while(false)


/******************************************************************************/
/* TESTS                                                                      */
/******************************************************************************/

BOOST_AUTO_TEST_CASE( smoke )
{
    FILTERED(check(basicConfig(), BidRequest()), noSpots, "no spots");
    OK(check(basicConfig(), basicRequest()), "1 spot");
}

BOOST_AUTO_TEST_CASE( segmentFiltering )
{
}

/* bidder_test.cc
   Eric Robert, 10 April 2014
   Copyright (c) 2013 Datacratic.  All rights reserved.
*/


#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "jml/utils/testing/watchdog.h"
#include "rtbkit/common/win_cost_model.h"
#include "rtbkit/plugins/exchange/openrtb_exchange_connector.h"
#include "rtbkit/plugins/exchange/rtbkit_exchange_connector.h"
#include "rtbkit/testing/bid_stack.h"

using namespace Datacratic;
using namespace RTBKIT;

BOOST_AUTO_TEST_CASE( bidder_http_test )
{
    ML::Watchdog watchdog(10.0);

    Json::Value upstreamRouterConfig;
    upstreamRouterConfig[0]["exchangeType"] = "openrtb";

    Json::Value downstreamRouterConfig;
    downstreamRouterConfig[0]["exchangeType"] = "rtbkit";

    Json::Value upstreamBidderConfig;
    upstreamBidderConfig["type"] = "http";
    upstreamBidderConfig["adserver"]["winPort"] = 18143;
    upstreamBidderConfig["adserver"]["eventPort"] = 18144;

    Json::Value downstreamBidderConfig;
    downstreamBidderConfig["type"] = "agents";

    BidStack upstreamStack;
    BidStack downstreamStack;

    downstreamStack.runThen(
        downstreamRouterConfig, downstreamBidderConfig,
        USD_CPM(10), 0, [&](Json::Value const & json) {

        std::cerr << json << std::endl;
        const auto &bids = json["workers"][0]["bids"];
        const auto &wins = json["workers"][0]["wins"];
        const auto &events = json["workers"][0]["events"];

        // We don't use them for now but we might later on if we decide to extend the test
        (void) wins;
        (void) events;

        auto url = bids["url"].asString();
        auto resource = bids.get("resource", "/").asString();
        upstreamBidderConfig["router"]["host"] = "http://" + url;
        upstreamBidderConfig["router"]["path"] = resource;
        upstreamBidderConfig["adserver"]["host"] = "";


        upstreamStack.runThen(
            upstreamRouterConfig, upstreamBidderConfig, USD_CPM(20), 10,
            [&](Json::Value const &json)
        {
            // Since the FilterRegistry is shared amongst the routers,
            // the ExternalIdsCreativeExchangeFilter will also be added
            // to the upstream stack FilterPool. Thus we remove it before
            // starting the MockExchange to avoid being filtered
            upstreamStack.services.router->filters.removeFilter(
                ExternalIdsCreativeExchangeFilter::name);

            auto proxies = std::make_shared<ServiceProxies>();
            MockExchange mockExchange(proxies);
            mockExchange.start(json);
        });
    });


    auto upstreamEvents = upstreamStack.proxies->events->get(std::cerr);
    int upstreamBidCount = upstreamEvents["router.bid"];
    std::cerr << "UPSTREAM BID COUNT=" << upstreamBidCount << std::endl;

    auto downstreamEvents = downstreamStack.proxies->events->get(std::cerr);
    int downstreamBidCount = downstreamEvents["router.bid"];
    std::cerr << "DOWNSTREAM BID COUNT=" << downstreamBidCount << std::endl;

    //BOOST_CHECK_EQUAL(bpcEvents["router.cummulatedBidPrice"], count * 1000);
    //BOOST_CHECK_EQUAL(bpcEvents["router.cummulatedAuthorizedPrice"], count * 505);
}


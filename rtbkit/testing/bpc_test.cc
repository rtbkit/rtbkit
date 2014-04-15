/* bpc_test.cc
   Eric Robert, 10 April 2014
   Copyright (c) 2013 Datacratic.  All rights reserved.
*/


#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "jml/utils/testing/watchdog.h"
#include "rtbkit/common/win_cost_model.h"
#include "rtbkit/plugins/exchange/openrtb_exchange_connector.h"
#include "rtbkit/testing/bid_stack.h"

using namespace Datacratic;
using namespace RTBKIT;

BOOST_AUTO_TEST_CASE( bpc_http_test )
{
    ML::Watchdog watchdog(10.0);

    std::string configuration = ML::format(
        "[{"
            "\"exchangeType\":\"openrtb\""
        "}]");

    std::cout << configuration << std::endl;

    BidStack bidStack;
    BidStack bpcStack;

    bidStack.runThen(configuration, USD_CPM(10), 0, [&](Json::Value const & json) {
        const auto &bids = json["workers"][0]["bids"];
        auto url = bids["url"].asString();
        auto resource = bids.get("resource", "/").asString();
        std::cerr << url << resource << std::endl;
        bpcStack.useForwardingUri(url, resource);
        bpcStack.run(configuration, USD_CPM(20), 10);
    });


    auto bpcEvents = bpcStack.proxies->events->get(std::cerr);
    int bpcCount = bpcEvents["router.bid"];
    std::cerr << "BPC BID COUNT=" << bpcCount << std::endl;

    auto bidEvents = bidStack.proxies->events->get(std::cerr);
    int bidCount = bidEvents["router.bid"];
    std::cerr << "BID BID COUNT=" << bidCount << std::endl;

    //BOOST_CHECK_EQUAL(bpcEvents["router.cummulatedBidPrice"], count * 1000);
    //BOOST_CHECK_EQUAL(bpcEvents["router.cummulatedAuthorizedPrice"], count * 505);
}


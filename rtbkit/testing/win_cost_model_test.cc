/* win_cost_model_test.cc
   Eric Robert, 16 May 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.

   Test for the win cost model.
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

namespace {

Amount linearWinCostModel(WinCostModel const & model,
                          Bid const & bid,
                          Amount const & price)
{
    double m = model.data["m"].asDouble();
    Amount b = Amount::fromJson(model.data["b"]);
    return price * m + b;
}

struct TestExchangeConnector : public OpenRTBExchangeConnector {
    TestExchangeConnector(ServiceBase & owner,
                          const std::string & name)
        : OpenRTBExchangeConnector(owner, name) {
    }

    static std::string exchangeNameString() {
        return "test";
    }

    std::string exchangeName() const {
        return exchangeNameString();
    }

    WinCostModel
    getWinCostModel(Auction const & auction, AgentConfig const & agent) {
        Json::Value data;
        data["m"] = 0.5;
        data["b"] = MicroUSD(5.0).toJson();
        return WinCostModel("test", data);
    }

    // avoid dropping bids
    void setAcceptBidRequestProbability(double prob) {
    }
};

} // file scope

BOOST_AUTO_TEST_CASE( win_cost_model_test )
{
    ML::Watchdog watchdog(10.0);

    // register the exchange
    ExchangeConnector::registerFactory<TestExchangeConnector>();

    // register the win cost model
    WinCostModel::registerModel("test", linearWinCostModel);

    Json::Value routerConfig;
    routerConfig[0]["exchangeType"] = "test";

    Json::Value bidderConfig;
    bidderConfig["type"] = "agents";

    BidStack stack;
    stack.run(routerConfig, bidderConfig, USD_CPM(1.0), 10);

    auto events = stack.proxies->events->get(std::cerr);

    int count = events["router.bid"];

    BOOST_CHECK_EQUAL(events["router.cummulatedBidPrice"], count * 1000);
    BOOST_CHECK_EQUAL(events["router.cummulatedAuthorizedPrice"], count * 505);
}


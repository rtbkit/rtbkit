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
#include "rtbkit/plugins/bidder_interface/multi_bidder_interface.h"
#include "rtbkit/openrtb/openrtb_parsing.h"
#include "rtbkit/plugins/exchange/http_auction_handler.h"

using namespace Datacratic;
using namespace RTBKIT;
using namespace std;

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
        upstreamBidderConfig["adserver"]["host"] = "http://invalid-url-but-its-intended.com";


        upstreamStack.runThen(
            upstreamRouterConfig, upstreamBidderConfig, USD_CPM(20), 10,
            [&](Json::Value const &json)
        {
            // Since the FilterRegistry is shared amongst the routers,
            // the CreativeIdsExchangeFilter will also be added
            // to the upstream stack FilterPool. Thus we remove it before
            // starting the MockExchange to avoid being filtered
            upstreamStack.services.router->filters.removeFilter(
                CreativeIdsExchangeFilter::name);

            auto proxies = std::make_shared<ServiceProxies>();
            MockExchange mockExchange(proxies);
            mockExchange.start(json);
        });
    });


    auto upstreamEvents = upstreamStack.proxies->events->get(std::cerr);
    int upstreamBidCount = upstreamEvents["router.bid"];
    std::cerr << "UPSTREAM BID COUNT=" << upstreamBidCount << std::endl;
    BOOST_CHECK(upstreamBidCount > 0);

    auto downstreamEvents = downstreamStack.proxies->events->get(std::cerr);
    int downstreamBidCount = downstreamEvents["router.bid"];
    std::cerr << "DOWNSTREAM BID COUNT=" << downstreamBidCount << std::endl;
    BOOST_CHECK(downstreamBidCount > 0);

    //BOOST_CHECK_EQUAL(bpcEvents["router.cummulatedBidPrice"], count * 1000);
    //BOOST_CHECK_EQUAL(bpcEvents["router.cummulatedAuthorizedPrice"], count * 505);
}

BOOST_AUTO_TEST_CASE( bidder_http_test_nobid )
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
        USD_CPM(0), 0, [&](Json::Value const & json) {

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
        upstreamBidderConfig["adserver"]["host"] = "http://invalid-url-but-its-intended.com";


        upstreamStack.runThen(
            upstreamRouterConfig, upstreamBidderConfig, USD_CPM(20), 10,
            [&](Json::Value const &json)
        {
            upstreamStack.services.router->filters.removeFilter(
                CreativeIdsExchangeFilter::name);

            auto proxies = std::make_shared<ServiceProxies>();
            MockExchange mockExchange(proxies);
            mockExchange.start(json);
        });
    });


    auto upstreamEvents = upstreamStack.proxies->events->get(std::cerr);
    int upstreamBidCount = upstreamEvents["router.bid"];
    std::cerr << "UPSTREAM BID COUNT=" << upstreamBidCount << std::endl;
    BOOST_CHECK(upstreamBidCount > 0);

    auto downstreamEvents = downstreamStack.proxies->events->get(std::cerr);
    int downstreamBidCount = downstreamEvents["router.bid"];
    std::cerr << "DOWNSTREAM BID COUNT=" << downstreamBidCount << std::endl;
    BOOST_CHECK(downstreamBidCount > 0);
}

struct BiddingAgentOfDestiny : public TestAgent {
    BiddingAgentOfDestiny(std::shared_ptr<ServiceProxies> proxies,
                          const std::string &bidderInterface,
                          const std::string &name = "biddingAgentOfDestiny",
                          const AccountKey &accountKey =
                             AccountKey({"testCampaign", "testStrategy"}))
        : TestAgent(proxies, name, accountKey)
     {
         config.bidderInterface = bidderInterface;
     }

};

BOOST_AUTO_TEST_CASE( multi_bidder_test )
{
    Json::Value upstreamRouterConfig;
    upstreamRouterConfig[0]["exchangeType"] = "openrtb";

    Json::Value downstreamRouterConfig;
    downstreamRouterConfig[0]["exchangeType"] = "rtbkit";

    Json::Value upstreamBidderConfig = Json::parse(
            R"JSON(
            {
               "type": "multi",
               "interfaces": [
                   {
                       "iface.agents": { "type": "agents" }
                   },
                   {
                       "iface.http": {
                           "type": "http",

                           "router": {
                           },

                           "adserver": {
                               "winPort": 18143,
                               "eventPort": 18144
                           }
                       }
                   }
               ]
            }
            )JSON");

    Json::Value httpAgentConfig = Json::parse(
            R"JSON(
            {
                "account": ["dummy_account"],
                "bidProbability": 1,
                "creatives": [ { "width": 300, "height": 250, "id": 1 } ],
                "externalId": 1,
                "bidderInterface": "iface.http"
            }
            )JSON");

    Json::Value downstreamBidderConfig;
    downstreamBidderConfig["type"] = "agents";

    BidStack upstreamStack;
    BidStack downstreamStack;

    auto destinyAgent =
        std::make_shared<BiddingAgentOfDestiny>(
                upstreamStack.proxies,
                "iface.agents",
                "bidding_agent_of_destiny_1");

    auto destinyAgent2 =
        std::make_shared<BiddingAgentOfDestiny>(
                upstreamStack.proxies,
                "iface.agents",
                "bidding_agent_of_destiny_2",
                AccountKey({ "testCampaign", "testStrategy2" }));

    upstreamStack.addAgent(destinyAgent);
    upstreamStack.addAgent(destinyAgent2);

    downstreamStack.runThen(downstreamRouterConfig, downstreamBidderConfig, USD_CPM(10), 0,
                            [&](const Json::Value &json) {

        const auto &bids = json["workers"][0]["bids"];
        const auto &wins = json["workers"][0]["wins"];
        const auto &events = json["workers"][0]["events"];

        (void) wins;
        (void) events;

        auto url = bids["url"].asString();
        auto resource = bids.get("resource", "/").asString();
        auto &httpIface = upstreamBidderConfig["interfaces"][1]["iface.http"];
        httpIface["router"]["host"] = "http://" + url;
        httpIface["router"]["path"] = resource;
        httpIface["adserver"]["host"] = "http://invalid-url-but-its-intended.com";

        upstreamStack.runThen(upstreamRouterConfig, upstreamBidderConfig, USD_CPM(10),
                              100,
                              [&](const Json::Value &json) {
            upstreamStack.services.router->filters.removeFilter(
                CreativeIdsExchangeFilter::name);

            upstreamStack.postConfig("sample_http_config", httpAgentConfig);

            auto proxies = std::make_shared<ServiceProxies>();
            MockExchange mockExchange(proxies);
                mockExchange.start(json);
        });

        auto bidder = std::static_pointer_cast<MultiBidderInterface>(
                upstreamStack.services.router->bidder);

        std::cerr << std::endl;
        bidder->stats().dump(std::cerr);

   });

   auto upstreamEvents = upstreamStack.proxies->events->get(std::cerr);
   int upstreamBidCount = upstreamEvents["router.bid"];
   std::cerr << "UPSTREAM BID COUNT=" << upstreamBidCount << std::endl;
   BOOST_CHECK(upstreamBidCount > 0);

   auto downstreamEvents = downstreamStack.proxies->events->get(std::cerr);
   int downstreamBidCount = downstreamEvents["router.bid"];
   std::cerr << "DOWNSTREAM BID COUNT=" << downstreamBidCount << std::endl;
   BOOST_CHECK(downstreamBidCount > 0);
}

struct DummyExchangeConnector : public OpenRTBExchangeConnector
{
    DummyExchangeConnector(ServiceBase & owner, const std::string & name)
        : OpenRTBExchangeConnector(owner, name)
    { }

    DummyExchangeConnector(const std::string & name,
                             std::shared_ptr<ServiceProxies> proxies)
        : OpenRTBExchangeConnector(name, proxies)
    { }

    std::string exchangeName() const { return "dummy"; }

    std::shared_ptr<BidRequest>
    parseBidRequest(HttpAuctionHandler& connection,
                    const HttpHeader& header,
                    const std::string& payload) {
        auto request = OpenRTBExchangeConnector::parseBidRequest(connection, header, payload);

        for (const auto& imp: request->imp) {
            BOOST_CHECK(imp.ext.isMember("creative-ids"));
            BOOST_CHECK(imp.ext.isMember("external-ids"));
        }

        return request;
    }
};

// Test to validate that every impression in a single BidRequest gets tagged with
// the "external-ids" and "creative-ids" extension fields, by the HttpBidderInterface
BOOST_AUTO_TEST_CASE( test_http_bidder_multiple_impressions_tagging )
{
    ML::Watchdog watchdog(10.0);

    auto proxies = make_shared<ServiceProxies>();
    auto acs = make_shared<AgentConfigurationService>(proxies, "acs");
    acs->unsafeDisableMonitor();
    acs->init();
    acs->bindTcp();
    acs->start();

    Json::Value downstreamBidderConfig;
    downstreamBidderConfig["type"] = "agents";

    auto router = make_shared<Router>(proxies, "router");
    router->unsafeDisableMonitor();
    router->initBidderInterface(downstreamBidderConfig);
    router->init();
    router->setBanker(make_shared<NullBanker>(true));
    router->bindTcp();
    router->start();

    auto dummyExchange = new DummyExchangeConnector("dummyExchange", proxies);
    dummyExchange->start();
    dummyExchange->enableUntil(Date::positiveInfinity());
    router->addExchange(dummyExchange);
    router->initFilters();

    Json::Value upstreamRouterConfig;
    upstreamRouterConfig[0]["exchangeType"] = "openrtb";

    Json::Value upstreamBidderConfig;
    upstreamBidderConfig["type"] = "http";
    upstreamBidderConfig["adserver"]["winPort"] = 18143;
    upstreamBidderConfig["adserver"]["eventPort"] = 18144;

    upstreamBidderConfig["router"]["host"] = "http://127.0.0.1:" + to_string(dummyExchange->port());
    upstreamBidderConfig["router"]["path"] = "/";
    upstreamBidderConfig["adserver"]["host"] = "http://invalid-url-but-its-intended.com";

    Json::Value httpAgentConfig = Json::parse(
        R"JSON(
        {
            "account": ["dummy_account"],
            "bidProbability": 1,
            "creatives": [ { "width": 300, "height": 250, "id": 1 } ],
            "externalId": 1
        }
        )JSON");

    BidStack upstreamStack;
    upstreamStack.enforceAgents = false;

    upstreamStack.runThen(
            upstreamRouterConfig, upstreamBidderConfig, USD_CPM(10), 0,
            [&](const Json::Value& json) {

        upstreamStack.postConfig("sample_http_config", httpAgentConfig);

        ML::sleep(1.0);

        ML::RNG rng;
        OpenRTB::BidRequest req;
        req.id = Id(rng.random());
        req.tmax.val = 50;
        req.at = AuctionType::SECOND_PRICE;

        req.imp.emplace_back();
        {
            auto& imp = req.imp.back();
            imp.id = Id(rng.random());
            imp.banner.reset(new OpenRTB::Banner);
            imp.banner->w.push_back(300);
            imp.banner->h.push_back(250);
        }

        req.imp.emplace_back();
        {
            auto& imp = req.imp.back();
            imp.id = Id(rng.random());
            imp.banner.reset(new OpenRTB::Banner);
            imp.banner->w.push_back(600);
            imp.banner->h.push_back(400);
        }

        StructuredJsonPrintingContext context;
        DefaultDescription<OpenRTB::BidRequest> desc;
        desc.printJson(&req, context);

        auto bids = json["workers"][0]["bids"];
        auto url = bids["url"].asString();
        auto resource = bids.get("resource", "/").asString();

        HttpRestProxy proxy(url);
        auto response = proxy.post(
                resource, context.output,
                RestParams() /* queryParams */,
                { { "x-openrtb-version", "2.2" } });

    });

}

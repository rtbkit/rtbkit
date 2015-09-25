/** rtbkit_exchange_connector_test.cc                                 -*- c++ -*-
    Mathieu Stefani, 15 May 2014
    copyright (c) 2014 datacratic.  all rights reserved.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "rtbkit/plugins/exchange/rtbkit_exchange_connector.h"
#include "rtbkit/core/router/router.h"
#include "rtbkit/plugins/bid_request/openrtb_bid_source.h"
#include "rtbkit/openrtb/openrtb_parsing.h"

using namespace RTBKIT;
using namespace Datacratic;

struct TestContext {
    std::shared_ptr<ServiceProxies> proxies;
    std::unique_ptr<Router> router;
    std::shared_ptr<RTBKitExchangeConnector> connector;
    std::unique_ptr<OpenRTBBidSource> bidSource;

    TestContext() {
        setup();
    }

    ~TestContext() {
        router->shutdown();
    }

    void setup() {
        proxies = std::make_shared<ServiceProxies>();

        router.reset(new Router(proxies, "router"));
        router->unsafeDisableMonitor();
        router->init();

        router->bindTcp();
        router->start();

        connector = std::make_shared<RTBKitExchangeConnector>("connector", proxies);
        connector->configureHttp(1, -1, "0.0.0.0");
        connector->start();
        connector->enableUntil(Date::positiveInfinity());

        router->addExchange(connector);
        router->initFilters();

        ML::sleep(1.0);

        auto jsonConf = Json::parse(
                        R"JSON(
                            {
                                "url": "",
                                "verb": "POST",
                                "resource": "/auctions"
                            }
                        )JSON");
        jsonConf["url"] = ML::format("%s:%d", "localhost", connector->port());
        OpenRTBBidSource source(jsonConf);
        bidSource.reset(new OpenRTBBidSource(jsonConf));
    }
};


BOOST_AUTO_TEST_CASE( test_extension_field )
{
    TestContext context;

    const auto& source = context.bidSource;
    auto br = source->generateRandomBidRequest();
    auto payload = source->read();

    auto extractErrorKey = [](const HttpHeader &header) {
        auto json = Json::parse(header.knownData);
        auto error = json["error"].asString();
        auto pos = error.find(':');
        if (pos == std::string::npos) {
             return std::string("");
        } 

        return error.substr(0, pos);
    };

    HttpHeader header;
    header.parse(payload);
    BOOST_CHECK_EQUAL(header.resource, "400");
    BOOST_CHECK_EQUAL(extractErrorKey(header), "MISSING_EXTENSION_FIELD");

}

BOOST_AUTO_TEST_CASE( test_augmentation_data )
{
    //ML::Watchdog watchdog(10);
    TestContext context;

    AugmentationList aug1;
    aug1[AccountKey("hello:world")] = Augmentation({"pass-aug"}, "meow");

    AugmentationList aug2;
    aug2[AccountKey("foo:bar")] = Augmentation({"pass-aug"}, "woof");

    const auto& source = context.bidSource;
    auto request = source->generateRequest();
    Json::Value extId(Json::arrayValue);
    extId.append(42);

    for (auto& imp: request.imp) {
        imp.ext["external-ids"] = extId;
    }

    typedef std::unordered_map<std::string, AugmentationList> Augmentations;

    auto &augmentationList = request.ext["rtbkit"]["augmentationList"];
    augmentationList["catty"] = aug1.toJson();
    augmentationList["doggy"] = aug2.toJson();

    Augmentations original;
    original["catty"] = aug1;
    original["doggy"] = aug2;

    // @Refactor @Copy-and-Paste This should be a function of the BidSource
    StructuredJsonPrintingContext ctxt;
    DefaultDescription<OpenRTB::BidRequest> desc;
    desc.printJson(&request, ctxt);
    std::string content = ctxt.output.toString();

    int length = content.length();

    const std::string host = ML::format("localhost:%d", context.connector->port());

    std::string message = ML::format(
        "POST /auctions HTTP/1.1\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "accept: */*\r\n"
        "connection: Keep-Alive\r\n"
        "host: %s\r\n"
        "user-agent: be2/1.0\r\n"
        "x-openrtb-version: 2.1\r\n"
        "\r\n%s",
        length, host.c_str(), content.c_str());

    auto augmentationEquals = [](const Augmentations& lhs, const Augmentations& rhs) {
        using std::begin;
        using std::endl;

        for (const auto& augList: lhs) {
            auto it = rhs.find(augList.first);
            if (it == std::end(rhs))
                return false;

            const AugmentationList& rhsAugList = it->second;
            for (const auto& lhsAug: augList.second) {
                auto it = rhsAugList.find(lhsAug.first);
                if (it == std::end(rhsAugList))
                    return false;

                const auto& aug = it->second;
                const auto& tags = aug.tags;
                if (lhsAug.second.data != aug.data ||
                    !std::equal(begin(tags), end(tags), begin(lhsAug.second.tags)))
                    return false;
            }

        }

        return true;

    };

    int done = 0;
    context.connector->onNewAuction = [&](std::shared_ptr<Auction> auction) {
        auto& augmentations = auction->augmentations;
        BOOST_CHECK_EQUAL(augmentations.empty(), false);


        BOOST_CHECK_EQUAL(augmentationEquals(original, augmentations), true);

        done = 1;
        ML::futex_wake(done);
    };

    source->write(message);
    source->read();

    while (!done) {
        ML::futex_wait(done, 0);
    }

}

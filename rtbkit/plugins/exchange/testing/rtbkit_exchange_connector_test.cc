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

using namespace RTBKIT;
using namespace Datacratic;

BOOST_AUTO_TEST_CASE( test_extension_field )
{
    auto proxies = std::make_shared<ServiceProxies>();

    Router router(proxies, "router");
    router.unsafeDisableMonitor();
    router.init();

    router.bindTcp();
    router.start();

    auto connector = std::make_shared<RTBKitExchangeConnector>("connector", proxies);
    connector->configureHttp(1, -1, "0.0.0.0");
    connector->start();
    connector->enableUntil(Date::positiveInfinity());

    router.addExchange(connector);

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

    auto br = source.generateRandomBidRequest();
    auto payload = source.read();

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

    router.shutdown();

}

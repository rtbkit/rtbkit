/** openrtb_exchange_connector_test.cc                                 -*- C++ -*-
    Eric Robert, 7 March 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <sstream>
#include <boost/test/unit_test.hpp>

#include "rtbkit/common/testing/exchange_source.h"
#include "rtbkit/plugins/exchange/openrtb_exchange_connector.h"
#include "rtbkit/core/router/router.h"


using namespace RTBKIT;


BOOST_AUTO_TEST_CASE( test_openrtb_error_codes )
{
    std::shared_ptr<ServiceProxies> proxies(new ServiceProxies());

    // We need a router for our exchange connector to work
    Router router(proxies, "router");
    router.unsafeDisableMonitor();  // Don't require a monitor service
    router.init();

    // Start the router up
    router.bindTcp();
    router.start();

    auto connector = std::make_shared<OpenRTBExchangeConnector>("connector", proxies);
    connector->configureHttp(1, -1, "0.0.0.0");
    connector->start();
    connector->enableUntil(Date::positiveInfinity());

    // Tell the router about the new exchange connector
    router.addExchange(connector);
    router.initFilters();

    ML::sleep(1.0);

    // prepare request
    NetworkAddress address(connector->port());
    BidSource source(address);

    auto validateHttpQuery = [&](std::string const & key, std::string const & text) {
        std::cerr << "http request:" << std::endl << text << std::endl;
        source.write(text);
        auto result = source.read();
        std::cerr << "http response:" << std::endl << result << std::endl;
        HttpHeader http;
        http.parse(result);
        auto json = Json::parse(http.knownData);
        auto error = json["error"].asString();
        return error.compare(0, key.length(), key) == 0;
    };

    auto validateJsonQuery = [&](std::string const & key, std::string const & text) {
        std::ostringstream stream;
        stream << "POST /auctions HTTP/1.1\r\n"
               << "Content-Length: " << text.size() << "\r\n"
               << "Content-Type: application/json\r\n"
               << "x-openrtb-version: 2.1\r\n"
               << "x-openrtb-verbose: 1\r\n"
               << "\r\n"
               << text;
        return validateHttpQuery(key, stream.str());
    };

    BOOST_CHECK(validateHttpQuery("UNKNOWN_RESOURCE", "POST /bad HTTP/1.1\r\n"
                                                      "\r\n"));
    BOOST_CHECK(validateHttpQuery("MISSING_CONTENT_TYPE_HEADER", "POST /auctions HTTP/1.1\r\n"
                                                                 "Content-Length: 0\r\n"
                                                                 "\r\n"));
    BOOST_CHECK(validateHttpQuery("UNSUPPORTED_CONTENT_TYPE", "POST /auctions HTTP/1.1\r\n"
                                                              "Content-Length: 0\r\n"
                                                              "Content-Type: text/html\r\n"
                                                              "\r\n"));
    BOOST_CHECK(validateHttpQuery("MISSING_OPENRTB_HEADER", "POST /auctions HTTP/1.1\r\n"
                                                            "Content-Length: 0\r\n"
                                                            "Content-Type: application/json\r\n"
                                                            "\r\n"));
    BOOST_CHECK(validateHttpQuery("UNSUPPORTED_OPENRTB_VERSION", "POST /auctions HTTP/1.1\r\n"
                                                                 "Content-Length: 0\r\n"
                                                                 "Content-Type: application/json\r\n"
                                                                 "x-openrtb-version: 2\r\n"
                                                                 "\r\n"));
    BOOST_CHECK(validateHttpQuery("EMPTY_BID_REQUEST", "POST /auctions HTTP/1.1\r\n"
                                                  "Content-Length: 0\r\n"
                                                  "Content-Type: application/json\r\n"
                                                  "x-openrtb-version: 2.1\r\n"
                                                  "\r\n"));


    BOOST_CHECK(validateJsonQuery("INVALID_BID_REQUEST", "asdf"));
    BOOST_CHECK(validateJsonQuery("MISSING_ID", "{}"));

    router.shutdown();
}


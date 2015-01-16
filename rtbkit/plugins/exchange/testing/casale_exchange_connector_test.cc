/* casale_exchange_connector_test.cc
   Mathieu Stefani, 08 December 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.
   
   Unit tests for the Casale Exchange Connector
*/


#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include "rtbkit/common/testing/exchange_source.h"
#include "rtbkit/plugins/exchange/casale_exchange_connector.h"
#include "rtbkit/testing/bid_stack.h"
#include "soa/service/http_header.h"

using namespace RTBKIT;

namespace {
    const std::string SampleFile = "rtbkit/plugins/exchange/testing/casale_bid_request.json";
}

struct BidRequestFileIterator {
    BidRequestFileIterator(std::string filePath)
        : currentIndex(0)
    {
        loadFile(std::move(filePath));
    }

    bool atEnd() const {
        return currentIndex == size;
    }

    BidRequestFileIterator& operator++() {
        ExcAssert(currentIndex < size);

        ++currentIndex;
        return *this;
    }

    std::string value() const {
        ExcAssert(currentIndex < size);

        return requests[currentIndex];
    }

    std::string formattedHttpRequest(std::string resourcePath = "/auctions") const {
        const std::string req = value();

        auto httpRequest = ML::format(
                "POST %s HTTP/1.1\r\n"
                "Content-Length: %zd\r\n"
                "Content-Type: application/json\r\n"
                "Connection: Keep-Alive\r\n"
                "x-openrtb-version: 2.0\r\n"
                "\r\n"
                "%s",
                resourcePath.c_str(),
                req.size(),
                req.c_str());

        return httpRequest;
    }

private:
    void loadFile(std::string filePath)
    {
        ML::filter_istream stream(filePath);
        if (!stream)
            throw ML::Exception("Could not open '%s'", filePath.c_str());

        std::string line;
        while (std::getline(stream, line)) {
            requests.push_back(line);
        }

        size = requests.size();
    }

    typedef std::vector<std::string> Requests;

    Requests::size_type currentIndex;
    Requests::size_type size;
    Requests requests;
};

BOOST_AUTO_TEST_CASE ( test_bid_request_exchange )
{
    BidStack stack;
    auto proxies = stack.proxies;

    Json::Value routerConfig;
    routerConfig[0]["exchangeType"] = "casale";

    Json::Value bidderConfig;
    bidderConfig["type"] = "agents";

    AgentConfig config;
    config.account = { "campaign", "strategy" };
    config.creatives.push_back(Creative::sampleLB);
    config.creatives.push_back(Creative::sampleWS);
    config.creatives.push_back(Creative::sampleBB);

    config.providerConfig["casale"]["seat"] = 3122;

    // Configure every creative
    for (auto& creative: config.creatives) {
        auto& creativeConfig = creative.providerConfig["casale"];
        creativeConfig["adomain"][0] = "rtbkit.org";
        creativeConfig["adm"]
            = "<script src=\"http://adserver.dsp.com/ad.js?price=${AUCTION_PRICE}\"></script>";
    }

    auto agent = std::make_shared<TestAgent>(proxies, "bobby");
    agent->config = config;
    agent->bidWithFixedAmount(USD_CPM(10));
    stack.addAgent(agent);

    int numBids { 0 };

    stack.runThen(
        routerConfig, bidderConfig, USD_CPM(10), 0,
        [&](const Json::Value& config)
    {
        const auto& bids = config["workers"][0]["bids"];
        auto url = bids["url"].asString();

        NetworkAddress address(url);
        ExchangeSource exchangeConnection(address);

        for (BidRequestFileIterator it(SampleFile); !it.atEnd(); ++it) {
            auto request = it.formattedHttpRequest("/bidder");
            std::cerr << request << std::endl;

            exchangeConnection.write(request);

            auto response = exchangeConnection.read();
            std::cerr << response << std::endl;

            HttpHeader header;
            header.parse(response);
            if (header.resource == "200") {
                ++numBids;
            }
        }
    });

    BOOST_CHECK(numBids > 0);

    BOOST_CHECK_EQUAL(agent->numBidRequests, numBids);
}



/* spotx_exchange_connector_test.cc
   Mathieu Stefani, 20 May 2015
   Copyright (c) 2015 Datacratic.  All rights reserved.
   
   Unit tests for the SpotX Exchange Connector
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include "rtbkit/common/testing/exchange_source.h"
#include "rtbkit/plugins/exchange/spotx_exchange_connector.h"
#include "rtbkit/testing/bid_stack.h"
#include "soa/service/http_header.h"

using namespace RTBKIT;

constexpr const char* SampleBR = R"JSON(
     {"id":"96525f60c48e4a4e316d4cf4f8e11d97","imp":[{"id":"1","tagid":"http://local.search.spotxchange.com/vast/2.0/94240?content_page_url=google.com","video":{"mimes":["video/x-flv","video/mp4"], "w": 300, "h": 250, "linearity":1,"minduration":1,"maxduration":60,"delivery":[2],"companionad":[{"w ":300,"h":250,"id":"1"}],"companiontype":[1,2,3],"ext":{"initiationtype":0,"spxplayersize":0},"protocols":[2,5]},"secure":0,"pmp":{"private_auction":0}}],"device":{"dnt":0,"devicetype":2,"ua":"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/37.0.2062.103Safari/537.36","ip":"165.236.183.1","geo":{"region":"US-CO"},"make":"Google","model":"Chrome -Windows","os":"Windows 7","osv":"NT 6.1","dpidsha1":"","dpidmd5":""},"at":2,"tmax":500,"site":{"domain":"google.com","page":"http://google.com","content":{"videoquality":0},"ext":{"channelid":"94240","isiframe":"U"},"publisher":{"id":"94239","domain":"google.com"}},"user":{"id":"56d1b3c5a4bbc465f4cdf51b000b905f"}})JSON";

BOOST_AUTO_TEST_CASE ( test_bid_request_exchange )
{
    BidStack stack;
    auto proxies = stack.proxies;

    Json::Value routerConfig;
    routerConfig[0]["exchangeType"] = "spotx";

    Json::Value bidderConfig;
    bidderConfig["type"] = "agents";

    AgentConfig config;
    config.bidProbability = 1;
    config.account = { "campaign", "strategy" };

    config.creatives.push_back(Creative::video(300, 250, 10, 16000, "cr1", 1));

    config.providerConfig["spotx"]["seat"] = "54321";
    config.providerConfig["spotx"]["bidid"] = "abc123";

    // Configure every creative
    for (auto& creative: config.creatives) {
        auto& creativeConfig = creative.providerConfig["spotx"];
        creativeConfig["adomain"][0] = "rtbkit.org";
        creativeConfig["adid"] = "TestAd";
        creativeConfig["adm"]
            = R"XML(
              <?xml version="1.0" encoding="UTF-8"?><VAST version="2.0">
               <Ad id="%{bidrequest.id}">
                   <InLine> <AdSystem version="1.0">PlayTime </AdSystem>
                   <AdTitle>Example Ad</AdTitle>
                   <Impression id="%{imp.id}">
                       <![CDATA[http://advertiserdomain.com/start?price=$MBR]]>
                   </Impression>
                   <Creatives>
                   <Creative id="PbZnV6mr0HEQPkc3hC3Q">
                       <Linear>
                           <Duration>00:00:15</Duration>
                           <TrackingEvents>
                               <Tracking event="firstQuartile"> <![CDATA[http://advertiserdomain.com/25]]></Tracking>
                               <Tracking event="midpoint"> <![CDATA[http://advertiserdomain.com/50]]></Tracking>
                               <Tracking event="thirdQuartile"> <![CDATA[http://advertiserdomain.com/75]]></Tracking>
                               <Tracking event="complete"> <![CDATA[http://advertiserdomain.com/100]></Tracking>
                           </TrackingEvents>
                           <VideoClicks>
                               <ClickThrough> <![CDATA[http://advertiserdomain.com/click]]></ClickThrough>
                           </VideoClicks>
                           <MediaFiles>
                               <MediaFile height="300" width="400" bitrate="1000" type="video/x- flv" delivery="progressive">
                                   <![CDATA[http://advertiserdomain.com/ad.flv]]>
                               </MediaFile>
                               <MediaFile height="300" width="400" bitrate="1000" type="video/mp4" delivery="progressive">
                                   <![CDATA[http://advertiserdomain.com/ad.mp4]]>
                               </MediaFile>
                           </MediaFiles>
                       </Linear>
                   </Creative>
               </Creatives>
         )XML";
    }

    auto agent = std::make_shared<TestAgent>(proxies, "bobby");
    agent->config = config;
    agent->bidWithFixedAmount(USD_CPM(10));
    stack.addAgent(agent);

    stack.runThen(
        routerConfig, bidderConfig, USD_CPM(10), 0,
        [&](const Json::Value& config)
    {

        const auto& bids = config["workers"][0]["bids"];
        auto url = bids["url"].asString();

        NetworkAddress address(url);
        ExchangeSource exchangeConnection(address);

        const auto httpRequest = ML::format(
            "POST /auctions HTTP/1.1\r\n"
            "Content-Length: %zd\r\n"
            "Content-Type: application/json\r\n"
            "Connection: Keep-Alive\r\n"
            "x-openrtb-version: 2.2\r\n"
            "\r\n"
            "%s",
            std::strlen(SampleBR),
            SampleBR);

        exchangeConnection.write(httpRequest);

        auto response = exchangeConnection.read();
        HttpHeader header;
        header.parse(response);

        BOOST_CHECK_EQUAL(header.resource, "200");

    });

}

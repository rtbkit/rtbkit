/* gumgum_exchange_connector_test.cc

   Exchange connector test for Gumgum.
   Based on rubicon_exchange_connector_test.cc
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include "rtbkit/common/testing/exchange_source.h"
#include "rtbkit/plugins/bid_request/openrtb_bid_request_parser.h"
#include "rtbkit/plugins/exchange/gumgum_exchange_connector.h"
#include "rtbkit/plugins/exchange/http_auction_handler.h"
#include "rtbkit/core/router/router.h"
#include "rtbkit/core/agent_configuration/agent_configuration_service.h"
#include "rtbkit/core/banker/null_banker.h"
#include "rtbkit/testing/test_agent.h"

#include "jml/arch/info.h"

#include <type_traits>


using namespace RTBKIT;


const std::string bid_sample_filename("rtbkit/plugins/exchange/testing/gumgum_bid_request.json");


std::string loadFile(const std::string & filename)
{
    ML::filter_istream stream(filename);
    
    std::string result;
    
    while (stream) {
        std::string line;
        getline(stream, line);
        result += line + "\n";
    }

    return result;
}

BOOST_AUTO_TEST_CASE( test_gumgum )
{
    std::shared_ptr<ServiceProxies> proxies(new ServiceProxies());

    // The agent config service lets the router know how our agent is configured
    AgentConfigurationService agentConfig(proxies, "config");
    agentConfig.unsafeDisableMonitor();
    agentConfig.init();
    agentConfig.bindTcp();
    agentConfig.start();

    // We need a router for our exchange connector to work
    Router router(proxies, "router");
    router.unsafeDisableMonitor();  // Don't require a monitor service
    router.init();

    // Set a null banker that blindly approves all bids so that we can
    // bid.
    router.setBanker(std::make_shared<NullBanker>(true));

    // Start the router up
    router.bindTcp();
    router.start();

    // Create our exchange connector and configure it to listen on port
    // 10002.  Note that we need to ensure that port 10002 is open on
    // our firewall.
    std::shared_ptr<GumgumExchangeConnector> connector
        (new GumgumExchangeConnector("connector", proxies));

    connector->configureHttp(1, -1, "0.0.0.0");
    connector->start();
    int port = connector->port();

    connector->enableUntil(Date::positiveInfinity());

    // Tell the router about the new exchange connector
    router.addExchange(connector);
    router.initFilters();

    // This is our bidding agent, that actually calculates the bid price
    TestAgent agent(proxies, "agent");

    std::string portName = std::to_string(port);
    std::string hostName = ML::fqdn_hostname(portName) + ":" + portName;

    agent.config.providerConfig["gumgum"]["seat"] = "123";

    // Configure the agent for bidding
    for (auto & c: agent.config.creatives) {
        c.providerConfig["gumgum"]["adomain"] = "rtbkit.org";
        c.providerConfig["gumgum"]["adm"] = "<iframe src=”http://ad”/>";
        c.providerConfig["gumgum"]["adid"] = c.name;
        c.providerConfig["gumgum"]["nurl"] = "http://dsp.com/win";
    }

    agent.onBidRequest = [&] (
            double timestamp,
            const Id & id,
            std::shared_ptr<BidRequest> br,
            Bids bids,
            double timeLeftMs,
            const Json::Value & augmentations,
            const WinCostModel & wcm)
        {
            Bid& bid = bids[0];

            bid.bid(bid.availableCreatives[0], USD_CPM(1.234));

            agent.doBid(id, bids, Json::Value(), wcm);
            ML::atomic_inc(agent.numBidRequests);

            std::cerr << "bid count=" << agent.numBidRequests << std::endl;
        };

    agent.init();
    agent.start();
    agent.configure();


    ML::sleep(1.0);

    // load bid json (this code is dropping tmax value)
    //ML::Parse_Context context(bid_sample_filename);
    //std::shared_ptr<BidRequest> request(OpenRtbBidRequestParser::parseBidRequest(context, "gumgum", "gumgum"));
    //std::cerr << request->toJson() << std::endl;
    //std::string strBidRequest = request->toJsonStr();

    // load bid json
    std::string strJson = loadFile(bid_sample_filename);
    std::cerr << strJson << std::endl;

    // prepare request
    NetworkAddress address(port);
    BidSource source(address);

    std::string httpRequest = ML::format(
            "POST /auctions HTTP/1.1\r\n"
            "Content-Length: %zd\r\n"
            "Content-Type: application/json\r\n"
            "Connection: Keep-Alive\r\n"
						"x-openrtb-version: 2.0\r\n"
            "\r\n"
            "%s",
            strJson.size(),
            strJson.c_str());

    // and send it
    source.write(httpRequest);
    std::cerr << source.read() << std::endl;

    BOOST_CHECK_EQUAL(agent.numBidRequests, 1);

    proxies->events->dump(std::cerr);

    router.shutdown();
    agentConfig.shutdown();
}

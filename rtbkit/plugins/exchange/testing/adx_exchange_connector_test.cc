/* adx_exchange_connector_test.cc

   Exchange connector test for AdX
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <cassert>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

#include "rtbkit/common/testing/exchange_source.h"
#include "rtbkit/plugins/exchange/adx_exchange_connector.h"
#include "rtbkit/plugins/exchange/http_auction_handler.h"
#include "rtbkit/core/router/router.h"
#include "rtbkit/core/agent_configuration/agent_configuration_service.h"
#include "rtbkit/core/banker/null_banker.h"
#include "rtbkit/testing/test_agent.h"

#include "jml/arch/info.h"

#include <type_traits>


using namespace RTBKIT;


const std::string bid_sample_filename("rtbkit/plugins/exchange/testing/adx-bidrequests.dat");


std::string loadFile(const std::string & filename)
{
    // the first 4 bytes of our file do contain the lenght of the following
    // serialized bid request.
    std::ifstream ifs(filename.c_str(), std::ios::in|std::ios::binary);
    assert (ifs.is_open());
    char buf[1024];
    uint32_t len =0;
    ifs.read ((char*)&len, 4);
    assert (ifs);
    assert (len<sizeof buf);
    ifs.read (buf, len);
    assert (ifs);
    return std::string (buf, len);
}

BOOST_AUTO_TEST_CASE( test_adx )
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
    std::shared_ptr<AdXExchangeConnector> connector
    (new AdXExchangeConnector("connector", proxies));

    connector->configureHttp(1, -1, "0.0.0.0");
    connector->start();
    int port = connector->port();

    connector->enableUntil(Date::positiveInfinity());

    // Tell the router about the new exchange connector
    router.addExchange(connector);
    router.initFilters();

    // This is our bidding agent, that actually calculates the bid price
    TestAgent agent(proxies, "BOB");
    agent.config.account = {"janCampaign", "janStrat"};
    agent.config.maxInFlight = 20000;
    agent.config.creatives.push_back(RTBKIT::Creative::sampleLB);
    agent.config.creatives.push_back(RTBKIT::Creative::sampleWS);
    agent.config.creatives.push_back(RTBKIT::Creative::sampleBB);
    agent.config.exchangeFilter.include.push_back("adx");
    std::string portName = std::to_string(port);
    std::string hostName = ML::fqdn_hostname(portName) + ":" + portName;


    // Configure the agent for bidding
    for (auto & c: agent.config.creatives) {
        c.exchangeFilter.include.push_back("adx");
        c.providerConfig["adx"]["externalId"] = "1234";
        c.providerConfig["adx"]["htmlTemplate"] = "<a href=\"http://usmc.com=%%WINNING_PRICE%%\"/>";
        c.providerConfig["adx"]["clickThroughUrl"] = "<a href=\"http://click.usmc.com\"/>";
        c.providerConfig["adx"]["restrictedCategories"] = "0";
        c.providerConfig["adx"]["agencyId"] = 59;
        c.providerConfig["adx"]["adGroupId"] = 33970612;
        c.providerConfig["adx"]["vendorType"] = "534 423";
        c.providerConfig["adx"]["attribute"]  = "";
        c.providerConfig["adx"]["sensitiveCategory"]  = "0";
    }

    agent.onBidRequest = [&] (
                             double timestamp,
                             const Id & id,
                             std::shared_ptr<BidRequest> br,
                             Bids bids,
                             double timeLeftMs,
                             const Json::Value & augmentations,
    const WinCostModel & wcm) {

        std::cerr << "************************ ON BID REQ\n";
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

    // load bid protocol buffer
    std::string google_bid_request = loadFile(bid_sample_filename);

    // prepare request
    NetworkAddress address(port);
    BidSource source(address);

    std::ostringstream oss ;
    oss << "POST /auctions HTTP/1.1\r\n"
        << "Content-Length: "<< google_bid_request.size() << "\r\n"
        << "Content-Type: application/octet-stream\r\n"
        << "Connection: Keep-Alive\r\n"
        << "\r\n";
    auto httpRequest = oss.str() + google_bid_request;

    // and send it
    source.write(httpRequest);
    std::cerr << source.read() << std::endl;

    BOOST_CHECK_EQUAL(agent.numBidRequests, 1);

    proxies->events->dump(std::cerr);

    router.shutdown();
    agentConfig.shutdown();
}

/* bid_stack.h
   Eric Robert, 16 May 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Tool to ease tests including the bid stack
*/

#include "rtbkit/core/router/router.h"
#include "rtbkit/core/agent_configuration/agent_configuration_service.h"
#include "rtbkit/core/banker/null_banker.h"
#include "rtbkit/common/bidder_interface.h"
#include "rtbkit/common/testing/exchange_source.h"
#include "rtbkit/testing/test_agent.h"
#include "rtbkit/testing/mock_exchange.h"
#include "jml/utils/file_functions.h"

namespace RTBKIT {

struct BidStack {
    std::shared_ptr<ServiceProxies> proxies;

    // components
    std::shared_ptr<Banker> banker;
    std::shared_ptr<TestAgent> agent;

    std::pair<std::string, std::string> forwardInfo;
    BidStack() {
        proxies.reset(new ServiceProxies());
    }

    void useForwardingUri(const std::string &host, const std::string &resource) {
        forwardInfo = { host, resource };
    }

    void run(std::string const & configuration, Amount amount = Amount(), int count = 0) {
        runThen(configuration, amount, count, [=](Json::Value const & config) {
            if(count) {
                auto proxies = std::make_shared<ServiceProxies>();
                MockExchange mockExchange(proxies);
                mockExchange.start(config);
            }
        });
    }

    template<typename T>
    void runThen(std::string const & configuration, Amount amount, int count, T const & then) {
        // The agent config service lets the router know how our agent is
        // configured
        AgentConfigurationService agentConfig(proxies, "config");
        agentConfig.unsafeDisableMonitor();
        agentConfig.init();
        agentConfig.bindTcp();
        agentConfig.start();

        // We need a router for our exchange connector to work
        Router router(proxies, "router");
        router.unsafeDisableMonitor();
        router.init();

        // Set a null banker that blindly approves all bids so that we can
        // bid.
        if(!banker) {
            banker = std::make_shared<NullBanker>(true);
        }

        router.setBanker(banker);
        if (!forwardInfo.first.empty()) {
            Json::Value json;
            json["type"] = "http";
            json["host"] = forwardInfo.first;
            json["path"] = forwardInfo.second;
            router.bidder = BidderInterface::create("bidder", proxies, json);
            router.bidder->init(&router.bridge, &router);
        }

        // Start the router up
        router.bindTcp();
        router.start();

        // Configure exchange connectors
        Json::Value exchangeConfiguration = Json::parse(configuration);
        for(auto & exchange : exchangeConfiguration) {
            router.startExchange(exchange);
        }

        std::string mock = "{\"workers\":[";

        router.forAllExchanges([&](std::shared_ptr<ExchangeConnector> const & item) {
            item->enableUntil(Date::positiveInfinity());

            auto json = item->getBidSourceConfiguration();
            std::cerr << json << std::endl;
            mock += ML::format("{\"bids\":{\"lifetime\":%d,%s,\"wins\":{\"type\":\"none\"},\"events\":{\"type\":\"none\"}},",
                               count,
                               json.substr(1));
        });

        mock.erase(mock.size() - 1);
        mock += "]}";

        // This is our bidding agent, that actually calculates the bid price
        if(!agent) {
            agent = std::make_shared<TestAgent>(proxies, "agent");
        }

        if (!forwardInfo.first.empty()) {
            agent->config.external = true;
        }

        agent->init();
        agent->bidWithFixedAmount(amount);
        agent->start();
        agent->strictMode(false);
        agent->configure();

        // Wait a little for the stack to startup...
        ML::sleep(1.0);

        then(Json::parse(mock));
    }
};

} // namespace RTBKIT

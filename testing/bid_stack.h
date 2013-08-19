/* bid_stack.h
   Eric Robert, 16 May 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Tool to ease tests including the bid stack
*/

#include "rtbkit/core/router/router.h"
#include "rtbkit/core/agent_configuration/agent_configuration_service.h"
#include "rtbkit/core/banker/null_banker.h"
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

    BidStack() {
        proxies.reset(new ServiceProxies());
    }

     void run(std::string const & configuration, Amount amount, int count) {
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
            mock += ML::format("{\"bids\":{\"lifetime\":%d,%s,\"wins\":{\"type\":\"none\"}},",
                               count,
                               json.substr(1));
        });

        mock.erase(mock.size() - 1);
        mock += "]}";

        // This is our bidding agent, that actually calculates the bid price
        if(!agent) {
            agent = std::make_shared<TestAgent>(proxies, "agent");
        }

        agent->init();
        agent->bidWithFixedAmount(amount);
        agent->start();
        agent->configure();

        // Wait a little for the stack to startup...
        ML::sleep(1.0);

        MockExchange mockExchange(proxies);
        mockExchange.start(Json::parse(mock));
    }
};

} // namespace RTBKIT

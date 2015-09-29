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
    bool enforceAgents;

    // components
    struct Services {
        std::shared_ptr<Banker> banker;
        std::vector<std::shared_ptr<TestAgent>> agents;
        std::shared_ptr<AgentConfigurationService> acs;
        std::shared_ptr<Router> router;
    } services;

    Json::Value bidderConfig;

    BidStack()
     : proxies(new ServiceProxies())
     , enforceAgents(true)
    { }

    void run(Json::Value const & routerConfig,
             Json::Value const & bidderConfig,
             Amount amount = Amount(), int count = 0) {
        runThen(routerConfig, bidderConfig, amount, count, [=](Json::Value const & config) {

            if(count) {
                auto proxies = std::make_shared<ServiceProxies>();
                MockExchange mockExchange(proxies);
                mockExchange.start(config);
            }
        });
    }

    void addAgent(const std::shared_ptr<TestAgent> &agent) {
        services.agents.push_back(agent);
    }

    void postConfig(const std::string &agent, const Json::Value &config) {
        ExcAssert(services.acs);

        services.acs->handleAgentConfig(agent, config);
    }

    template<typename T>
    void runThen(Json::Value const & routerConfig,
                 Json::Value const & bidderConfig,
                 Amount amount, int count, T const & then) {
        // The agent config service lets the router know how our agent is
        // configured
        services.acs.reset(new AgentConfigurationService(proxies, "AgentConfigurationService"));
        services.acs->unsafeDisableMonitor();
        services.acs->init();
        services.acs->bindTcp();
        services.acs->start();

        // We need a router for our exchange connector to work
        services.router.reset(new Router(proxies, "router"));
        services.router->unsafeDisableMonitor();
        services.router->initBidderInterface(bidderConfig);
        services.router->init();

        // Set a null banker that blindly approves all bids so that we can
        // bid.
        if(!services.banker) {
            services.banker.reset(new NullBanker(true));
        }

        services.router->setBanker(services.banker);

        services.router->initExchanges(routerConfig);
        services.router->initFilters();

        // Start the router up
        services.router->bindTcp();
        services.router->start();

        std::string mock = "{\"workers\":[";

        services.router->forAllExchanges([&](std::shared_ptr<ExchangeConnector> const & item) {
            item->enableUntil(Date::positiveInfinity());

            auto json = item->getBidSourceConfiguration();
            mock += ML::format("{\"bids\":{\"lifetime\":%d,%s,\"wins\":{\"type\":\"none\"},\"events\":{\"type\":\"none\"}},",
                               count,
                               json.substr(1));
        });

        mock.erase(mock.size() - 1);
        mock += "]}";

        // This is our bidding agent, that actually calculates the bid price
        if(services.agents.empty() && enforceAgents) {
            auto agent = std::make_shared<TestAgent>(proxies, "agent");
            agent->bidWithFixedAmount(amount);
            services.agents.push_back(agent);
        }

        for (auto &agent: services.agents) {
            agent->init();
            agent->start();
            agent->strictMode(false);
            agent->configure();
        }

        // Wait a little for the stack to startup...
        ML::sleep(1.0);

        then(Json::parse(mock));
    }
};

} // namespace RTBKIT

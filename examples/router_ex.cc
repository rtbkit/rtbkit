 /* router_ex.cc
   Eric Robert, 11 April 2013
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Program to run the router.
*/

#include "rtbkit/core/router/router_runner.h"
#include "rtbkit/plugins/exchange/http_exchange_connector.h"
#include "jml/arch/timers.h"
#include "jml/utils/json_parsing.h"
#include "mock_exchange_connector.h"

using namespace std;
using namespace Datacratic;
using namespace RTBKIT;

namespace {
    struct Init {
        static ExchangeConnector * createMockExchange(ServiceBase * owner, std::string const & name) {
            return new MockExchangeConnector(*owner, name);
        }

        Init() {
            ExchangeConnector::registerFactory("mock", createMockExchange);
        }
    } init;
}

struct MockRouterRunner: public RouterRunner {

    MockRouterRunner()
    {
        logUris                   = {};
        exchangeConfigurationFile = "examples/mock-config.json";
        lossSeconds               = 15;
    }
};

int main(int argc, char ** argv)
{
    MockRouterRunner runner;

    runner.doOptions(argc, argv);

    runner.init();
    runner.start();

    runner.router->forAllExchanges([](std::shared_ptr<ExchangeConnector> const & item) {
        item->enableUntil(Date::positiveInfinity());
    });

    for (;;) {
        ML::sleep(10.0);
    }
}

 /* router_ex.cc
   Eric Robert, 11 April 2013
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Program to run the router.
*/

#include "rtbkit/core/router/router_runner.h"
#include "rtbkit/plugins/exchange/exchanges.h"
#include "jml/arch/timers.h"

using namespace std;
using namespace Datacratic;
using namespace RTBKIT;

struct GenericRouterRunner: public RouterRunner {
    GenericRouterRunner()
    {
        logUris                   = {};
        exchangeConfigurationFile = "examples/router-config.json";
        lossSeconds               = 15;
    }
};

int main(int argc, char ** argv)
{
    GenericRouterRunner runner;

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

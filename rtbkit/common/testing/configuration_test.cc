/* configuration_test.cc
   Mathieu Stefani, 16 décembre 2015
   Copyright (c) 2015 Datacratic.  All rights reserved.
   
   Tests for the static configuration system
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "rtbkit/common/static_configuration.h"
#include "soa/service/zmq_endpoint.h"

#include <boost/test/unit_test.hpp>

using namespace RTBKIT;
using namespace std;

BOOST_AUTO_TEST_CASE( basic_test )
{
    auto d = Discovery::StaticDiscovery::fromFile("static.json");

    auto config = std::make_shared<Datacratic::NullConfigurationService>();

    Datacratic::ServiceProxies proxies;
    Datacratic::ZmqNamedClientBus agents(proxies.zmqContext);
    Datacratic::ZmqNamedPublisher logger(proxies.zmqContext);

    Datacratic::RestServiceEndpoint banker(proxies.zmqContext);

    d
        .configure("rtbRequestRouter", "rtb1.mtl.router")
        .bind(&agents, "router.agents")
        .bind(&logger, "logger");

    d
        .configure("rtbBanker", "rtb1.mtl.masterBanker")
        .bind(&banker, "banker.rest");
}

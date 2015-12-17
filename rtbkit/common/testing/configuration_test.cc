/* configuration_test.cc
   Mathieu Stefani, 16 décembre 2015
   Copyright (c) 2015 Datacratic.  All rights reserved.
   
   Tests for the static configuration system
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "rtbkit/common/static_configuration.h"

#include <boost/test/unit_test.hpp>

using namespace RTBKIT;
using namespace std;

BOOST_AUTO_TEST_CASE( basic_test )
{
    auto d = Discovery::StaticDiscovery::fromFile("static.json");

    auto palAgents = d.namedEndpoint("pal.agents");
    auto port = palAgents.port();
    std::cout << "serviceName = " << palAgents.serviceName() << std::endl;
    std::cout << "Port = " << static_cast<uint16_t>(port) << std::endl;

    auto logger = d.namedEndpoint("logger");
    std::cout << "serviceName = " << logger.serviceName() << std::endl;
}

/* multiple_service_test.cc
   Jeremy Barnes, 10 December 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/make_shared.hpp>
#include "soa/service/named_endpoint.h"
#include "soa/service/message_loop.h"
#include "soa/service/zmq_endpoint.h"
#include "soa/service/testing/zookeeper_temporary_server.h"
#include "soa/service/zookeeper_configuration_service.h"
#include "jml/utils/guard.h"
#include "jml/arch/exception_handler.h"
#include "jml/utils/testing/watchdog.h"
#include "jml/utils/testing/fd_exhauster.h"
#include "jml/utils/vector_utils.h"
#include "jml/arch/timers.h"
#include <thread>
#include "soa/service/zmq_utils.h"
#include "soa/service/zookeeper.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "service_discovery_scenario.h"

using namespace std;
using namespace ML;
using namespace Datacratic;

BOOST_AUTO_TEST_CASE( test_service_zk_disconnect )
{
    ServiceDiscoveryScenario scenario("test_service_zk_disconnect");
    ServiceDiscoveryScenarioTest test(scenario);

    int port = scenario.startTemporaryServer();
    ML::sleep(2);

    cerr << "Starting multiple service zk disconnect " << endl;
    scenario.createProxies(formatHost("localhost", port));

    scenario.createConnectionAndStart("client1");

    test.assertConnectionCount("client1", 0);

    std::cerr << "Connecting all service providers" << std::endl;
    scenario.connectServiceProviders("client1", "echo", "echo");

    test.assertTriggeredWatches("client1", 0);
    test.assertConnectionCount("client1", 0);

    std::cerr << "Creating service" << std::endl;
    scenario.createServiceAndStart("echo0");

    ML::sleep(10);
    // Service must be registered in ZooKeeper, a watch must have been triggered
    test.assertTriggeredWatches("client1", 1);

    std::cerr << "About to suspend zookeeper..." << std::endl;
    scenario.suspendServer();
    std::cerr << "zookeeper suspended " << std::endl;
    ML::sleep(10);

    // When suspending the server, a watch with SESSION_EXPIRED is triggered
    test.assertTriggeredWatches("client1", 2);
    std::cerr <<"resuming zookeeper " << std::endl;
    scenario.resumeServer();
    ML::sleep(10);

    // After resuming the server, the watch must have been reinstalled. We
    // suspend the server again to check if the watch is triggered
    scenario.suspendServer();
    std::cerr << "zookeeper suspended again" << std::endl;
    ML::sleep(10);
    test.assertTriggeredWatches("client1", 3);
    std::cerr <<"resuming zookeeper again " << std::endl;
    scenario.resumeServer();

    cerr << "going to sleep for 10 seconds.." << endl;
    ML::sleep(10);
    cerr << "shutting down" << endl;

    scenario.reset();
}

BOOST_AUTO_TEST_CASE( test_early_connection )
{
    /** Test that we can do a "connect", then start the service, and
        have the connection come up once the service appears.
    */
    cerr << "Testing early connection..." << endl;

    ServiceDiscoveryScenario scenario("test_early_connection");
    ServiceDiscoveryScenarioTest test(scenario);

    int port = scenario.startTemporaryServer();
    scenario.createProxies(formatHost("localhost", port));

    scenario.createConnectionAndStart("client1");

    scenario.connectServiceProviders("client1", "echo", "echo");

    test.assertConnectionCount("client1", 0);

    scenario.createServiceAndStart("echo0");

    ML::sleep(3);

    cerr << "Checking that we are connected " << endl;
    test.assertConnectionCount("client1", 1);

    std::cerr << "done." << std::endl;

    scenario.reset();
}

BOOST_AUTO_TEST_CASE( test_multiple_services )
{
    ServiceDiscoveryScenario scenario("test_multiple_services");
    ServiceDiscoveryScenarioTest test(scenario);

    int port = scenario.startTemporaryServer();
    scenario.createProxies(formatHost("localhost", port));

    cerr << "Starting multiple services test " << endl;

    scenario.createConnectionAndStart("client1");

    test.assertConnectionCount("client1", 0);

    scenario.connectServiceProviders("client1", "echo", "echo");

    test.assertConnectionCount("client1", 0);

    scenario.createServiceAndStart("echo0");
    scenario.createServiceAndStart("echo1");

    ML::sleep(3);

    test.assertConnectionCount("client1", 2);

    scenario.reset();
}


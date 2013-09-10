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

    auto connection = scenario.createConnectionAndStart("client1");

    connection->connectHandler = [&] (const std::string & svc)
        {
            cerr << "connected to " << svc << endl;
        };

    connection->disconnectHandler = [&] (const std::string  & svc)
        {
            cerr << "disconnected from " << svc << endl;
        };

    test.assertConnectionCount("client1", 0);

    scenario.connectServiceProviders("client1", "echo", "echo");

    test.assertConnectionCount("client1", 0);

    scenario.createServiceAndStart("echo0");

    ML::sleep(5);

    std::cerr << "About to suspend zookeeper..." ;
//    test.assertTriggeredWatches("client1", 1);
    std::cerr << "Watches = " << connection->triggeredWatches << std::endl;
    scenario.suspendServer();
    std::cerr << "zookeeper suspended " << std::endl;
    ML::sleep(10);
    std::cerr <<"resuming zookeeper " << std::endl;
    scenario.resumeServer();
    ML::sleep(10);
    std::cerr << "Watches = " << connection->triggeredWatches << std::endl;
//    test.assertTriggeredWatches("client1", 4);
    scenario.suspendServer();
    std::cerr << "zookeeper suspended again" << std::endl;
    ML::sleep(10);
    std::cerr <<"resuming zookeeper again " << std::endl;
    scenario.resumeServer();
    cerr << "going to sleep for 30 seconds.." << endl;
    ML::sleep(10);
    cerr << "shutting down" << endl;
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
    auto proxies = scenario.createProxies(formatHost("localhost", port));

    auto connection = scenario.createConnectionAndStart("client1");
    connection->connectHandler = [&] (const std::string & svc) {
        cerr << "connected to " << svc << endl;
    };

    connection->disconnectHandler = [&] (const std::string  & svc) {
        cerr << "disconnected from " << svc << endl;
    };

    scenario.connectServiceProviders("client1", "echo", "echo");

    test.assertConnectionCount("client1", 0);

    proxies->config->removePath("");

    scenario.createServiceAndStart("echo");
    //proxies->config->dump(cerr);

    scenario.waitForClientConnected("client1");

    cerr << "Checking that we are connected " << endl;
    test.assertConnectionCount("client1", 1);

    std::cerr << "done." << std::endl;
}

BOOST_AUTO_TEST_CASE( test_multiple_services )
{
    ServiceDiscoveryScenario scenario("test_multiple_services");
    ServiceDiscoveryScenarioTest test(scenario);

    int port = scenario.startTemporaryServer();
    scenario.createProxies(formatHost("localhost", port));

    cerr << "Starting multiple services test " << endl;

    auto connection = scenario.createConnectionAndStart("client1");
    connection->connectHandler = [&] (const std::string & svc)
        {
            cerr << "connected to " << svc << endl;
        };

    connection->disconnectHandler = [&] (const std::string  & svc)
        {
            cerr << "disconnected from " << svc << endl;
        };

    test.assertConnectionCount("client1", 0);

    scenario.connectServiceProviders("client1", "echo", "echo");

    test.assertConnectionCount("client1", 0);

    int instance { 0 };
    auto startService = [&] ()
    {
        scenario.createServiceAndStart("echo" + to_string(instance++));
    };

    startService();

    ML::sleep(0.1);

    test.assertConnectionCount("client1", 1);

}


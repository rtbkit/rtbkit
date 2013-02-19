/* monitor_behaviour_test.cc
   Wolfgang Sourdeau, Janyary 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
   
   Functional tests for the Monitor classes
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <memory>
#include "boost/shared_ptr.hpp"
#include "boost/test/unit_test.hpp"

#include "soa/jsoncpp/value.h"

#include "jml/arch/timers.h"
#include "jml/utils/testing/watchdog.h"
#include "jml/utils/pair_utils.h" // ostream << pair

#include "soa/service/rest_proxy.h"
#include "soa/service/service_base.h"

#include "rtbkit/core/monitor/monitor.h"
#include "rtbkit/core/monitor/monitor_provider.h"
#include "rtbkit/core/monitor/monitor_provider_proxy.h"
#include "rtbkit/core/monitor/monitor_proxy.h"

#include "mock_monitor_provider.h"

using namespace std;
using namespace Datacratic;
using namespace RTBKIT;

/* test the response returned by the monitor provider endpoint */
BOOST_AUTO_TEST_CASE( test_monitor_provider )
{
    ML::Watchdog watchdog(20.0);
    
    MockMonitorProvider provider;

    auto proxies = std::make_shared<ServiceProxies>();
    ServiceBase parentService("parentservice", proxies);
    MonitorProviderEndpoint endpoint(parentService, provider);
    endpoint.init();
    endpoint.start();

    auto addr = endpoint.bindTcp();
    // cerr << "monitor is listening on " << addr.first << ","
    //      << addr.second << endl;

    int rc;
    Json::Value bodyJson;

    auto doRequestSync = [&] () {
        int done(false);
        auto onDone = [&] (std::exception_ptr excPtr,
                           int newRc, string newBody) {
            rc = newRc;
            // cerr << "body: " << newBody << endl;
            bodyJson = Json::parse(newBody);
            done = true;
            ML::futex_wake(done);
        };

        RestProxy proxy(proxies->zmqContext);
        auto config = proxies->config;
        // config->dump(cerr);
        proxy.init(config, "parentservice/monitor-provider");
        proxy.start();
        proxy.push(onDone, "GET", "/status");
        while (!done) {
            cerr << "waiting for response" << endl;
            ML::futex_wait(done, false);
        }
    };

    /* initial value is "null" */
    doRequestSync();
    BOOST_CHECK_EQUAL(rc, 200);
    Json::Value testJson = Json::Value();
    BOOST_CHECK_EQUAL(bodyJson, testJson);

    /* set status to true, "status" value becomes "ok" */
    provider.status = true;
    ML::sleep(2);
    doRequestSync();
    BOOST_CHECK_EQUAL(rc, 200);
    testJson = Json::parse("{'status': 'ok'}");
    BOOST_CHECK_EQUAL(bodyJson, testJson);

    /* set status to false, "status" value becomes "failure" */
    provider.status = false;
    ML::sleep(2);
    doRequestSync();
    BOOST_CHECK_EQUAL(rc, 200);
    testJson = Json::parse("{'status': 'failure'}");
    BOOST_CHECK_EQUAL(bodyJson, testJson);
}

/* test the Monitor/MonitorProviderProxy pair using 2 mock monitor services */
BOOST_AUTO_TEST_CASE( test_monitor )
{
    ML::Watchdog watchdog(30.0);

    auto proxies = std::make_shared<ServiceProxies>();

    Monitor monitor(proxies);

    MockMonitorProvider provider1;
    ServiceBase parentService1("parentservice1", proxies);
    MonitorProviderEndpoint endpoint1(parentService1, provider1);
    endpoint1.init();
    endpoint1.start();
    endpoint1.bindTcp();

    MockMonitorProvider provider2;
    ServiceBase parentService2("parentservice2", proxies);
    MonitorProviderEndpoint endpoint2(parentService2, provider2);
    endpoint2.init();
    endpoint2.start();
    endpoint2.bindTcp();

    MonitorProviderProxy proxy(proxies->zmqContext, monitor);
    proxy.init(proxies->config,
                {"parentservice1", "parentservice2"});
    proxy.start();

    /* provider1 status is false and provider2's is false
       => proxy status is false */
    cerr << ("test: "
             "provider1 status is false and provider2's is false\n"
             "=> proxy status is false\n");
    monitor.lastStatus = true;
    Date initialCheck = monitor.lastCheck;
    while (monitor.lastCheck == initialCheck) {
        ML::sleep(3);
    }
    BOOST_CHECK_EQUAL(monitor.getMonitorStatus(), false);

    /* provider1 status is true but provider2's is false
       => proxy status is false */
    cerr << ("test: "
             "provider1 status is true but provider2's is false\n"
             "=> proxy status is false\n");
    provider1.status = true;
    monitor.lastStatus = true;
    initialCheck = monitor.lastCheck;
    while (monitor.lastCheck == initialCheck) {
        ML::sleep(3);
    }
    BOOST_CHECK_EQUAL(monitor.getMonitorStatus(), false);

    /* provider1 status is true and provider2's is true
       => proxy status is true */
    cerr << ("test: "
             "provider1 status is true and provider2's is true\n"
             "=> proxy status is true\n");
    provider2.status = true;
    monitor.lastStatus = false;
    initialCheck = monitor.lastCheck;
    while (monitor.lastCheck == initialCheck) {
        ML::sleep(3);
    }
    BOOST_CHECK_EQUAL(monitor.getMonitorStatus(), true);

    /* all providers answer with a delay of one second
       => proxy status is true */
    cerr << ("test: "
             "all providers answer with a delay of one second\n"
             "=> proxy status is true\n");
    provider1.delay = 1;
    provider2.delay = 1;
    monitor.lastStatus = false;
    initialCheck = monitor.lastCheck;
    while (monitor.lastCheck == initialCheck) {
        ML::sleep(3);
    }
    BOOST_CHECK_EQUAL(monitor.getMonitorStatus(), true);

    /* one providers answers with a delay of three seconds
       => proxy status is false */
    cerr << ("test: "
             "one provider answers with a delay of three seconds\n"
             "=> proxy status is false\n");
    provider2.delay = 3;
    monitor.lastStatus = true;
    initialCheck = monitor.lastCheck;
    while (monitor.lastCheck == initialCheck) {
        ML::sleep(3);
    }
    BOOST_CHECK_EQUAL(monitor.getMonitorStatus(), false);
}

/* test the ability of a MonitorProxy to update itself via http, using a
 * Monitor endpoint and zookeeper */
BOOST_AUTO_TEST_CASE( test_monitor_proxy )
{
    ML::Watchdog watchdog(30.0);

    auto proxies = std::make_shared<ServiceProxies>();
    string zookeeperAddress = "localhost:2181";
    string zookeeperPath = "CWD";
    proxies->useZookeeper(zookeeperAddress, zookeeperPath);

    Monitor monitor(proxies);
    monitor.init();
    monitor.bindTcp();
    monitor.start();
    
    MonitorProxy proxy(proxies->zmqContext);
    proxy.init(proxies->config);
    proxy.start();

    cerr << "test: expect computed status to be TRUE"
         << " after quering the Monitor" << endl;
    proxy.lastStatus = false;
    monitor.lastCheck = Date::now();
    monitor.lastStatus = true;
    Date initialCheck = proxy.lastCheck;
    while (proxy.lastCheck == initialCheck) {
        /* make sure the monitor does not return false */
        monitor.lastCheck = Date::now();
        ML::sleep(1);
    }
    BOOST_CHECK_EQUAL(proxy.getStatus(), true);

    cerr << "test: expect computed status to be FALSE"
         << " after quering the Monitor" << endl;
    proxy.lastStatus = true;
    monitor.lastStatus = false;
    initialCheck = proxy.lastCheck;
    while (proxy.lastCheck == initialCheck) {
        /* make sure the monitor does not return false */
        monitor.lastCheck = Date::now();
        ML::sleep(1);
    }
    BOOST_CHECK_EQUAL(proxy.getStatus(), false);
}

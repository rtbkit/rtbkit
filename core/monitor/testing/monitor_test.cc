/* monitor_test.cc
   Wolfgang Sourdeau, January 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
   
   Unit tests for the Monitor class
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/shared_ptr.hpp>
#include <boost/test/unit_test.hpp>

#include "rtbkit/core/monitor/monitor.h"


using namespace std;
using namespace Datacratic;
using namespace RTBKIT;


BOOST_AUTO_TEST_CASE( test_monitor_restGetMonitorStatus )
{
    auto proxies = std::make_shared<ServiceProxies>();
    Monitor monitor(proxies);

    /* setup */
    monitor.lastCheck = Date::now();

    cerr << ("lastStatus = false, lastCheck = now, checkTimeout_ = 2\n"
             "=> status: failure\n");
    monitor.lastStatus = false;
    string body = monitor.restGetMonitorStatus();
    BOOST_CHECK_EQUAL(body, "{\"status\":\"failure\"}");

    cerr << ("lastStatus = true, lastCheck = now, checkTimeout_ = 2\n"
             "=> status: ok\n");
    monitor.lastStatus = true;
    body = monitor.restGetMonitorStatus();
    BOOST_CHECK_EQUAL(body, "{\"status\":\"ok\"}");

    cerr << ("lastStatus = true, lastCheck = now - 5, checkTimeout_ = 2\n"
             "=> status: failure\n");
    monitor.lastCheck = monitor.lastCheck.plusSeconds(-5);
    body = monitor.restGetMonitorStatus();
    BOOST_CHECK_EQUAL(body, "{\"status\":\"failure\"}");
}

BOOST_AUTO_TEST_CASE( test_monitor_getMonitorStatus )
{
    auto proxies = std::make_shared<ServiceProxies>();
    Monitor monitor(proxies);

    /* setup */
    monitor.lastCheck = Date::now();

    cerr << ("lastStatus = false, lastCheck = now, checkTimeout_ = 2\n"
             "=> status: false\n");
    monitor.lastStatus = false;
    bool status = monitor.getMonitorStatus();
    BOOST_CHECK_EQUAL(status, false);

    cerr << ("lastStatus = true, lastCheck = now, checkTimeout_ = 2\n"
             "=> status: true\n");
    monitor.lastStatus = true;
    status = monitor.getMonitorStatus();
    BOOST_CHECK_EQUAL(status, true);

    cerr << ("lastStatus = true, lastCheck = now - 5, checkTimeout_ = 2\n"
             "=> status: false\n");
    monitor.lastCheck = monitor.lastCheck.plusSeconds(-5);
    status = monitor.getMonitorStatus();
    BOOST_CHECK_EQUAL(status, false);
}

BOOST_AUTO_TEST_CASE( test_monitor_onProviderStatusLoaded )
{
    auto proxies = std::make_shared<ServiceProxies>();
    Monitor monitor(proxies);

    Date oldLastCheck = Date::now().plusSeconds(-10);

    MonitorProviderResponses responses;

    cerr << "test: empty responses\n=> lastCheck = now, lastStatus = true\n";
    monitor.lastCheck = oldLastCheck;
    monitor.lastStatus = false;
    responses.clear();
    monitor.onProvidersStatusLoaded(responses);
    BOOST_CHECK(monitor.lastCheck != oldLastCheck);
    BOOST_CHECK_EQUAL(monitor.lastStatus, true);

    cerr << ("test: 1 'false' response\n"
             "=> lastCheck = now, lastStatus = false\n");
    monitor.lastCheck = oldLastCheck;
    monitor.lastStatus = true;
    responses.clear();
    responses.emplace_back(MonitorProviderResponse("service1",
                                                   100, "nojson"));
    monitor.onProvidersStatusLoaded(responses);
    BOOST_CHECK(monitor.lastCheck != oldLastCheck);
    BOOST_CHECK_EQUAL(monitor.lastStatus, false);

    cerr << ("test: 1 'true' response + 1 'false' response\n"
             "=> lastCheck = now, lastStatus = false\n");
    monitor.lastCheck = oldLastCheck;
    monitor.lastStatus = true;
    responses.clear();
    responses.emplace_back(MonitorProviderResponse("service1",
                                                   100, "nojson"));
    responses.emplace_back(MonitorProviderResponse("service2",
                                                   200, "{'status':'ok'}"));
    monitor.onProvidersStatusLoaded(responses);
    BOOST_CHECK(monitor.lastCheck != oldLastCheck);
    BOOST_CHECK_EQUAL(monitor.lastStatus, false);

    cerr << ("test: 2 'true' responses\n"
             "=> lastCheck = now, lastStatus = true\n");
    monitor.lastCheck = oldLastCheck;
    monitor.lastStatus = false;
    responses.clear();
    responses.emplace_back(MonitorProviderResponse("service1",
                                                   200, "{'status':'ok'}"));
    responses.emplace_back(MonitorProviderResponse("service2",
                                                   200, "{'status':'ok'}"));
    monitor.onProvidersStatusLoaded(responses);
    BOOST_CHECK(monitor.lastCheck != oldLastCheck);
    BOOST_CHECK_EQUAL(monitor.lastStatus, true);
}

BOOST_AUTO_TEST_CASE( test_monitor_checkProviderResponse )
{
    auto proxies = std::make_shared<ServiceProxies>();
    Monitor monitor(proxies);

    MonitorProviderResponse response("service1", 200, "this is no json");
    cerr << "test: code 200, invalid json\n=> status = false\n";
    bool status = monitor.checkProviderResponse(response);
    BOOST_CHECK_EQUAL(status, false);

    response.body = "null";
    cerr << "test: code 200, valid json, invalid message\n=> status = false\n";
    status = monitor.checkProviderResponse(response);
    BOOST_CHECK_EQUAL(status, false);

    response.code = 100;
    response.body = "{ 'status': 'ok' }";
    cerr << "test: code 100, valid ok json message\n=> status = false\n";
    status = monitor.checkProviderResponse(response);
    BOOST_CHECK_EQUAL(status, false);

    cerr << "test: code 200, valid failure json message\n=> status = false\n";
    response.code = 200;
    response.body = "{ 'status': 'failure' }";
    status = monitor.checkProviderResponse(response);
    BOOST_CHECK_EQUAL(status, false);

    cerr << "test: code 200, valid ok json message\n=> status = true\n";
    response.body = "{ 'status': 'ok' }";
    status = monitor.checkProviderResponse(response);
    BOOST_CHECK_EQUAL(status, true);
}

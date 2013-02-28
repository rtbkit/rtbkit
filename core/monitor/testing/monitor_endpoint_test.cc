/* monitor_test.cc
   Wolfgang Sourdeau, January 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
   
   Unit tests for the MonitorEndpoint class
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/shared_ptr.hpp>
#include <boost/test/unit_test.hpp>

#include "rtbkit/core/monitor/monitor_endpoint.h"


using namespace std;
using namespace Datacratic;
using namespace RTBKIT;


BOOST_AUTO_TEST_CASE( test_monitor_getMonitorStatus )
{
    auto proxies = std::make_shared<ServiceProxies>();
    MonitorEndpoint endpoint(proxies);
    endpoint.init({"service1", "service2"});

    Date now = Date::now();
    Date oldLastCheck = now.plusSeconds(-10);

    endpoint.providersStatus_["service1"].lastCheck = now;
    endpoint.providersStatus_["service1"].lastStatus = true;
    endpoint.providersStatus_["service2"].lastCheck = now;
    endpoint.providersStatus_["service2"].lastStatus = false;
    BOOST_CHECK_EQUAL(endpoint.getMonitorStatus(), false);

    endpoint.providersStatus_["service1"].lastCheck = now;
    endpoint.providersStatus_["service1"].lastStatus = false;
    endpoint.providersStatus_["service2"].lastCheck = now;
    endpoint.providersStatus_["service2"].lastStatus = true;
    BOOST_CHECK_EQUAL(endpoint.getMonitorStatus(), false);

    endpoint.providersStatus_["service1"].lastCheck = oldLastCheck;
    endpoint.providersStatus_["service1"].lastStatus = true;
    endpoint.providersStatus_["service2"].lastCheck = now;
    endpoint.providersStatus_["service2"].lastStatus = true;
    BOOST_CHECK_EQUAL(endpoint.getMonitorStatus(), false);

    endpoint.providersStatus_["service1"].lastCheck = now;
    endpoint.providersStatus_["service1"].lastStatus = true;
    endpoint.providersStatus_["service2"].lastCheck = oldLastCheck;
    endpoint.providersStatus_["service2"].lastStatus = true;
    BOOST_CHECK_EQUAL(endpoint.getMonitorStatus(), false);

    endpoint.providersStatus_["service1"].lastCheck = now;
    endpoint.providersStatus_["service1"].lastStatus = true;
    endpoint.providersStatus_["service2"].lastCheck = now;
    endpoint.providersStatus_["service2"].lastStatus = true;
    BOOST_CHECK_EQUAL(endpoint.getMonitorStatus(), true);
}

BOOST_AUTO_TEST_CASE( test_monitor_postServiceIndicators )
{
    auto proxies = std::make_shared<ServiceProxies>();
    MonitorEndpoint endpoint(proxies);
    endpoint.init({"service1"});

    Date oldDate = Date::now().plusSeconds(-3600);

    endpoint.providersStatus_["service1"].lastCheck = oldDate;
    endpoint.providersStatus_["service1"].lastStatus = false;

    string statusStr = "this is no json";
    // cerr << "test: invalid json\n=> rc = false\n";
    bool rc = endpoint.postServiceIndicators("service1", statusStr);
    BOOST_CHECK_EQUAL(rc, false);
    BOOST_CHECK_EQUAL(endpoint.providersStatus_["service1"].lastCheck, oldDate);
    BOOST_CHECK_EQUAL(endpoint.providersStatus_["service1"].lastStatus, false);

    statusStr = "null";
    // cerr << "test: valid json, invalid message\n=> rc = false\n";
    rc = endpoint.postServiceIndicators("service1", statusStr);
    BOOST_CHECK_EQUAL(rc, false);
    BOOST_CHECK_EQUAL(endpoint.providersStatus_["service1"].lastCheck, oldDate);
    BOOST_CHECK_EQUAL(endpoint.providersStatus_["service1"].lastStatus, false);

    statusStr = "{ 'status': 'ok' }";
    // cerr << "test: valid ok json message\n=> rc = true\n";
    rc = endpoint.postServiceIndicators("service1", statusStr);
    BOOST_CHECK_EQUAL(rc, true);
    BOOST_CHECK(endpoint.providersStatus_["service1"].lastCheck != oldDate);
    BOOST_CHECK_EQUAL(endpoint.providersStatus_["service1"].lastStatus, true);

    // cerr << "test: valid failure json message\n=> rc = false\n";
    endpoint.providersStatus_["service1"].lastCheck = oldDate;
    statusStr = "{ 'status': 'failure' }";
    rc = endpoint.postServiceIndicators("service1", statusStr);
    BOOST_CHECK_EQUAL(rc, true);
    BOOST_CHECK(endpoint.providersStatus_["service1"].lastCheck != oldDate);
    BOOST_CHECK_EQUAL(endpoint.providersStatus_["service1"].lastStatus, false);

    // cerr << "test: valid ok json message\n=> rc = true\n";
    endpoint.providersStatus_["service1"].lastCheck = oldDate;
    statusStr = "{ 'status': 'ok' }";
    rc = endpoint.postServiceIndicators("service1", statusStr);
    BOOST_CHECK_EQUAL(rc, true);
    BOOST_CHECK(endpoint.providersStatus_["service1"].lastCheck != oldDate);
    BOOST_CHECK_EQUAL(endpoint.providersStatus_["service1"].lastStatus, true);
}

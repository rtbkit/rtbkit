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



MonitorEndpoint::MonitorProviderStatus
makeStatus(bool status, Date date)
{
    MonitorEndpoint::MonitorProviderStatus s;
    s.lastCheck = date;
    s.lastStatus = status;
    return s;
}

BOOST_AUTO_TEST_CASE( test_monitor_getMonitorStatus )
{
    auto proxies = std::make_shared<ServiceProxies>();
    MonitorEndpoint endpoint(proxies);
    endpoint.init({"c1", "c2"});

    Date now = Date::now();
    Date oldLastCheck = now.plusSeconds(-10);

    BOOST_CHECK_EQUAL(endpoint.getMonitorStatus(), false);

    endpoint.providersStatus_["c1"]["s1"] = makeStatus(true, now);
    BOOST_CHECK_EQUAL(endpoint.getMonitorStatus(), false);

    endpoint.providersStatus_["c1"]["s1"] = makeStatus(true, now);
    endpoint.providersStatus_["c1"]["s2"] = makeStatus(true, now);
    endpoint.providersStatus_["c2"]["s1"] = makeStatus(true, now);
    BOOST_CHECK_EQUAL(endpoint.getMonitorStatus(), true);

    endpoint.providersStatus_["c1"]["s1"] = makeStatus(false, now);
    endpoint.providersStatus_["c1"]["s2"] = makeStatus(true, now);
    endpoint.providersStatus_["c2"]["s1"] = makeStatus(true, now);
    BOOST_CHECK_EQUAL(endpoint.getMonitorStatus(), true);

    endpoint.providersStatus_["c1"]["s1"] = makeStatus(false, now);
    endpoint.providersStatus_["c1"]["s2"] = makeStatus(false, now);
    endpoint.providersStatus_["c2"]["s1"] = makeStatus(true, now);
    BOOST_CHECK_EQUAL(endpoint.getMonitorStatus(), false);

    endpoint.providersStatus_["c1"]["s1"] = makeStatus(true, now);
    endpoint.providersStatus_["c1"]["s2"] = makeStatus(true, now);
    endpoint.providersStatus_["c2"]["s1"] = makeStatus(false, now);
    BOOST_CHECK_EQUAL(endpoint.getMonitorStatus(), false);

    endpoint.providersStatus_["c1"]["s1"] = makeStatus(true, oldLastCheck);
    endpoint.providersStatus_["c1"]["s2"] = makeStatus(true, now);
    endpoint.providersStatus_["c2"]["s1"] = makeStatus(true, now);
    BOOST_CHECK_EQUAL(endpoint.getMonitorStatus(), true);


    endpoint.providersStatus_["c1"]["s1"] = makeStatus(false, now);
    endpoint.providersStatus_["c1"]["s2"] = makeStatus(true, oldLastCheck);
    endpoint.providersStatus_["c2"]["s1"] = makeStatus(true, now);
    BOOST_CHECK_EQUAL(endpoint.getMonitorStatus(), false);


    endpoint.providersStatus_["c1"]["s1"] = makeStatus(true, now);
    endpoint.providersStatus_["c1"]["s2"] = makeStatus(true, now);
    endpoint.providersStatus_["c2"]["s1"] = makeStatus(true, oldLastCheck);
    BOOST_CHECK_EQUAL(endpoint.getMonitorStatus(), false);
}

BOOST_AUTO_TEST_CASE( test_monitor_postServiceIndicators )
{
    auto proxies = std::make_shared<ServiceProxies>();
    MonitorEndpoint endpoint(proxies);
    endpoint.init({"c1"});

    Date oldDate = Date::now().plusSeconds(-3600);

    endpoint.providersStatus_["c1"]["s1"] = makeStatus(false, oldDate);

    string statusStr = "this is no json";
    // cerr << "test: invalid json\n=> rc = false\n";
    bool rc = endpoint.postServiceIndicators("c1", statusStr);
    BOOST_CHECK_EQUAL(rc, false);
    BOOST_CHECK_EQUAL(endpoint.providersStatus_["c1"]["s1"].lastCheck, oldDate);
    BOOST_CHECK_EQUAL(endpoint.providersStatus_["c1"]["s1"].lastStatus, false);

    statusStr = "null";
    // cerr << "test: valid json, invalid message\n=> rc = false\n";
    rc = endpoint.postServiceIndicators("c1", statusStr);
    BOOST_CHECK_EQUAL(rc, false);
    BOOST_CHECK_EQUAL(endpoint.providersStatus_["c1"]["s1"].lastCheck, oldDate);
    BOOST_CHECK_EQUAL(endpoint.providersStatus_["c1"]["s1"].lastStatus, false);

    statusStr = "{ 'status': true }";
    // cerr << "test: valid json, invalid message\n=> rc = false\n";
    rc = endpoint.postServiceIndicators("c1", statusStr);
    BOOST_CHECK_EQUAL(rc, false);
    BOOST_CHECK_EQUAL(endpoint.providersStatus_["c1"]["s1"].lastCheck, oldDate);
    BOOST_CHECK_EQUAL(endpoint.providersStatus_["c1"]["s1"].lastStatus, false);

    statusStr = "{ 'status': true, 'serviceName': 's1' }";
    // cerr << "test: valid ok json message\n=> rc = true\n";
    rc = endpoint.postServiceIndicators("c1", statusStr);
    BOOST_CHECK_EQUAL(rc, true);
    BOOST_CHECK(endpoint.providersStatus_["c1"]["s1"].lastCheck != oldDate);
    BOOST_CHECK_EQUAL(endpoint.providersStatus_["c1"]["s1"].lastStatus, true);

    // cerr << "test: valid failure json message\n=> rc = false\n";
    endpoint.providersStatus_["c1"]["s1"].lastCheck = oldDate;
    statusStr = "{ 'status': false, 'serviceName': 's1', 'message': 'err' }";
    rc = endpoint.postServiceIndicators("c1", statusStr);
    BOOST_CHECK_EQUAL(rc, true);
    BOOST_CHECK(endpoint.providersStatus_["c1"]["s1"].lastCheck != oldDate);
    BOOST_CHECK_EQUAL(endpoint.providersStatus_["c1"]["s1"].lastStatus, false);
    BOOST_CHECK_EQUAL(endpoint.providersStatus_["c1"]["s1"].lastMessage, "err");

    // cerr << "test: valid ok json message\n=> rc = true\n";
    endpoint.providersStatus_["c1"]["s1"].lastCheck = oldDate;
    statusStr = "{ 'status': true, 'serviceName': 's1' }";
    rc = endpoint.postServiceIndicators("c1", statusStr);
    BOOST_CHECK_EQUAL(rc, true);
    BOOST_CHECK(endpoint.providersStatus_["c1"]["s1"].lastCheck != oldDate);
    BOOST_CHECK_EQUAL(endpoint.providersStatus_["c1"]["s1"].lastStatus, true);
    BOOST_CHECK(endpoint.providersStatus_["c1"]["s1"].lastMessage.empty());
}

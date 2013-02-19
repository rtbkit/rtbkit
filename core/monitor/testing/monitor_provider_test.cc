/* monitor_provider_test.cc
   Wolfgang Sourdeau, January 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
   
   Unit tests for the MonitorProvider class
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <memory>
#include <boost/shared_ptr.hpp>
#include <boost/test/unit_test.hpp>

#include "soa/jsoncpp/value.h"

#include <jml/arch/timers.h>

#include "soa/service/rest_proxy.h"
#include "soa/service/service_base.h"

#include "rtbkit/core/monitor/monitor_provider.h"

#include "mock_monitor_provider.h"

using namespace std;
using namespace Datacratic;
using namespace RTBKIT;

BOOST_AUTO_TEST_CASE( test_monitor_provider_restGetServiceStatus )
{
    MockMonitorProvider provider;

    auto proxies = std::make_shared<ServiceProxies>();
    ServiceBase parentService("parentservice", proxies);
    MonitorProviderEndpoint endpoint(parentService, provider);

    endpoint.lastStatus_ = Json::parse("{\"status\": \"failure\"}");
    string body = endpoint.restGetServiceStatus();
    BOOST_CHECK_EQUAL(body, "{\"status\":\"failure\"}");

    endpoint.lastStatus_ = Json::parse("{\"status\": \"ok\"}");
    body = endpoint.restGetServiceStatus();
    BOOST_CHECK_EQUAL(body, "{\"status\":\"ok\"}");
}

BOOST_AUTO_TEST_CASE( test_monitor_provider_refreshStatus )
{
    MockMonitorProvider provider;

    auto proxies = std::make_shared<ServiceProxies>();
    ServiceBase parentService("parentservice", proxies);
    MonitorProviderEndpoint endpoint(parentService, provider);

    endpoint.lastStatus_ = Json::parse("{\"status\": \"ok\"}");
 
    provider.status = false;
    endpoint.refreshStatus();
    BOOST_CHECK_EQUAL(endpoint.lastStatus_,
                      Json::parse("{\"status\": \"failure\"}"));

    provider.status = true;
    endpoint.refreshStatus();
    BOOST_CHECK_EQUAL(endpoint.lastStatus_,
                      Json::parse("{\"status\": \"ok\"}"));
}

/* monitor_proxy_test.cc
   Wolfgang Sourdeau, January 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
   
   Unit tests for the MonitorProxy class
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <iostream>

#include <boost/test/unit_test.hpp>
#include "soa/types/date.h"

#include "rtbkit/core/monitor/monitor_proxy.h"

using namespace std;
using namespace Datacratic;
using namespace RTBKIT;

BOOST_AUTO_TEST_CASE( test_monitor_proxy_getStatus )
{
    Date now = Date::now();
    Date lateDate = now.plusSeconds(-3);
    Date pastDate = now.plusSeconds(-0.5);

    /* setup */
    std::shared_ptr<zmq::context_t> zero_context;
    MonitorProxy provider_proxy(zero_context);
    provider_proxy.checkTimeout_ = 1;

    /* no check yet, status = false -> false */
    provider_proxy.lastStatus = false;
    BOOST_CHECK_EQUAL(provider_proxy.getStatus(), false);

    /* timeout, status = true -> false */
    provider_proxy.lastCheck = lateDate;
    provider_proxy.lastStatus = true;
    BOOST_CHECK_EQUAL(provider_proxy.getStatus(), false);

    /* no timeout, status = false -> false */
    provider_proxy.lastCheck = pastDate;
    provider_proxy.lastStatus = false;
    BOOST_CHECK_EQUAL(provider_proxy.getStatus(), false);

    /* no timeout, status = true -> true */
    provider_proxy.lastCheck = pastDate;
    provider_proxy.lastStatus = true;
    BOOST_CHECK_EQUAL(provider_proxy.getStatus(), true);
}

BOOST_AUTO_TEST_CASE( test_monitor_proxy_onResponseReceived )
{
    Date now = Date::now();
    Date pastDate = now.plusSeconds(-0.5);

    /* setup */
    std::shared_ptr<zmq::context_t> zero_context;
    MonitorProxy proxy(zero_context);

    cerr << "test: code 200, invalid json\n=> status = false\n";
    proxy.lastStatus = true;
    proxy.lastCheck = pastDate;
    proxy.onResponseReceived(nullptr, 200, "poil");
    BOOST_CHECK_EQUAL(proxy.lastStatus, false);
    BOOST_CHECK(proxy.lastCheck != pastDate);

    cerr << "test: code 200, valid json, invalid message\n=> status = false\n";
    proxy.lastStatus = true;
    proxy.lastCheck = pastDate;
    proxy.onResponseReceived(nullptr, 200, "null");
    BOOST_CHECK_EQUAL(proxy.lastStatus, false);
    BOOST_CHECK(proxy.lastCheck != pastDate);

    cerr << "test: code 100, valid ok json message\n=> status = false\n";
    proxy.lastStatus = true;
    proxy.lastCheck = pastDate;
    proxy.onResponseReceived(nullptr, 100, "{ 'status': 'ok' }");
    BOOST_CHECK_EQUAL(proxy.lastStatus, false);
    BOOST_CHECK(proxy.lastCheck != pastDate);

    cerr << "test: code 200, valid failure json message\n=> status = false\n";
    proxy.lastStatus = true;
    proxy.lastCheck = pastDate;
    proxy.onResponseReceived(nullptr, 200, "{ 'status': 'failure' }");
    BOOST_CHECK_EQUAL(proxy.lastStatus, false);
    BOOST_CHECK(proxy.lastCheck != pastDate);

    cerr << "test: code 200, valid ok json message\n=> status = true\n";
    proxy.lastStatus = false;
    proxy.lastCheck = pastDate;
    proxy.onResponseReceived(nullptr, 200, "{ 'status': 'ok' }");
    BOOST_CHECK_EQUAL(proxy.lastStatus, true);
    BOOST_CHECK(proxy.lastCheck != pastDate);
}

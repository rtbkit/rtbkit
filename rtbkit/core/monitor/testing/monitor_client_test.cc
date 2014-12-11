/* monitor_client_test.cc
   Wolfgang Sourdeau, January 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
   
   Unit tests for the MonitorClient class
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <iostream>

#include <boost/test/unit_test.hpp>
#include "soa/types/date.h"

#include "rtbkit/core/monitor/monitor_client.h"

using namespace std;
using namespace Datacratic;
using namespace RTBKIT;

BOOST_AUTO_TEST_CASE( test_monitor_client_getStatus )
{
    /* setup */
    Date now = Date::now();
    std::shared_ptr<zmq::context_t> zero_context;
    MonitorClient client(zero_context);
    client.checkTimeout_ = 1;

    BOOST_CHECK_EQUAL(client.getStatus(), false);
    BOOST_CHECK_EQUAL(client.getStatus(10.0), false);

    client.lastSuccess = now.plusSeconds(-0.5);
    BOOST_CHECK_EQUAL(client.getStatus(), true);
}

BOOST_AUTO_TEST_CASE( test_monitor_client_onResponseReceived )
{
    Date now = Date::now();
    Date pastDate = now.plusSeconds(-5.0);

    /* setup */
    std::shared_ptr<zmq::context_t> zero_context;
    MonitorClient client(zero_context);

    cerr << "test: code 200, invalid json\n=> status = false\n";
    client.lastSuccess = pastDate;
    client.onResponseReceived(nullptr, 200, "poil");
    BOOST_CHECK_EQUAL(client.getStatus(), false);

    cerr << "test: code 200, valid json, invalid message\n=> status = false\n";
    client.lastSuccess = pastDate;
    client.onResponseReceived(nullptr, 200, "null");
    BOOST_CHECK_EQUAL(client.getStatus(), false);

    cerr << "test: code 100, valid ok json message\n=> status = false\n";
    client.lastSuccess = pastDate;
    client.onResponseReceived(nullptr, 100, "{ 'status': 'ok' }");
    BOOST_CHECK_EQUAL(client.getStatus(), false);

    cerr << "test: code 200, valid failure json message\n=> status = false\n";
    client.lastSuccess = pastDate;
    client.onResponseReceived(nullptr, 200, "{ 'status': 'failure' }");
    BOOST_CHECK_EQUAL(client.getStatus(), false);

    cerr << "test: code 200, valid ok json message\n=> status = true\n";
    client.lastSuccess = pastDate;
    client.onResponseReceived(nullptr, 200, "{ 'status': 'ok' }");
    BOOST_CHECK_EQUAL(client.getStatus(), true);
}


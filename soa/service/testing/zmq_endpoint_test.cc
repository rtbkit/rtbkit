/** zmq_endpoint_test.cc                                 -*- C++ -*-
    Rémi Attab, 13 Feb 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Tests for the ZMQ endpoints.

*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "jml/arch/timers.h"
#include "jml/utils/testing/watchdog.h"

#include "soa/service/zmq_endpoint.h"

using namespace std;
using namespace ML;
using namespace Datacratic;


/** Tests a specific edge case where a message loop attached to a
    ZmqNamedClientBusProxy would get blocked in zmq because it was attempting to
    send a HEARTBEAT message to a socket that wasn't connected to anything.
 */
BOOST_AUTO_TEST_CASE( test_no_connect )
{
    Watchdog watchdog(10.0);

    auto proxies = std::make_shared<ServiceProxies>();

    ZmqNamedClientBusProxy socket;
    socket.init(proxies->config);
    socket.connectToServiceClass("foo", "bar");

    socket.start();
    ML::sleep(1.0);
    socket.shutdown();
}

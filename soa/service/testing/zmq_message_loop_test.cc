/** zmq_message_loop_test.cc                                 -*- C++ -*-
    Jeremy Barnes, 13 Feb 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Tests for the ZMQ endpoints.

*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "jml/arch/timers.h"
#include "jml/utils/testing/watchdog.h"
#include <atomic>

#include "soa/service/zmq_endpoint.h"

using namespace std;
using namespace ML;
using namespace Datacratic;


BOOST_AUTO_TEST_CASE( test_plain_message_loop )
{
    //Watchdog watchdog(10.0);

    auto proxies = std::make_shared<ServiceProxies>();

    zmq::socket_t sock1(*proxies->zmqContext, ZMQ_PULL);
    zmq::socket_t sock2(*proxies->zmqContext, ZMQ_PUSH);
    
    PortRange ports(10000, 20000);
    std::string uri = bindToOpenTcpPort(sock1, ports, "127.0.0.1");
    sock2.connect(uri);

    ZmqEventSource source(sock1);

    // Allow the message loop to sleep for 2 seconds at a time so that
    // we can easily tell if we're blocking or not
    MessageLoop loop(1, 0.5 /* max added time (seconds) */);
    loop.addSource("asyncMessageSource", source);


    MessageLoop loop2(1, 0.5);
    loop2.addSource("loop1", loop);


    MessageLoop loop3(1, 0.05);
    loop3.addSource("loop2", loop2);

    loop3.start();
    

    std::atomic<int> latency(0);
    std::atomic<int> numSent(0);
    std::atomic<int> numDone(0);

    source.asyncMessageHandler = [&] (const std::vector<std::string> & msg)
        {
            cerr << "got message " << msg << endl;
            Date date = Date::parseIso8601(msg.at(2));
            double elapsed = date.secondsUntil(Date::now());
            latency = elapsed * 1000;
            cerr << "elapsed " << elapsed * 1000.0 << "ms" << endl;
            ++numDone;
        };

    
    while (numSent < 10) {
        sendAll(sock2, { "hello", "world", Date::now().printIso8601() }); 
        ++numSent;
        ML::sleep(0.1);
    }
    
    while (numDone < numSent) {
        ML::sleep(0.1);
    }

    BOOST_CHECK_LE(latency, 100);
    
    //cerr << loop.stats() << endl;

    cerr << "shutting down" << endl;
    Date before = Date::now();
    loop.shutdown();
    Date after = Date::now();
    cerr << "shutdown done" << endl;
    
    BOOST_CHECK_LE(after.secondsSince(before), 0.1);
}

/* message_channel_test.cc                                         -*- C++ -*-
   Jeremy Barnes, 24 September 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Test for message channel ()
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/make_shared.hpp>
#include "soa/service/named_endpoint.h"
#include "soa/service/message_loop.h"
#include "soa/service/typed_message_channel.h"
#include <sys/socket.h>
#include "jml/utils/guard.h"
#include "jml/arch/exception_handler.h"
#include "jml/utils/testing/watchdog.h"
#include "jml/utils/testing/fd_exhauster.h"
#include "jml/utils/vector_utils.h"
#include "jml/arch/timers.h"
#include <thread>
#include "soa/service/zmq_utils.h"
#include <boost/thread/thread.hpp>
#include "jml/utils/testing/watchdog.h"


using namespace std;
using namespace ML;
using namespace Datacratic;



BOOST_AUTO_TEST_CASE( test_message_channel )
{
    TypedMessageSink<std::string> sink(1000);
    
    int numSent = 0;
    int numReceived = 0;
    
    
    sink.onEvent = [&] (const std::string & str)
        {
            ML::atomic_inc(numReceived);
        };

    volatile bool finished = false;

    auto pushThread = [&] ()
        {
            for (unsigned i = 0;  i < 1000;  ++i) {
                sink.push("hello");
                ML::atomic_inc(numSent);
            }
        };

    auto processThread = [&] ()
        {
            while (!finished) {
                sink.processOne();
            }
        };

    int numPushThreads = 2;
    int numProcessThreads = 1;

    for (unsigned i = 0;  i < 100;  ++i) {
        // Test for PLAT-106; the expected behaviour is no deadlock.
        ML::Watchdog watchdog(2.0);

        finished = false;

        boost::thread_group pushThreads;
        for (unsigned i = 0;  i < numPushThreads;  ++i)
            pushThreads.create_thread(pushThread);

        boost::thread_group processThreads;
        for (unsigned i = 0;  i < numProcessThreads;  ++i)
            processThreads.create_thread(processThread);
    
        pushThreads.join_all();

        cerr << "finished push threads" << endl;
    
        finished = true;

        processThreads.join_all();
    }
}

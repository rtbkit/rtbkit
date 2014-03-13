#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <iostream>

#include <boost/test/unit_test.hpp>

#include "jml/arch/timers.h"

#include "soa/service/typed_message_channel.h"
#include "soa/service/message_loop.h"

using namespace std;
using namespace Datacratic;

/* this test causes a crash because sink is destroyed before
 * MessageLoop::shutdown is invoked */
BOOST_AUTO_TEST_CASE( test_destruction_order )
{
    bool doCrash(false);
    MessageLoop loop;
    TypedMessageSink<string> sink(12);

    loop.addSource("sink", sink);
    loop.start();

    string result;
    auto onEvent = [&] (string && msg) {
        cerr << "received and sleeping\n";
        ML::sleep(5.0);
        result += msg;
        cerr << "done sleeping\n";
    };
    sink.onEvent = onEvent;

    cerr << "sending msg 1\n";
    sink.push("This would not cause a crash...");
    ML::sleep(1.0);

    if (!doCrash) {
        loop.shutdown();
    }
}

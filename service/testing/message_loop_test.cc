#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <iostream>

#include <boost/test/unit_test.hpp>

#include "jml/arch/timers.h"
#include "jml/utils/testing/watchdog.h"

#include "soa/service/typed_message_channel.h"
#include "soa/service/message_loop.h"

using namespace std;
using namespace Datacratic;


/* This test ensures that adding sources works correctly when needsPoll is
 * set. Otherwise, the watchdog will be triggered. */
BOOST_AUTO_TEST_CASE( test_addSource_with_needsPoll )
{
    ML::Watchdog wd(5);
    MessageLoop loop;
    loop.needsPoll = true;

    TypedMessageSink<string> aSource(123);
    loop.addSource("source", aSource);
    loop.start();
    aSource.waitConnectionState(AsyncEventSource::CONNECTED);
}

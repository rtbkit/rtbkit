#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "jml/arch/timers.h"

#include "soa/service/typed_message_channel.h"
#include "soa/service/message_loop.h"

using namespace std;
using namespace Datacratic;


/* ensure MessageLoop and TypedMessageSink behave properly when destroyed */
BOOST_AUTO_TEST_CASE( test_destruction )
{
    MessageLoop loop;
    TypedMessageSink<string> sink(12);

    loop.addSource("sink", sink);
    loop.start();

    string result;
    auto onEvent = [&] (string && msg) {
        string recv(msg);
        cerr << "received and sleeping\n";
        ML::sleep(3.0);
        result += recv;
    };
    sink.onEvent = onEvent;

    /* We need to sleep to ensure epoll has caught the wakeup signals from the
     * sink, otherwise the buffer will be destroyed too soon and the crash
     * not occurs */
    sink.push("This would not cause a crash...");
    ML::sleep(1.0);
    sink.push("Hope fully this would not either");
    ML::sleep(1.0);
}

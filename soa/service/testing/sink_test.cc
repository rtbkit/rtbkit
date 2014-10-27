#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <unistd.h>
#include <fcntl.h>

#include <boost/test/unit_test.hpp>

#include <iostream>
#include <sstream>

#include "jml/arch/exception.h"
#include "jml/arch/timers.h"
#include "jml/utils/file_functions.h"

#include "soa/service/message_loop.h"
#include "soa/service/sink.h"

#include "signals.h"

using namespace std;
using namespace Datacratic;

/* OUTPUT SINK */
#if 1
BOOST_AUTO_TEST_CASE( test_outputsink_istream )
{
    string received;
    bool closed(false);

    auto onData = [&] (string && data) {
        received += move(data);
        return true;
    };
    auto onClose = [&] () {
        closed = true;
    };

    CallbackOutputSink outputSink(onData, onClose);

    string data("I am sending data.");
    string expected("Iamsendingdata.");
    std::istringstream datastream(data);

    string testdata;
    while (datastream) {
        datastream >> outputSink;
    }

    BOOST_CHECK_EQUAL(received, expected);
}
#endif

#if 1
BOOST_AUTO_TEST_CASE( test_asyncfdoutputsink_hup )
{
    BlockedSignals sigs(SIGPIPE);

    int fds[2];

    int res = pipe(fds);
    if (res == -1) {
        throw ML::Exception(errno, "pipe");
    }

    bool hup(false), closed(false);
    auto onHangup = [&] () {
        hup = true;
    };
    auto onClose = [&] () {
        closed = true;
    };

    MessageLoop loop;
    loop.start();

    auto sink = make_shared<AsyncFdOutputSink>(onHangup, onClose);
    loop.addSource("sink", sink);
    ML::set_file_flag(fds[1], O_NONBLOCK);
    sink->init(fds[1]);

    ML::sleep(1.0);
    ::close(fds[0]);
    sink->write("test message");
    ML::sleep(1.0);
    BOOST_CHECK_EQUAL(hup, true);

    loop.shutdown();
}
#endif

#if 0
/* Disabled test because of the time it requires, for a feature we know is
 * working. */

/* Ensures that the all messages are sent and in correct order, in spite of
 * the bursting of the queue */
BOOST_AUTO_TEST_CASE( test_asyncfdoutputsink_many_msgs )
{
    const int nmsgs(200000);

    int fds[2];

    int res = pipe(fds);
    if (res == -1) {
        throw ML::Exception(errno, "pipe");
    }

    bool hup(false), closed(false);
    auto onHangup = [&] () {
        cerr << "hungup\n";
        hup = true;
    };
    auto onClose = [&] () {
        cerr << "closed\n";
        closed = true;
    };

    MessageLoop loop;
    loop.start();

    AsyncFdOutputSink sink(onHangup, onClose);
    loop.addSource("sink", sink);
    ML::set_file_flag(fds[1], O_NONBLOCK);
    sink.init(fds[1]);

    const string basemsg("this is a message");
    size_t fullmsgsize(basemsg.size() + 4);

    cerr << "piping " + to_string(nmsgs) + " messages\n";
    char buffer[8];
    int i;
    for (i = 0; i < nmsgs; i++) {
        sprintf(buffer, "%.4x\n", i);
        string message = basemsg + string(buffer, 4);
        while (sink.state == OutputSink::OPEN
               && !sink.write(string(message))) {
            cerr << "retrying at " + to_string(i) << endl;
        }

        char buffer[1024];
        int res = ::read(fds[0], buffer, fullmsgsize);
        if (res == -1) {
            throw ML::Exception(errno, "read");
        }

        string recvmsg(buffer, fullmsgsize);
        if (recvmsg != message) {
            throw ML::Exception("message " + to_string(i) + " does not match");
        }
    }
    sink.requestClose();
    sink.waitState(OutputSink::CLOSED);

    loop.removeSource(&sink);
    while (loop.poll()) {
        ML::sleep(1.0);
    }

    BOOST_CHECK_EQUAL(i, nmsgs);

    loop.shutdown();
}
#endif

/* INPUT SINK */
#if 1
BOOST_AUTO_TEST_CASE( test_callbackinputsink )
{
    string received;
    bool closed(false);

    auto onData = [&] (string && data) {
        received += move(data);
    };
    auto onClose = [&] () {
        closed = true;
    };

    CallbackInputSink inputSink(onData, onClose);

    string data1("I am ");
    string expected1(data1);
    inputSink.notifyReceived(move(data1));
    BOOST_CHECK_EQUAL(received, expected1);
    BOOST_CHECK_EQUAL(closed, false);

    string data2("sending data.");
    string expected2(data1 + data2);
    inputSink.notifyReceived(move(data2));
    BOOST_CHECK_EQUAL(received, expected2);
    BOOST_CHECK_EQUAL(closed, false);

    inputSink.notifyClosed();
    BOOST_CHECK_EQUAL(closed, true);
}
#endif

#if 1
BOOST_AUTO_TEST_CASE( test_ostreaminputsink )
{
    ostringstream received;

    OStreamInputSink outputSink(&received);

    string data("I am sending data.");
    string expected(data);
    outputSink.notifyReceived(move(data));

    BOOST_CHECK_EQUAL(received.str(), expected);
}
#endif

#if 1
BOOST_AUTO_TEST_CASE( test_chaininputsink )
{
    ostringstream stream1;
    ostringstream stream2;

    auto sink1 = make_shared<OStreamInputSink>(&stream1);
    auto sink2 = make_shared<OStreamInputSink>(&stream2);

    ChainInputSink chainSink;
    chainSink.appendSink(sink1);
    chainSink.appendSink(sink2);

    string data("I am sending data.");
    string expected(data);
    chainSink.notifyReceived(move(data));

    BOOST_CHECK_EQUAL(stream1.str(), expected);
    BOOST_CHECK_EQUAL(stream2.str(), expected);
}
#endif

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <sys/types.h>
#include <sys/wait.h>

#include <mutex>
#include <thread>

#include <boost/test/unit_test.hpp>

#include "jml/arch/atomic_ops.h"
#include "jml/arch/exception.h"
#include "jml/arch/futex.h"
#include "jml/arch/timers.h"
#include "jml/utils/exc_assert.h"
#include "jml/utils/string_functions.h"
#include "soa/service/message_loop.h"
#include "soa/service/runner.h"
#include "soa/service/sink.h"
#include "soa/types/date.h"

#include <iostream>

#include "runner_test.h"

#include "signals.h"

using namespace std;
using namespace Datacratic;

// #define BOOST_CHECK_EQUAL(x,y)  { ExcCheckEqual((x), (y), ""); }

struct _Init {
    _Init() {
        signal(SIGPIPE, SIG_IGN);
    }
} myInit;


#if 1
/* ensures that the basic callback system works */
BOOST_AUTO_TEST_CASE( test_runner_callbacks )
{
    BlockedSignals blockedSigs2(SIGCHLD);

    MessageLoop loop;

    RunnerTestHelperCommands commands;
    commands.sendOutput(true, "hello stdout");
    commands.sendOutput(true, "hello stdout2");
    commands.sendOutput(false, "hello stderr");
    commands.sendExit(0);

    string receivedStdOut, expectedStdOut;
    string receivedStdErr, expectedStdErr;

    expectedStdOut = ("helper: ready\nhello stdout\nhello stdout2\n"
                      "helper: exit with code 0\n");
    expectedStdErr = "hello stderr\n";

    int done = false;
    auto onTerminate = [&] (const Runner::RunResult & result) {
        done = true;
        ML::futex_wake(done);
    };

    auto onStdOut = [&] (string && message) {
        // cerr << "received message on stdout: /" + message + "/" << endl;
        receivedStdOut += message;
    };
    auto stdOutSink = make_shared<CallbackInputSink>(onStdOut);

    auto onStdErr = [&] (string && message) {
        // cerr << "received message on stderr: /" + message + "/" << endl;
        receivedStdErr += message;
    };
    auto stdErrSink = make_shared<CallbackInputSink>(onStdErr);

    Runner runner;
    loop.addSource("runner", runner);
    loop.start();

    auto & stdInSink = runner.getStdInSink();
    runner.run({"build/x86_64/bin/runner_test_helper"},
               onTerminate, stdOutSink, stdErrSink);
    for (const string & command: commands) {
        while (!stdInSink.write(string(command))) {
            ML::sleep(0.1);
        }
    }
    stdInSink.requestClose();

    while (!done) {
        ML::futex_wait(done, false);
    }

    BOOST_CHECK_EQUAL(ML::hexify_string(receivedStdOut),
                      ML::hexify_string(expectedStdOut));
    BOOST_CHECK_EQUAL(ML::hexify_string(receivedStdErr),
                      ML::hexify_string(expectedStdErr));

    loop.shutdown();
}
#endif

#if 1
/* ensures that the returned status is properly set after termination */
BOOST_AUTO_TEST_CASE( test_runner_normal_exit )
{
    BlockedSignals blockedSigs(SIGCHLD);

    auto nullSink = make_shared<NullInputSink>();

    /* normal termination, with code */
    {
        MessageLoop loop;

        RunnerTestHelperCommands commands;
        commands.sendExit(123);

        Runner::RunResult result;
        auto onTerminate = [&] (const Runner::RunResult & newResult) {
            result = newResult;
        };
        Runner runner;
        loop.addSource("runner", runner);
        loop.start();

        auto & stdInSink = runner.getStdInSink();
        runner.run({"build/x86_64/bin/runner_test_helper"},
                   onTerminate, nullSink, nullSink);
        for (const string & command: commands) {
            stdInSink.write(string(command));
        }
        stdInSink.requestClose();
        runner.waitTermination();

        BOOST_CHECK_EQUAL(result.signaled, false);
        BOOST_CHECK_EQUAL(result.returnCode, 123);

        loop.shutdown();
    }

    /* aborted termination, with signum */
    {
        MessageLoop loop;

        RunnerTestHelperCommands commands;
        commands.sendAbort();

        Runner::RunResult result;
        auto onTerminate = [&] (const Runner::RunResult & newResult) {
            result = newResult;
        };
        Runner runner;
        loop.addSource("runner", runner);
        loop.start();

        auto & stdInSink = runner.getStdInSink();
        runner.run({"build/x86_64/bin/runner_test_helper"},
                   onTerminate, nullSink, nullSink);
        for (const string & command: commands) {
            stdInSink.write(string(command));
        }
        stdInSink.requestClose();
        runner.waitTermination();

        BOOST_CHECK_EQUAL(result.signaled, true);
        BOOST_CHECK_EQUAL(result.returnCode, SIGABRT);

        loop.shutdown();
    }
}
#endif

#if 1
/* test the behaviour of the Runner class when attempting to launch a missing
 * executable, mostly mimicking bash */
BOOST_AUTO_TEST_CASE( test_runner_missing_exe )
{
    MessageLoop loop;

    loop.start();

    Runner::RunResult result;
    auto onTerminate = [&] (const Runner::RunResult & newResult) {
        result = newResult;
    };

    /* running a program that does not exist */
    {
        Runner runner;
        loop.addSource("runner1", runner);

        runner.run({"/this/command/is/missing"}, onTerminate);
        runner.waitTermination();

        BOOST_CHECK_EQUAL(result.signaled, false);
        BOOST_CHECK_EQUAL(result.returnCode, 127);

        loop.removeSource(&runner);
        runner.waitConnectionState(AsyncEventSource::DISCONNECTED);
    }

    /* running a non-executable but existing file */
    {
        Runner runner;
        loop.addSource("runner2", runner);

        runner.run({"/dev/null"}, onTerminate);
        runner.waitTermination();

        BOOST_CHECK_EQUAL(result.signaled, false);
        BOOST_CHECK_EQUAL(result.returnCode, 126);

        loop.removeSource(&runner);
        runner.waitConnectionState(AsyncEventSource::DISCONNECTED);
    }

    /* running a non-executable but existing non-file */
    {
        Runner runner;
        loop.addSource("runner2", runner);

        runner.run({"/dev"}, onTerminate);
        runner.waitTermination();

        BOOST_CHECK_EQUAL(result.signaled, false);
        BOOST_CHECK_EQUAL(result.returnCode, 126);

        loop.removeSource(&runner);
        runner.waitConnectionState(AsyncEventSource::DISCONNECTED);
    }

    loop.shutdown();
}
#endif

#if 1
/* test the "execute" function */
BOOST_AUTO_TEST_CASE( test_runner_execute )
{
    string received;
    auto onStdOut = [&] (string && message) {
        received = move(message);
    };
    auto stdOutSink = make_shared<CallbackInputSink>(onStdOut, nullptr);

    auto result = execute({"/bin/cat", "-"},
                          stdOutSink, nullptr, "hello callbacks");
    BOOST_CHECK_EQUAL(received, "hello callbacks");
    BOOST_CHECK_EQUAL(result.signaled, false);
    BOOST_CHECK_EQUAL(result.returnCode, 0);
}
#endif

#if 1
/* perform multiple runs with the same Runner and ensures task-specific
 * components are properly segregated */
BOOST_AUTO_TEST_CASE( test_runner_cleanup )
{
    MessageLoop loop;

    Runner runner;
    loop.addSource("runner", runner);
    loop.start();

    auto nullSink = make_shared<NullInputSink>();

    auto performLoop = [&] (const string & loopData) {
        RunnerTestHelperCommands commands;
        commands.sendOutput(true, loopData);
        commands.sendExit(0);

        string expectedStdOut("helper: ready\n" + loopData
                              + "\nhelper: exit with code 0\n");
        string receivedStdOut;
        auto onStdOut = [&] (string && message) {
            // cerr << "received message on stdout: /" + message + "/" << endl;
            receivedStdOut += message;
        };
        auto stdOutSink = make_shared<CallbackInputSink>(onStdOut);

        auto & stdInSink = runner.getStdInSink();
        runner.run({"build/x86_64/bin/runner_test_helper"},
                   nullptr, stdOutSink, nullSink);
        for (const string & command: commands) {
            stdInSink.write(string(command));
        }
        stdInSink.requestClose();
        runner.waitTermination();

        BOOST_CHECK_EQUAL(ML::hexify_string(receivedStdOut),
                          ML::hexify_string(expectedStdOut));
    };

    for (int i = 0; i < 5; i++) {
        performLoop(to_string(i));
    }

    loop.shutdown();
}
#endif

#if 1
/* Ensures that the output is received as soon as it is emitted, and not by
 * chunks. This is done by expecting different types of strings: a simple one
 * with a few chars, another one with two carriage returns and a third one
 * with 1024 chars. The test works by ensuring that all strings are received
 * one by one, with a relatively precise and constant delay of 1 second
 * between them. */
static void
test_runner_no_output_delay_helper(bool stdout)
{
    double delays[3];
    int sizes[3];
    int pos(stdout ? -1 : 0);
    shared_ptr<CallbackInputSink> stdOutSink(nullptr);
    shared_ptr<CallbackInputSink> stdErrSink(nullptr);

    Date start = Date::now();
    Date last = start;

    auto onCapture = [&] (string && message) {
        Date now = Date::now();
        if (pos > -1 && pos < 3) {
            /* skip "helper: ready" message */
            delays[pos] = now.secondsSinceEpoch() - last.secondsSinceEpoch();
            sizes[pos] = message.size();
        }
        pos++;
        last = now;
    };
    if (stdout) {
        stdOutSink.reset(new CallbackInputSink(onCapture));
    }
    else {
        stdErrSink.reset(new CallbackInputSink(onCapture));
    }

    RunnerTestHelperCommands commands;
    commands.sendSleep(10);
    commands.sendOutput(stdout, "first");
    commands.sendSleep(10);
    commands.sendOutput(stdout, "second\nsecond");
    commands.sendSleep(10);

    string third;
    for (int i = 0; i < 128; i++) {
        third += "abcdefgh";
    }
    commands.sendOutput(stdout, third);
    commands.sendSleep(10);
    commands.sendExit(0);

    MessageLoop loop;
    Runner runner;
    loop.addSource("runner", runner);
    loop.start();

    auto & stdInSink = runner.getStdInSink();
    runner.run({"build/x86_64/bin/runner_test_helper"},
               nullptr, stdOutSink, stdErrSink);
    for (const string & command: commands) {
        while (!stdInSink.write(string(command))) {
            ML::sleep(0.1);
        }
    }
    stdInSink.requestClose();
    Date end = Date::now();
    runner.waitTermination();

    BOOST_CHECK_EQUAL(sizes[0], 6);
    BOOST_CHECK(delays[0] >= 0.9);
    BOOST_CHECK_EQUAL(sizes[1], 14);
    BOOST_CHECK(delays[1] >= 0.9);
    BOOST_CHECK_EQUAL(sizes[2], 1025);
    BOOST_CHECK(delays[2] >= 0.9);

    for (int i = 0; i < 3; i++) {
        ::fprintf(stderr, "%d: size: %d; delay: %f\n", i, sizes[i], delays[i]);
    }
}

BOOST_AUTO_TEST_CASE( test_runner_no_output_delay_stdout )
{
    test_runner_no_output_delay_helper(true);
}

BOOST_AUTO_TEST_CASE( test_runner_no_output_delay_stderr )
{
    test_runner_no_output_delay_helper(false);
}
#endif


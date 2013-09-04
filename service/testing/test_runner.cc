#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "jml/arch/exception.h"
#include "jml/arch/futex.h"
#include "jml/utils/exc_assert.h"
#include "jml/utils/string_functions.h"
#include "soa/service/message_loop.h"
#include "soa/service/runner.h"

#include <iostream>

#include "signals.h"

using namespace std;
using namespace Datacratic;

// #define BOOST_CHECK_EQUAL(x,y)  { ExcCheckEqual((x), (y), ""); }

struct HelperCommands : vector<string>
{
    HelperCommands()
        : vector<string>(),
          active_(0)
    {}

    void reset() { active_ = 0; }

    string nextCommand()
    {
        if (active_ < size()) {
            int active = active_;
            active_++;
            return at(active);
        }
        else {
            return "";
        }
    }

    void sendOutput(bool isStdOut, const string & data)
    {
        char cmdBuffer[1024];
        int len = data.size();
        int totalLen = len + 3 + sizeof(int);
        sprintf(cmdBuffer, (isStdOut ? "out" : "err"));
        memcpy(cmdBuffer + 3, &len, sizeof(int));
        memcpy(cmdBuffer + 3 + sizeof(int), data.c_str(), len);
        push_back(string(cmdBuffer, totalLen));
    }

    void sendExit(int code)
    {
        char cmdBuffer[1024];
        int totalLen = 3 + sizeof(int);
        sprintf(cmdBuffer, "xit");
        memcpy(cmdBuffer + 3, &code, sizeof(int));
        push_back(string(cmdBuffer, totalLen));
    };

    void sendAbort()
    {
        push_back("abt");
    }

    int active_;
};

/* ensures that the basic callback system works */
BOOST_AUTO_TEST_CASE( test_runner_callbacks )
{
    BlockedSignals blockedSigs(SIGCHLD);

    MessageLoop loop;

    HelperCommands commands;
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
    auto onTerminate = [&] (const AsyncRunner::RunResult & result) {
        done = true;
        ML::futex_wake(done);
    };

    auto onStdIn = [&] () {
        return commands.nextCommand();
    };
    auto onStdOut = [&] (const string & message) {
        // cerr << "received message on stdout: /" + message + "/" << endl;
        receivedStdOut += message;
    };
    auto onStdErr = [&] (const string & message) {
        // cerr << "received message on stderr: /" + message + "/" << endl;
        receivedStdErr += message;
    };

    AsyncRunner runner;
    loop.addSource("runner", runner);
    loop.start();

    runner.run({"build/x86_64/bin/test_runner_helper"},
               onTerminate, onStdOut, onStdErr, onStdIn);

    while (!done) {
        ML::futex_wait(done, false);
    }

    BOOST_CHECK_EQUAL(receivedStdOut, expectedStdOut);
    BOOST_CHECK_EQUAL(receivedStdErr, expectedStdErr);
}

/* ensures that the returned status is properly set after termination */
BOOST_AUTO_TEST_CASE( test_runner_normal_exit )
{
    BlockedSignals blockedSigs(SIGCHLD);

    /* normal termination, with code */
    {
        MessageLoop loop;

        HelperCommands commands;
        commands.sendExit(123);

        AsyncRunner::RunResult result;
        auto onTerminate = [&] (const AsyncRunner::RunResult & newResult) {
            result = newResult;
        };
        auto onStdIn = [&] () {
            return commands.nextCommand();
        };
        auto discard = [&] (const string & message) {
        };
        AsyncRunner runner;
        loop.addSource("runner", runner);
        loop.start();

        runner.run({"build/x86_64/bin/test_runner_helper"},
                   onTerminate, discard, discard, onStdIn);
        runner.waitTermination();

        BOOST_CHECK_EQUAL(result.signaled, false);
        BOOST_CHECK_EQUAL(result.returnCode, 123);
    }

    /* aborted termination, with signum */
    {
        MessageLoop loop;

        HelperCommands commands;
        commands.sendAbort();

        AsyncRunner::RunResult result;
        auto onTerminate = [&] (const AsyncRunner::RunResult & newResult) {
            result = newResult;
        };
        auto onStdIn = [&] () {
            return commands.nextCommand();
        };
        auto discard = [&] (const string & message) {
        };
        AsyncRunner runner;
        loop.addSource("runner", runner);
        loop.start();

        runner.run({"build/x86_64/bin/test_runner_helper"},
                   onTerminate, discard, discard, onStdIn);
        runner.waitTermination();

        BOOST_CHECK_EQUAL(result.signaled, true);
        BOOST_CHECK_EQUAL(result.returnCode, SIGABRT);
    }
}

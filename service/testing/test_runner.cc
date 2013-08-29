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


BOOST_AUTO_TEST_CASE( test_runner )
{
    BlockedSignals blockedSigs(SIGCHLD);

    HelperCommands commands;

    MessageLoop loop;
    AsyncRunner runner({"build/x86_64/bin/test_runner_helper"});

    int done(0);
    auto onTerminate = [&] (int rc, int code) {
        done = 1;
        ML::futex_wake(done);
    };

    auto onStdIn = [&] () {
        return commands.nextCommand();
    };
    auto onStdOut = [&] (const string & message) {
        fprintf(stderr, "test: stdout: /%s/\n", message.c_str());
    };
    auto onStdErr = [&] (const string & message) {
        fprintf(stderr, "test: stderr: /%s/\n", message.c_str());
    };

    runner.init(onTerminate, onStdOut, onStdErr, onStdIn);
    loop.addSource("runner", runner);
    loop.start();

    commands.sendOutput(true, "hello stdout");
    commands.sendOutput(true, "hello stdout2");
    commands.sendOutput(false, "hello stderr");
    commands.sendExit(0);

    runner.run();

    while (!done) {
        ML::futex_wait(done, 1);
    }
}

BOOST_AUTO_TEST_CASE( test_runner_normal_exit )
{
    BlockedSignals blockedSigs(SIGCHLD);
    MessageLoop loop;

    AsyncRunner runner({"build/x86_64/bin/test_runner_helper"});
    HelperCommands commands;

    int done(0);
    auto onTerminate = [&] (int rc, int code) {
        cerr << "test: onTerminate\n";
        done = 1;
        ML::futex_wake(done);
    };
    auto onStdIn = [&] () {
        return commands.nextCommand();
    };
    auto discard = [&] (const string & message) {
    };
    runner.init(onTerminate, discard, discard, onStdIn);
    loop.addSource("runner", runner);
    loop.start();

    commands.sendExit(123);

    runner.run();
    while (!done) {
        ML::futex_wait(done, 1);
    }

    BOOST_CHECK_EQUAL(runner.lastSignal(), -1);
    BOOST_CHECK_EQUAL(runner.lastReturnCode(), 123);
}

BOOST_AUTO_TEST_CASE( test_runner_abort )
{
    BlockedSignals blockedSigs(SIGCHLD);
    MessageLoop loop;

    AsyncRunner runner({"build/x86_64/bin/test_runner_helper"});
    HelperCommands commands;

    int done(0);
    auto onTerminate = [&] (int rc, int code) {
        cerr << "test: onTerminate\n";
        done = 1;
        ML::futex_wake(done);
    };
    auto onStdIn = [&] () {
        return commands.nextCommand();
    };
    auto discard = [&] (const string & message) {
    };
    runner.init(onTerminate, discard, discard, onStdIn);
    loop.addSource("runner", runner);
    loop.start();

    commands.sendAbort();

    runner.run();
    while (!done) {
        ML::futex_wait(done, 1);
    }

    BOOST_CHECK_EQUAL(runner.lastSignal(), SIGABRT);
    BOOST_CHECK_EQUAL(runner.lastReturnCode(), -1);
}

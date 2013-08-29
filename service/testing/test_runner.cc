// #define BOOST_TEST_MAIN
// #define BOOST_TEST_DYN_LINK

// #include <boost/test/unit_test.hpp>

#include "jml/arch/exception.h"
#include "jml/arch/futex.h"
#include "jml/utils/exc_assert.h"
#include "jml/utils/string_functions.h"
#include "soa/service/message_loop.h"
#include "soa/service/runner.h"

#include "signals.h"

using namespace std;
// using namespace ML;
using namespace Datacratic;

#define BOOST_CHECK_EQUAL(x,y)  { ExcCheckEqual((x), (y), ""); }

// BOOST_AUTO_TEST_CASE( test_runner )
int main()
{
    // DisabledSignal disSig(SIGCHLD);

    // char * newEnv = strdup("BOOST_TEST_CATCH_SYSTEM_ERRORS=no");
    // ::putenv(newEnv);

    vector<string> commands;
    size_t current(0);

    MessageLoop loop;
    AsyncRunner runner({"build/x86_64/bin/test_runner_helper"});

    int done(0);
    auto onTerminate = [&] (int rc, int code) {
        cerr << "test: onTerminate\n";
        done = 1;
        ML::futex_wake(done);
    };

    auto onStdIn = [&] () {
        if (current < commands.size()) {
            return commands[current++];
        }
        else {
            return string("");
        }
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

    auto sendOutput = [&] (bool isStdOut, const string & data) {
        char cmdBuffer[1024];
        int len = data.size();
        int totalLen = len + 3 + sizeof(int);
        sprintf(cmdBuffer, (isStdOut ? "out" : "err"));
        memcpy(cmdBuffer + 3, &len, sizeof(int));
        memcpy(cmdBuffer + 3 + sizeof(int), data.c_str(), len);
        commands.push_back(string(cmdBuffer, totalLen));
    };
    auto sendExit = [&] (int code) {
        char cmdBuffer[1024];
        int totalLen = 3 + sizeof(int);
        sprintf(cmdBuffer, "xit");
        memcpy(cmdBuffer + 3, &code, sizeof(int));
        commands.push_back(string(cmdBuffer, totalLen));
    };

    cerr << "test: preparing commands\n";
    sendOutput(true, "hello stdout");
    sendOutput(true, "hello stdout2");
    sendOutput(false, "hello stderr");
    sendExit(123);

    cerr << "test: runner.run()\n";
    runner.run();

    while (!done) {
        ML::futex_wait(done, 1);
    }

    BOOST_CHECK_EQUAL(runner.lastSignal(), 6);
    BOOST_CHECK_EQUAL(runner.lastReturnCode(), -1);

    loop.shutdown();

    return 0;
}

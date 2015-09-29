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
#include "jml/utils/testing/watchdog.h"
#include "soa/service/message_loop.h"
#include "soa/service/runner.h"
#include "soa/service/sink.h"
#include "soa/types/date.h"

#include <iostream>

#include "signals.h"

using namespace std;
using namespace Datacratic;

// #define BOOST_CHECK_EQUAL(x,y)  { ExcCheckEqual((x), (y), ""); }

struct _Init {
    _Init() {
        signal(SIGPIPE, SIG_IGN);
    }
} myInit;

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

#if 1
/* stress test that runs 20 threads in parallel, where each thread:
- invoke "execute", with 10000 messages to stderr and stdout (each)
  received from the stdin sink
- compare those messages with a fixture
and
- the parent thread that outputs messages on stderr and on stdout until all
  threads are done
- wait for the termination of all threads
- ensures that all child process have properly exited
*/
BOOST_AUTO_TEST_CASE( test_stress_runner )
{
    ML::Watchdog wd(120);
    vector<thread> threads;
    int nThreads(20), activeThreads;
    vector<pid_t> childPids(nThreads);
    int msgsToSend(10000);
    atomic<int> nRunning(0);

    activeThreads = nThreads;
    auto runThread = [&] (int threadNum) {
        /* preparation */
        HelperCommands commands;
        string receivedStdOut, expectedStdOut;
        string receivedStdErr, expectedStdErr;
        size_t stdInBytes(0);

        receivedStdOut.reserve(msgsToSend * 80);
        expectedStdOut.reserve(msgsToSend * 80);
        receivedStdErr.reserve(msgsToSend * 80);
        expectedStdErr.reserve(msgsToSend * 80);

        expectedStdOut = "helper: ready\n";
        for (int i = 0; i < msgsToSend; i++) {
            string stdOutData = (to_string(threadNum)
                                 + ":" + to_string(i)
                                 + ": this is a message to stdout\n\t"
                                 + "and a tabbed line");
            commands.sendOutput(true, stdOutData);
            expectedStdOut += stdOutData + "\n";
            string stdErrData = (to_string(threadNum)
                                 + ":" + to_string(i)
                                 + ": this is a message to stderr\n\t"
                                 + "and a tabbed line");
            commands.sendOutput(false, stdErrData);
            expectedStdErr += stdErrData + "\n";
        }
        commands.sendExit(0);

        expectedStdOut += "helper: exit with code 0\n";

        /* execution */
        MessageLoop loop;
        Runner runner;

        loop.addSource("runner", runner);
        loop.start();

        auto onStdOut = [&] (string && message) {
            receivedStdOut += message;
        };
        auto stdOutSink = make_shared<CallbackInputSink>(onStdOut);
        auto onStdErr = [&] (string && message) {
            receivedStdErr += message;
        };
        auto stdErrSink = make_shared<CallbackInputSink>(onStdErr);

        auto & stdInSink = runner.getStdInSink();
        runner.run({"build/x86_64/bin/runner_test_helper"},
                   nullptr, stdOutSink, stdErrSink);

        for (const string & command: commands) {
            while (!stdInSink.write(string(command))) {
                ML::sleep(0.01);
            }
            stdInBytes += command.size();
        }

        runner.waitStart();
        pid_t pid = runner.childPid();
        nRunning++;
        // cerr << "running with pid: " + to_string(pid) + "\n";

        childPids[threadNum] = pid;

        // cerr << "sleeping\n";
        ML::sleep(1.0);

        /* before closing stdinsink, wait until all bytes are correctly
           sent */
        {
            AsyncFdOutputSink & sinkPtr = (AsyncFdOutputSink &) stdInSink;
            while (sinkPtr.bytesSent() != stdInBytes) {
                ML::sleep(0.2);
            }
            stdInSink.requestClose();
        }

        // cerr << "waiting termination...\n";
        runner.waitTermination();
        // cerr << "terminated\n";

        loop.shutdown();

        BOOST_CHECK_EQUAL(receivedStdOut, expectedStdOut);
        BOOST_CHECK_EQUAL(receivedStdErr, expectedStdErr);

        ML::atomic_dec(activeThreads);
        // cerr << "activeThreads now: " + to_string(activeThreads) + "\n";
        if (activeThreads == 0) {
            ML::futex_wake(activeThreads);
        }
        cerr << "thread shutting down\n";
    };

    /* initialize childPids with a non-random bad value, so that we can know
     * later whether the pids were correctly initialized from the workers */
    for (int i = 0; i < nThreads; i++) {
        childPids[i] = 0xdeadface;
    }

    ML::memory_barrier();

    for (int i = 0; i < nThreads; i++) {
        threads.emplace_back(runThread, i);
    }

    ML::memory_barrier();

    /* attempting to interfere with stdout/stderr as long as all thread have
     * not redirected their output channels yet (are not running) */
    while (nRunning < nThreads) {
        cout << "performing interference on stdout\n";
        cerr << "performing interference on stderr\n";
        // int n = activeThreads;
        // ML::futex_wait(activeThreads, n);
    }

    for (thread & current: threads) {
        current.join();
    }
 
    /* ensure children have all exited... */
    BOOST_CHECK_EQUAL(childPids.size(), threads.size());
    for (const int & pid: childPids) {
        if (pid != 0xdeadface) {
            /* the child may already be done when childPid was invoked */
            if (pid > 0) {
                waitpid(pid, NULL, WNOHANG);
                int errno_ = errno;
                BOOST_CHECK_EQUAL(errno_, ECHILD);
            }
        }
        else {
            throw ML::Exception("pid was never set");
        }
    }
}

#endif

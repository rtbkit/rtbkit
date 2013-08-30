#pragma once

#include <signal.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "epoller.h"


namespace Datacratic {

struct AsyncRunner: public Epoller {
    struct RunResult {
        RunResult()
        : signaled(false), signum(-1)
        {}

        void updateFromStatus(int status);

        bool signaled;
        union {
            // signum if signaled, error code otherwise
            int signum;
            int returnCode;
        };
    };

    typedef std::function<void (const RunResult & result)> OnTerminate;
    typedef std::function<void (const std::string & newOutput)> OnOutput;
    typedef std::function<std::string (void)> OnInput;

    AsyncRunner(const std::vector<std::string> & command);

    void init(const OnTerminate & onTerminate = nullptr,
              const OnOutput & onStdOut = nullptr,
              const OnOutput & onStdErr = nullptr,
              const OnInput & onStdIn = nullptr);

    void run();
    void kill(int signal = SIGTERM);
    void waitTermination();

    bool running() const { return running_; }
    int childPid() const { return childPid_; }
    RunResult lastRunResult() const { return runResult_; }

private:
    void prepareChild();
    bool handleEpollEvent(const struct epoll_event & event);
    void handleChildStatus(const struct epoll_event & event);
    void updateTerminationStatus(int status);
    void handleChildInput(const struct epoll_event & event);
    void handleChildOutput(const struct epoll_event & event,
                           const OnOutput & onOutputFn);
    void postTerminate();

    std::vector<std::string> command_;
    OnTerminate onTerminate_;
    OnInput onStdIn_;
    OnOutput onStdOut_;
    OnOutput onStdErr_;

    int running_;
    pid_t childPid_;
    pid_t wrapperPid_;

    RunResult runResult_;

    int stdInFd_;
    int stdOutFd_;
    int stdErrFd_;
    int statusFd_;
};

}

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

        RunResult(int status)
        {
            updateFromStatus(status);
        }

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

    AsyncRunner();

    void run(const std::vector<std::string> & command,
             const OnTerminate & onTerminate = nullptr,
             const OnOutput & onStdOut = nullptr,
             const OnOutput & onStdErr = nullptr,
             const OnInput & onStdIn = nullptr);
    void kill(int signal = SIGTERM);
    void waitTermination();

    bool running() const { return task_ != nullptr; }
    int childPid() const { return task_->childPid; }

private:
    struct Task {
        Task()
            : childPid(-1),
              wrapperPid(-1),
              stdInFd(-1),
              stdOutFd(-1),
              stdErrFd(-1),
              statusFd(-1)
        {}

        void postTerminate(AsyncRunner & runner, const RunResult & runResult);

        std::vector<std::string> command;
        OnTerminate onTerminate;
        OnInput onStdIn;
        OnOutput onStdOut;
        OnOutput onStdErr;

        pid_t childPid;
        pid_t wrapperPid;

        int stdInFd;
        int stdOutFd;
        int stdErrFd;
        int statusFd;
    };

    void prepareChild();
    bool handleEpollEvent(const struct epoll_event & event);
    void handleChildStatus(const struct epoll_event & event,
                           int fd, Task & task);
    void handleChildInput(const struct epoll_event & event,
                          int fd, const OnInput & onInputFn);
    void handleChildOutput(const struct epoll_event & event,
                           int fd, const OnOutput & onOutputFn);
    void postTerminate();

    int running_;
    std::unique_ptr<Task> task_;
};

AsyncRunner::RunResult Execute(const std::vector<std::string> & command,
                               const AsyncRunner::OnOutput & onStdOut
                               = nullptr,
                               const AsyncRunner::OnOutput & onStdErr
                               = nullptr,
                               const AsyncRunner::OnInput & onStdIn
                               = nullptr);

}

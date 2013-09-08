/* runner.h                                                        -*- C++ -*-
   Wolfgang Sourdeau, September 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.

   A command runner class that hides the specifics of the underlying unix
   system calls and can intercept input and output.
*/

#pragma once

#include <signal.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "jml/utils/ring_buffer.h"
#include "epoller.h"
#include "sink.h"
#include "typed_message_channel.h"


namespace Datacratic {

/* ASYNCRUNNER */

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

    AsyncRunner();
    ~AsyncRunner();

    OutputSink & getStdInSink();

    void run(const std::vector<std::string> & command,
             const OnTerminate & onTerminate = nullptr,
             const std::shared_ptr<InputSink> & stdOutSink = nullptr,
             const std::shared_ptr<InputSink> & stdErrSink = nullptr);
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

        void setupInSink();
        void flushInSink();
        void flushStdInBuffer();
        void postTerminate(AsyncRunner & runner, const RunResult & runResult);

        std::vector<std::string> command;
        OnTerminate onTerminate;

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
    void handleOutputStatus(const struct epoll_event & event,
                            int fd, InputSink & inputSink);

    void onStdInClosed(const OutputSink & sink);

    int running_;

    std::shared_ptr<AsyncFdOutputSink> stdInSink_;
    std::shared_ptr<InputSink> stdOutSink_;
    std::shared_ptr<InputSink> stdErrSink_;

    std::unique_ptr<Task> task_;
};

/* EXECUTE */

AsyncRunner::RunResult Execute(const std::vector<std::string> & command,
                               const std::shared_ptr<InputSink> & stdOutSink
                               = nullptr,
                               const std::shared_ptr<InputSink> & stdErrSink
                               = nullptr,
                               const std::string & stdInData = "");

}

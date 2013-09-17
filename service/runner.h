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

#include "epoller.h"
#include "jml/utils/ring_buffer.h"
#include "sink.h"


namespace Datacratic {

/* RUNNER */

struct Runner: public Epoller {
    struct RunResult {
        RunResult()
        : signaled(false), returnCode(-1)
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

    Runner();
    ~Runner();

    OutputSink & getStdInSink();

    void run(const std::vector<std::string> & command,
             const OnTerminate & onTerminate = nullptr,
             const std::shared_ptr<InputSink> & stdOutSink = nullptr,
             const std::shared_ptr<InputSink> & stdErrSink = nullptr);
    void kill(int signal = SIGTERM) const;
    void waitStart() const;
    void waitTermination() const;

    bool running() const { return running_; }
    pid_t childPid() const { return childPid_; }

private:
    struct Task {
        enum StatusState {
            START,
            STOP,
            DONE
        };

        Task()
            : wrapperPid(-1),
              stdInFd(-1),
              stdOutFd(-1),
              stdErrFd(-1),
              statusFd(-1),
              statusState(DONE)
        {}

        void setupInSink();
        void flushInSink();
        void flushStdInBuffer();
        void postTerminate(Runner & runner);

        std::vector<std::string> command;
        OnTerminate onTerminate;
        RunResult runResult;

        pid_t wrapperPid;

        int stdInFd;
        int stdOutFd;
        int stdErrFd;
        int statusFd;
        StatusState statusState;
        std::string statusStateAsString() {
            if (statusState == START) {
                return "START";
            }
            else if (statusState == STOP) {
                return "STOP";
            }
            else if (statusState == DONE) {
                return "DONE";
            }
            else {
                throw ML::Exception("unknown status");
            }
        }
    };

    void prepareChild();
    bool handleEpollEvent(const struct epoll_event & event);
    void handleChildStatus(const struct epoll_event & event);
    void handleOutputStatus(const struct epoll_event & event,
                            int fd, std::shared_ptr<InputSink> & sink);

    void attemptTaskTermination();

    void closeStdInSink();

    int running_;
    pid_t childPid_;

    std::shared_ptr<AsyncFdOutputSink> stdInSink_;
    std::shared_ptr<InputSink> stdOutSink_;
    std::shared_ptr<InputSink> stdErrSink_;

    Task task_;
};


/* EXECUTE */

/* Execute a command synchronously using the specified message loop. */
Runner::RunResult execute(MessageLoop & loop,
                          const std::vector<std::string> & command,
                          const std::shared_ptr<InputSink> & stdOutSink
                          = nullptr,
                          const std::shared_ptr<InputSink> & stdErrSink
                          = nullptr,
                          const std::string & stdInData = "");

/* Execute a command synchronously using its own message loop. */
Runner::RunResult execute(const std::vector<std::string> & command,
                          const std::shared_ptr<InputSink> & stdOutSink
                          = nullptr,
                          const std::shared_ptr<InputSink> & stdErrSink
                          = nullptr,
                          const std::string & stdInData = "");

}

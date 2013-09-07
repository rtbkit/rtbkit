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
#include "typed_message_channel.h"


namespace Datacratic {

/* SINK */

/* Note: the receiving end is responsible for thread isolation. */
struct Sink {
    Sink()
        : closed_(false)
    {}

    virtual void write(std::string && data) = 0;

    void close(void)
    { closed_ = true; }

    bool closed() const
    { return closed_; }

private:
    bool closed_;
};

struct CallbackSink : Sink {
    typedef std::function<void(std::string && data)> OnData;

    CallbackSink(const OnData & onData)
        : Sink(), onData_(onData)
    {}

    virtual void write(std::string && data)
    { onData_(std::move(data)); }

private:
    OnData onData_;
};

struct NullSink : public Sink {
    virtual void write(std::string && data)
    {}
};

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

    Sink & getStdInSink();

    void run(const std::vector<std::string> & command,
             const OnTerminate & onTerminate = nullptr,
             const std::shared_ptr<Sink> & stdOutSink = nullptr,
             const std::shared_ptr<Sink> & stdErrSink = nullptr);
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
              stdInReady(false),
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

        std::unique_ptr<TypedMessageSink<std::string> > inSink;
        std::string buffer;

        pid_t childPid;
        pid_t wrapperPid;

        int stdInFd;
        bool stdInReady;
        int stdOutFd;
        int stdErrFd;
        int statusFd;
    };

    void prepareChild();
    bool handleEpollEvent(const struct epoll_event & event);
    void handleChildStatus(const struct epoll_event & event,
                           int fd, Task & task);
    void handleClientData(const struct epoll_event & event,
                          TypedMessageSink<std::string> & inSink);
    void handleStdInStatus(const struct epoll_event & event, int fd);
    void handleOutputStatus(const struct epoll_event & event,
                            int fd, Sink & inputSink);
    void postTerminate();

    int running_;

    std::unique_ptr<Sink> stdInSink_;
    std::shared_ptr<Sink> stdOutSink_;
    std::shared_ptr<Sink> stdErrSink_;

    std::unique_ptr<Task> task_;
};

/* EXECUTE */

AsyncRunner::RunResult Execute(const std::vector<std::string> & command,
                               const std::shared_ptr<Sink> & stdOutSink
                               = nullptr,
                               const std::shared_ptr<Sink> & stdErrSink
                               = nullptr,
                               const std::string & stdInData = "");

}

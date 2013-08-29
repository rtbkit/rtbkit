#pragma once

#include <signal.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "epoller.h"


/* disabled until signalfd is handled properly */
#define ASYNCRUNNER_SIGNALFD 0


namespace Datacratic {

struct AsyncRunner: public Epoller {
    // typedef std::function<void (int signal, int errorCode)> OnTerminate;
    typedef std::function<void (const std::string & newOutput)> OnOutput;
    typedef std::function<std::string (void)> OnInput;
    typedef std::pair<int, int> RunResult;

    AsyncRunner(const std::vector<std::string> & command);

#if ASYNCRUNNER_SIGNALFD
    void init(const OnTerminate & onTerminate = nullptr,
              const OnOutput & onStdOut = nullptr,
              const OnOutput & onStdErr = nullptr,
              const OnInput & onStdIn = nullptr);
#else
    void init(const OnOutput & onStdOut = nullptr,
              const OnOutput & onStdErr = nullptr,
              const OnInput & onStdIn = nullptr);
#endif

    void run();
    void kill(int signal = SIGTERM);
    void waitTermination();

    void prepareChild();

    void launch(MessageLoop & messageLoop);

    std::vector<std::string> command_;
#if ASYNCRUNNER_SIGNALFD
    OnTerminate onTerminate_;
#endif
    OnInput onStdIn_;
    OnOutput onStdOut_;
    OnOutput onStdErr_;

    bool running() const
    {
        return running_;
    }
    bool running_;

    int childPid() const
    {
        return childPid_;
    }
    pid_t childPid_;

    int lastSignal() const
    {
        return lastSignal_;
    }
    int lastSignal_;

    int lastReturnCode() const
    {
        return lastRc_;
    }
    int lastRc_;

    bool handleEpollEvent(const struct epoll_event & event);

#if ASYNCRUNNER_SIGNALFD
    void handleSigChild();
#endif

    void handleChildInput(const struct epoll_event & event);
    void handleChildOutput(const struct epoll_event & event,
                           const OnOutput & onOutputFn);

    void postTerminate();

#if ASYNCRUNNER_SIGNALFD
    int sigChildFd_;
#endif

    int stdInFd_;
    int stdOutFd_;
    int stdErrFd_;
};

}

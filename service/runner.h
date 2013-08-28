#pragma once

#include <signal.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "epoller.h"


namespace Datacratic {

struct AsyncRunner: public Epoller {
    typedef std::function<void (int signal, int errorCode)> OnTerminate;
    typedef std::function<void (const std::string & newOutput)> OnOutput;
    typedef std::function<std::string (void)> OnInput;
    typedef std::pair<int, int> RunResult;

    AsyncRunner(const std::vector<std::string> & command);

    void init(const OnTerminate & onTerminate = nullptr,
              const OnOutput & onStdOut = nullptr,
              const OnOutput & onStdErr = nullptr,
              const OnInput & onStdIn = nullptr);

    void run();
    void kill(int signal = SIGTERM);

    void prepareChild();
    // RunResult runSync();
    // void runWithMessageLoop(MessageLoop & messageLoop);
    // RunResult runWithMessageLoopSync(MessageLoop & messageLoop);

    void launch(MessageLoop & messageLoop);
    
    std::vector<std::string> command_;
    OnTerminate onTerminate_;
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
        return running_;
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

    void handleSigChild();
    void handleInput(const struct epoll_event & event,
                     const OnOutput & onOutputFn);
    void handleOutput(const struct epoll_event & event);

    void postTerminate();

    int sigChildFd_;
    int stdInFd_;
    int stdOutFd_;
    int stdErrFd_;
};

}

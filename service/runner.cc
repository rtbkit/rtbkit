
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/epoll.h>
#include <sys/signalfd.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/resource.h>

#include <utility>

#include "jml/arch/futex.h"

#include "message_loop.h"

#include "runner.h"

using namespace std;
using namespace Datacratic;

/* TODO:
   - interface without message loop
 */

namespace {

struct ChildFds {
    ChildFds()
        : stdIn_(::fileno(stdin)),
          stdOut_(::fileno(stdout)),
          stdErr_(::fileno(stderr))
    {
    }

    void closeRemainingFds()
    {
        struct rlimit limits;
        ::getrlimit(RLIMIT_NOFILE, &limits);

        for (int fd = 0; fd < limits.rlim_cur; fd++) {
            if (fd != STDIN_FILENO
                && fd != STDOUT_FILENO && fd != STDERR_FILENO) {
                ::close(fd);
            }
        }
    }

    void reSetupStream(int oldFd, int newFd)
    {
        if (oldFd != newFd) {
            int rc = ::dup2(oldFd, newFd);
            if (rc == -1) {
                throw ML::Exception(errno, "ChildFds::reSetupStream dup2");
            }
        }
    }

    void reSetupStdStreams()
    {
        reSetupStream(stdIn_, STDIN_FILENO);
        reSetupStream(stdOut_, STDOUT_FILENO);
        reSetupStream(stdErr_, STDERR_FILENO);
    }

    void close()
    {
        ::close(stdIn_);
        ::close(stdOut_);
        ::close(stdErr_);
    }

    int stdIn_;
    int stdOut_;
    int stdErr_;
};

void CreateStdPipe(int & parentFd, int & childFd, bool forWriting)
{
    int fds[2];
    int rc = pipe2(fds, O_NONBLOCK);
    if (rc == 0) {
        if (forWriting) {
            parentFd = fds[1];
            childFd = fds[0];
        }
        else {
            parentFd = fds[0];
            childFd = fds[1];
        }
    }
    else {
        throw ML::Exception(errno, "AsyncRunner::run:: pipe");
    }
}

}


AsyncRunner::
AsyncRunner(const std::vector<std::string> & command)
    : Epoller(),
      command_(command),
      running_(false),
      childPid_(-1),
      lastSignal_(-1),
      lastRc_(-1),
      sigChildFd_(-1),
      stdInFd_(-1),
      stdOutFd_(-1),
      stdErrFd_(-1)
{
}

void
AsyncRunner::
init(const OnTerminate & onTerminate,
     const OnOutput & onStdOut, const OnOutput & onStdErr,
     const OnInput & onStdIn)
{
    Epoller::init(4);

    handleEvent
        = bind(&AsyncRunner::handleEpollEvent, this, placeholders::_1);
    onTerminate_ = onTerminate;
    onStdIn_ = onStdIn;
    onStdOut_ = onStdOut;
    onStdErr_ = onStdErr;
}

void
AsyncRunner::
postTerminate()
{
    if (onTerminate_) {
        onTerminate_(lastSignal_, lastRc_);
    }

    running_ = false;
    childPid_ = -1;

    auto unregisterFd = [&] (int & fd)  {
        if (fd > -1) {
            ::close(fd);
            removeFd(fd);
            fd = -1;
        }
    };

    unregisterFd(sigChildFd_);
    unregisterFd(stdInFd_);
    unregisterFd(stdOutFd_);
    unregisterFd(stdErrFd_);
}

bool
AsyncRunner::
handleEpollEvent(const struct epoll_event & event)
{
    if (event.data.ptr == &sigChildFd_) {
        fprintf(stderr, "handle signal\n");
        handleSigChild();
    }
    else if (event.data.ptr == &stdInFd_) {
        handleOutput(event);
    }
    else if (event.data.ptr == &stdOutFd_) {
        handleInput(event, onStdOut_);
    }
    else if (event.data.ptr == &stdErrFd_) {
        handleInput(event, onStdErr_);
    }
    else {
        throw ML::Exception("this should never occur");
    }

    return false;
}

void
AsyncRunner::
handleSigChild()
{
    struct signalfd_siginfo siginfo;

    cerr << "handleSigChild\n";
    ssize_t s = ::read(sigChildFd_, &siginfo, sizeof(siginfo));
    if (s != sizeof(siginfo))
        throw ML::Exception(errno, "AsyncRunner::handleSigChild");

    if (siginfo.ssi_signo == SIGCHLD) {
        if (WIFEXITED(siginfo.ssi_status)) {
            lastSignal_ = 0;
            lastRc_ = WEXITSTATUS(siginfo.ssi_status);
            postTerminate();
        }
        else if (WIFSIGNALED(siginfo.ssi_status)) {
            lastSignal_ = WTERMSIG(siginfo.ssi_status);
            lastRc_ = 0;
            postTerminate();
        }
    }
    else {
        throw ML::Exception("AsyncRunner::handleSigChild: unexpected signal "
                            + to_string(siginfo.ssi_signo));
    }
}

void
AsyncRunner::
handleOutput(const struct epoll_event & event)
{
    string data = onStdIn_();

    const char *buffer = data.c_str();
    ssize_t remaining = data.size();
    size_t written(0);
    while (remaining > 0) {
        ssize_t len = ::write(stdInFd_, buffer + written, remaining);
        if (len > 0) {
            written += len;
            remaining -= len;
        }
        else {
            throw ML::Exception(errno, "AsyncRunner::handleOutput");
        }
    }

    if (written > 0) {
        restartFdOneShot(stdInFd_, event.data.ptr, true);
    }
}

void
AsyncRunner::
handleInput(const struct epoll_event & event, const OnOutput & onOutputFn)
{
    char buffer[4096];

    int inputFd = *static_cast<int *>(event.data.ptr);

    if ((event.events & EPOLLIN) != 0) {
        ssize_t len = ::read(inputFd, buffer, sizeof(buffer));
        if (len < 0) {
            if (errno != EWOULDBLOCK) {
                throw ML::Exception(errno, "AsyncRunner::handleInput");
            }
        }
        else if (len > 0) {
            string data(buffer, len);
            onOutputFn(data);
        }
    }

    if ((event.events & EPOLLHUP) == 0)
        restartFdOneShot(inputFd, event.data.ptr);
}

void
AsyncRunner::
run()
{
    if (running_)
        throw ML::Exception("already running");

    /* catch SIGCHLD */
    {
        sigset_t mask;

        sigemptyset(&mask);
        sigaddset(&mask, SIGCHLD);
        sigChildFd_ = signalfd(-1, &mask, SFD_NONBLOCK);
        addFd(sigChildFd_, &sigChildFd_);
    }

    ChildFds childFds;
    if (onStdIn_) {
        CreateStdPipe(stdInFd_, childFds.stdIn_, true);
        addFdOneShot(stdInFd_, &stdInFd_, true);
    }
    if (onStdOut_) {
        CreateStdPipe(stdOutFd_, childFds.stdOut_, false);
        addFdOneShot(stdOutFd_, &stdOutFd_);
    }
    if (onStdErr_) {
        CreateStdPipe(stdErrFd_, childFds.stdErr_, false);
        addFdOneShot(stdErrFd_, &stdErrFd_);
    }

    running_ = true;

    childPid_ = fork();
    if (childPid_ == 0) {
        childFds.reSetupStdStreams();
        childFds.closeRemainingFds();

        size_t len = command_.size();
        char * argv[len + 1];
        for (int i = 0; i < len; i++) {
            argv[i] = strdup(command_[i].c_str());
        }
        argv[len] = NULL;

        execv(command_[0].c_str(), argv);
    }
    else if (childPid_ == -1) {
        throw ML::Exception(errno, "AsyncRunner::run fork");
    }
    else {
        childFds.close();
    }
}

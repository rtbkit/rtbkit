
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

#include <iostream>
#include <utility>

#include "jml/arch/futex.h"
#include "jml/arch/timers.h"
#include "jml/utils/file_functions.h"

#include "message_loop.h"

#include "runner.h"

using namespace std;
using namespace Datacratic;

/* TODO:
   - signalfd can "compact" multiple and unrelated sigchilds, we must handle
     this in a thread-safe and lockless way
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
        if (stdIn_ != STDIN_FILENO) {
            cerr << "closing fd " << stdIn_ << endl;
            ::close(stdIn_);
        }
        if (stdOut_ != STDOUT_FILENO) {
            cerr << "closing fd " << stdOut_ << endl;
            ::close(stdOut_);
        }
        if (stdErr_ != STDERR_FILENO) {
            cerr << "closing fd " << stdErr_ << endl;
            ::close(stdErr_);
        }
    }

    int stdIn_;
    int stdOut_;
    int stdErr_;
};

pair<int, int> CreateStdPipe(bool forWriting)
{
    int fds[2];
    int rc = pipe(fds);
    if (rc == -1) {
        throw ML::Exception(errno, "AsyncRunner::run:: pipe2");
    }

    if (forWriting) {
        return pair<int, int>(fds[1], fds[0]);
    }
    else {
        return pair<int, int>(fds[0], fds[1]);
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
    onStdOut_ = onStdOut;
    onStdErr_ = onStdErr;
    onStdIn_ = onStdIn;
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
            removeFd(fd);
            ::close(fd);
            fd = -1;
        }
    };

    unregisterFd(stdInFd_);
    unregisterFd(stdOutFd_);
    unregisterFd(stdErrFd_);
    unregisterFd(sigChildFd_);
}

bool
AsyncRunner::
handleEpollEvent(const struct epoll_event & event)
{
    if (event.data.ptr == &sigChildFd_) {
        fprintf(stderr, "parent: handle signal\n");
        handleSigChild();
    }
    else if (event.data.ptr == &stdInFd_) {
        fprintf(stderr, "parent: handle child input\n");
        handleChildInput(event);
    }
    else if (event.data.ptr == &stdOutFd_) {
        fprintf(stderr, "parent: handle child output from stdout\n");
        handleChildOutput(event, onStdOut_);
    }
    else if (event.data.ptr == &stdErrFd_) {
        fprintf(stderr, "parent: handle child output from stderr\n");
        handleChildOutput(event, onStdErr_);
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
        if (siginfo.ssi_code == CLD_EXITED) {
            lastSignal_ = -1;
            lastRc_ = siginfo.ssi_status;
        }
        else {
            lastSignal_ = siginfo.ssi_status;
            lastRc_ = -1;
        }
        postTerminate();
    }
    else {
        throw ML::Exception("AsyncRunner::handleSigChild: unexpected signal "
                            + to_string(siginfo.ssi_signo));
    }
}

void
AsyncRunner::
handleChildInput(const struct epoll_event & event)
{
    size_t written(0);

    if ((event.events & EPOLLOUT) != 0) {
        string data = onStdIn_();

        const char *buffer = data.c_str();
        ssize_t remaining = data.size();
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

        if ((event.events & EPOLLHUP) == 0) {
            if (written > 0) {
                restartFdOneShot(stdInFd_, event.data.ptr, true);
            }
        }
        else {
            cerr << "parent: child stdin closed\n";
        }
    }
    else {
        if ((event.events & EPOLLHUP) == 0) {
            restartFdOneShot(stdInFd_, event.data.ptr, true);
        }
        else {
            cerr << "parent: child stdin closed\n";
        }
    }
}

void
AsyncRunner::
handleChildOutput(const struct epoll_event & event, const OnOutput & onOutputFn)
{
    char buffer[4096];
    string data;

    int inputFd = *static_cast<int *>(event.data.ptr);

    if ((event.events & EPOLLIN) != 0) {
        cerr << "parent: handling epollin from output" << endl;
        while (1) {
            ssize_t len = ::read(inputFd, buffer, sizeof(buffer));
            if (len < 0) {
                if (errno == EWOULDBLOCK) {
                    break;
                }
                else {
                    throw ML::Exception(errno, "AsyncRunner::handleInput");
                }
            }
            else if (len > 0) {
                data.append(buffer, len);
            }
            else {
                break;
            }
        }

        if (data.size() > 0) {
            cerr << "sending child output to output handler\n";
            onOutputFn(data);
        }
        else {
            cerr << "ignoring child output due to size == 0\n";
        }
    }

    if ((event.events & EPOLLHUP) == 0) {
        restartFdOneShot(inputFd, event.data.ptr);
    }
    else {
        cerr << "parent: child stdout or stderr closed\n";
    }
}

void
AsyncRunner::
run()
{
    if (running_)
        throw ML::Exception("already running");

    ChildFds childFds;
    if (onStdIn_) {
        auto fds = CreateStdPipe(true);
        stdInFd_ = fds.first;
        childFds.stdIn_ = fds.second; 
        cerr << "stdinFd_: " + to_string(stdInFd_) + "\n";
        cerr << "child stdinFd_: " + to_string(childFds.stdIn_) + "\n";
    }
    if (onStdOut_) {
        auto fds = CreateStdPipe(false);
        stdOutFd_ = fds.first;
        childFds.stdOut_ = fds.second; 
        cerr << "stdoutFd_: " + to_string(stdOutFd_) + "\n";
        cerr << "child stdoutFd_: " + to_string(childFds.stdOut_) + "\n";
    }
    if (onStdErr_) {
        auto fds = CreateStdPipe(false);
        stdErrFd_ = fds.first;
        childFds.stdErr_ = fds.second; 
        cerr << "stderrFd_: " + to_string(stdErrFd_) + "\n";
        cerr << "child stderrFd_: " + to_string(childFds.stdErr_) + "\n";
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
        /* catch SIGCHLD */
        {
            sigset_t mask;

            sigemptyset(&mask);
            sigaddset(&mask, SIGCHLD);
            int err = pthread_sigmask(SIG_BLOCK, &mask, NULL);
            if (err != 0) {
                 throw ML::Exception(err, "AsyncRunner::run pthread_sigmask");
            }
            sigChildFd_ = signalfd(-1, &mask, SFD_NONBLOCK);
            if (sigChildFd_ == -1) {
                throw ML::Exception(errno, "AsyncRunner::run signalfd");
            }
            addFd(sigChildFd_, &sigChildFd_);
            // cerr << "sigChildFd_: " + to_string(sigChildFd_) + "\n";
        }

        if (onStdIn_) {
            ML::set_file_flag(stdInFd_, O_NONBLOCK);
            addFdOneShot(stdInFd_, &stdInFd_, true);
        }
        if (onStdOut_) {
            ML::set_file_flag(stdOutFd_, O_NONBLOCK);
            addFdOneShot(stdOutFd_, &stdOutFd_);
        }
        if (onStdErr_) {
            ML::set_file_flag(stdErrFd_, O_NONBLOCK);
            addFdOneShot(stdErrFd_, &stdErrFd_);
        }
        printf ("parent: added fds to poll\n");

        childFds.close();
        printf ("parent: closed child ends\n");

        printf ("parent: fds setup\n");
    }
}


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
   - interface without external message loop
 */

namespace {

struct ChildStatus
{
    bool running;
    union {
        int pid;
        int status;
    };
};

tuple<int, int>
CreateStdPipe(bool forWriting)
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

struct ChildFds {
    ChildFds()
        : stdIn_(::fileno(stdin)),
          stdOut_(::fileno(stdout)),
          stdErr_(::fileno(stderr)),
          statusFd_(-1)
    {
    }

    /* child api */
    void closeRemainingFds()
    {
        struct rlimit limits;
        ::getrlimit(RLIMIT_NOFILE, &limits);

        for (int fd = 0; fd < limits.rlim_cur; fd++) {
            if (fd != STDIN_FILENO
                && fd != STDOUT_FILENO && fd != STDERR_FILENO
                && fd != statusFd_) {
                ::close(fd);
            }
        }
    }

    void dupToStdStreams()
    {
        auto dupToStdStream = [&] (int oldFd, int newFd) {
            if (oldFd != newFd) {
                int rc = ::dup2(oldFd, newFd);
                if (rc == -1) {
                    throw ML::Exception(errno,
                                        "ChildFds::dupToStdStream dup2");
                }
            }
        };
        dupToStdStream(stdIn_, STDIN_FILENO);
        dupToStdStream(stdOut_, STDOUT_FILENO);
        dupToStdStream(stdErr_, STDERR_FILENO);
    }

    /* parent & child api */
    void close()
    {
        auto closeIfNotEqual = [&] (int & fd, int notValue) {
            if (fd != notValue) {
                ::close(stdIn_);
            }
        };
        closeIfNotEqual(stdIn_, STDIN_FILENO);
        closeIfNotEqual(stdOut_, STDOUT_FILENO);
        closeIfNotEqual(stdErr_, STDERR_FILENO);
        closeIfNotEqual(statusFd_, -1);
    }

    int stdIn_;
    int stdOut_;
    int stdErr_;
    int statusFd_;
};

void
RunWrapper(const vector<string> & command, ChildFds & fds)
{
    fds.dupToStdStreams();
    fds.closeRemainingFds();

    int childPid = fork();
    if (childPid == -1) {
        throw ML::Exception(errno, "RunWrapper fork");
    }
    else if (childPid == 0) {
        size_t len = command.size();
        char * argv[len + 1];
        for (int i = 0; i < len; i++) {
            argv[i] = strdup(command[i].c_str());
        }
        argv[len] = NULL;
        execv(command[0].c_str(), argv);
        throw ML::Exception("The Alpha became the Omega.");
    }
    else {
        // FILE * terminal = ::fopen("/dev/tty", "a");

        // ::fprintf(terminal, "wrapper: real child pid: %d\n", childPid);
        ChildStatus status;

        status.running = true;
        status.pid = childPid;
        if (::write(fds.statusFd_, &status, sizeof(status)) == -1) {
            throw ML::Exception(errno, "RunWrapper write");
        }

        // ::fprintf(terminal, "wrapper: waiting child...\n");

        waitpid(childPid, &status.status, 0);

        // ::fprintf(terminal, "wrapper: child terminated\n");

        status.running = false;
        if (::write(fds.statusFd_, &status, sizeof(status)) == -1) {
            throw ML::Exception(errno, "RunWrapper write");
        }

        fds.close();

        exit(0);
    }
}

} // namespace

AsyncRunner::
AsyncRunner(const std::vector<std::string> & command)
    : Epoller(),
      command_(command),
      running_(false),
      childPid_(-1),
      stdInFd_(-1),
      stdOutFd_(-1),
      stdErrFd_(-1),
      statusFd_(-1)
{
}

void
AsyncRunner::
init(const OnTerminate & onTerminate,
     const OnOutput & onStdOut, const OnOutput & onStdErr,
     const OnInput & onStdIn)
{
    Epoller::init(4);

    handleEvent = bind(&AsyncRunner::handleEpollEvent, this,
                       placeholders::_1);
    onTerminate_ = onTerminate;
    onStdOut_ = onStdOut;
    onStdErr_ = onStdErr;
    onStdIn_ = onStdIn;
}

bool
AsyncRunner::
handleEpollEvent(const struct epoll_event & event)
{
    if (event.data.ptr == &statusFd_) {
        // fprintf(stderr, "parent: handle child status input\n");
        handleChildStatus(event);
    }
    else if (event.data.ptr == &stdInFd_) {
        // fprintf(stderr, "parent: handle child input to stdin\n");
        handleChildInput(event);
    }
    else if (event.data.ptr == &stdOutFd_) {
        // fprintf(stderr, "parent: handle child output from stdout\n");
        handleChildOutput(event, onStdOut_);
    }
    else if (event.data.ptr == &stdErrFd_) {
        // fprintf(stderr, "parent: handle child output from stderr\n");
        handleChildOutput(event, onStdErr_);
    }
    else {
        throw ML::Exception("this should never occur");
    }

    return false;
}

void
AsyncRunner::
handleChildStatus(const struct epoll_event & event)
{
    ChildStatus status;

    // cerr << "handleChildStatus\n";
    ssize_t s = ::read(statusFd_, &status, sizeof(status));
    if (s == -1) {
        throw ML::Exception(errno, "AsyncRunner::handleChildStatus read");
    }

    if (status.running) {
        childPid_ = status.pid;
        restartFdOneShot(statusFd_, event.data.ptr);
    }
    else {
        runResult_.updateFromStatus(status.status);
        postTerminate();
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
    bool closedFd(false);
    string data;

    int inputFd = *static_cast<int *>(event.data.ptr);

    if ((event.events & EPOLLIN) != 0) {
        while (1) {
            ssize_t len = ::read(inputFd, buffer, sizeof(buffer));
            // cerr << "returned len: " << len << endl;
            if (len < 0) {
                // perror("  len -1");
                if (errno == EWOULDBLOCK) {
                    break;
                }
                else if (errno == EBADF) {
                    closedFd = true;
                    break;
                }
                else {
                    throw ML::Exception(errno,
                                        "AsyncRunner::handleChildOutput read");
                }
            }
            else if (len == 0) {
                closedFd = true;
                break;
            }
            else if (len > 0) {
                data.append(buffer, len);
            }
        }

        if (data.size() > 0) {
            // cerr << "sending child output to output handler\n";
            onOutputFn(data);
        }
        else {
            cerr << "ignoring child output due to size == 0\n";
        }
    }

    // cerr << "closedFd: " << closedFd << endl;
    if (!closedFd && (event.events & EPOLLHUP) == 0) {
        JML_TRACE_EXCEPTIONS(false);
        try {
            restartFdOneShot(inputFd, event.data.ptr);
        }
        catch (const ML::Exception & exc) {
        }
    }
    else {
        cerr << "parent: child stdout or stderr closed\n";
    }
}

void
AsyncRunner::
postTerminate()
{
    // cerr << "postTerminate\n";

    waitpid(wrapperPid_, NULL, 0);
    wrapperPid_ = -1;

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
    unregisterFd(statusFd_);

    running_ = false;
    childPid_ = -1;

    if (onTerminate_) {
        onTerminate_(runResult_);
    }

    ML::futex_wake(running_);
}

void
AsyncRunner::
run()
{
    if (running_)
        throw ML::Exception("already running");

    ChildFds childFds;

    tie(statusFd_, childFds.statusFd_) = CreateStdPipe(false);
    if (onStdIn_) {
        tie(stdInFd_, childFds.stdIn_) = CreateStdPipe(true);
    }
    if (onStdOut_) {
        tie(stdOutFd_, childFds.stdOut_) = CreateStdPipe(false);
    }
    if (onStdErr_) {
        tie(stdErrFd_, childFds.stdErr_) = CreateStdPipe(false);
    }

    wrapperPid_ = fork();
    if (wrapperPid_ == -1) {
        throw ML::Exception(errno, "AsyncRunner::run fork");
    }
    else if (wrapperPid_ == 0) {
        RunWrapper(command_, childFds);
    }
    else {
        running_ = true;

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
        ML::set_file_flag(statusFd_, O_NONBLOCK);
        addFdOneShot(statusFd_, &statusFd_);

        childFds.close();
    }
}

void
AsyncRunner::
kill(int signum)
{
    if (!running_)
        throw ML::Exception("subprocess has already terminated");

    ::kill(childPid_, signum);
    waitTermination();
}

void
AsyncRunner::
waitTermination()
{
    if (!running_)
        throw ML::Exception("subprocess has already terminated");
    
    while (running_) {
        ML::futex_wait(running_, true);
    }
}

void
AsyncRunner::
RunResult::
updateFromStatus(int status)
{
    if (WIFEXITED(status)) {
        signaled = false;
        returnCode = WEXITSTATUS(status);
    }
    else if (WIFSIGNALED(status)) {
        signaled = true;
        signum = WTERMSIG(status);
    }
}

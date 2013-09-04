
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
        throw ML::Exception(errno, "CreateStdPipe pipe2");
    }

    if (forWriting) {
        return tuple<int, int>(fds[1], fds[0]);
    }
    else {
        return tuple<int, int>(fds[0], fds[1]);
    }
}

struct ChildFds {
    ChildFds()
        : stdIn(::fileno(stdin)),
          stdOut(::fileno(stdout)),
          stdErr(::fileno(stderr)),
          statusFd(-1)
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
                && fd != statusFd) {
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
        dupToStdStream(stdIn, STDIN_FILENO);
        dupToStdStream(stdOut, STDOUT_FILENO);
        dupToStdStream(stdErr, STDERR_FILENO);
    }

    /* parent & child api */
    void close()
    {
        auto closeIfNotEqual = [&] (int & fd, int notValue) {
            if (fd != notValue) {
                ::close(stdIn);
            }
        };
        closeIfNotEqual(stdIn, STDIN_FILENO);
        closeIfNotEqual(stdOut, STDOUT_FILENO);
        closeIfNotEqual(stdErr, STDERR_FILENO);
        closeIfNotEqual(statusFd, -1);
    }

    int stdIn;
    int stdOut;
    int stdErr;
    int statusFd;
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
        ::close(fds.statusFd);
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
        if (::write(fds.statusFd, &status, sizeof(status)) == -1) {
            throw ML::Exception(errno, "RunWrapper write");
        }

        // ::fprintf(terminal, "wrapper: waiting child...\n");

        waitpid(childPid, &status.status, 0);

        // ::fprintf(terminal, "wrapper: child terminated\n");

        status.running = false;
        if (::write(fds.statusFd, &status, sizeof(status)) == -1) {
            throw ML::Exception(errno, "RunWrapper write");
        }

        fds.close();

        exit(0);
    }
}

} // namespace

/* ASYNCRUNNER */

AsyncRunner::
AsyncRunner()
    : running_(false)
{
    Epoller::init(4);

    handleEvent = bind(&AsyncRunner::handleEpollEvent, this,
                       placeholders::_1);
}

bool
AsyncRunner::
handleEpollEvent(const struct epoll_event & event)
{
    if (task_) {
        Task & task = *task_;
        if (event.data.ptr == &task.statusFd) {
            // fprintf(stderr, "parent: handle child status input\n");
            handleChildStatus(event, task.statusFd, task);
        }
        else if (event.data.ptr == &task.stdInFd) {
            // fprintf(stderr, "parent: handle child input to stdin\n");
            handleChildInput(event, task.stdInFd, task.onStdIn);
        }
        else if (event.data.ptr == &task.stdOutFd) {
            // fprintf(stderr, "parent: handle child output from stdout\n");
            handleChildOutput(event, task.stdOutFd, task.onStdOut);
        }
        else if (event.data.ptr == &task.stdErrFd) {
            // fprintf(stderr, "parent: handle child output from stderr\n");
            handleChildOutput(event, task.stdErrFd, task.onStdErr);
        }
        else {
            throw ML::Exception("this should never occur");
        }
    }
    else {
        throw ML::Exception("received event for ghost task");
    }

    return false;
}

void
AsyncRunner::
handleChildStatus(const struct epoll_event & event, int statusFd, Task & task)
{
    ChildStatus status;

    // cerr << "handleChildStatus\n";
    ssize_t s = ::read(statusFd, &status, sizeof(status));
    if (s == -1) {
        throw ML::Exception(errno, "AsyncRunner::handleChildStatus read");
    }

    if (status.running) {
        task.childPid = status.pid;
        restartFdOneShot(statusFd, event.data.ptr);
    }
    else {
        RunResult result(status.status);
        task.postTerminate(*this, result);
        task_.reset();
        running_ = false;
        ML::futex_wake(running_);
    }
}

void
AsyncRunner::
handleChildInput(const struct epoll_event & event,
                 int stdInFd, const OnInput & onInputFn)
{
    size_t written(0);

    if ((event.events & EPOLLOUT) != 0) {
        string data = onInputFn();

        const char *buffer = data.c_str();
        ssize_t remaining = data.size();
        while (remaining > 0) {
            ssize_t len = ::write(stdInFd, buffer + written, remaining);
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
                restartFdOneShot(stdInFd, event.data.ptr, true);
            }
        }
        else {
            cerr << "parent: child stdin closed\n";
        }
    }
    else {
        if ((event.events & EPOLLHUP) == 0) {
            restartFdOneShot(stdInFd, event.data.ptr, true);
        }
        else {
            cerr << "parent: child stdin closed\n";
        }
    }
}

void
AsyncRunner::
handleChildOutput(const struct epoll_event & event,
                  int outputFd, const OnOutput & onOutputFn)
{
    char buffer[4096];
    bool closedFd(false);
    string data;

    if ((event.events & EPOLLIN) != 0) {
        while (1) {
            ssize_t len = ::read(outputFd, buffer, sizeof(buffer));
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
            restartFdOneShot(outputFd, event.data.ptr);
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
run(const std::vector<std::string> & command,
    const OnTerminate & onTerminate,
    const OnOutput & onStdOut,
    const OnOutput & onStdErr,
    const OnInput & onStdIn)
{
    if (task_)
        throw ML::Exception("already running");

    running_ = true;
    task_.reset(new Task());
    Task & task = *task_;

    task.onTerminate = onTerminate;

    ChildFds childFds;
    tie(task.statusFd, childFds.statusFd) = CreateStdPipe(false);
    if (onStdIn) {
        task.onStdIn = onStdIn;
        tie(task.stdInFd, childFds.stdIn) = CreateStdPipe(true);
    }
    if (onStdOut) {
        task.onStdOut = onStdOut;
        tie(task.stdOutFd, childFds.stdOut) = CreateStdPipe(false);
    }
    if (onStdErr) {
        task.onStdErr = onStdErr;
        tie(task.stdErrFd, childFds.stdErr) = CreateStdPipe(false);
    }

    task.wrapperPid = fork();
    if (task.wrapperPid == -1) {
        throw ML::Exception(errno, "AsyncRunner::run fork");
    }
    else if (task.wrapperPid == 0) {
        RunWrapper(command, childFds);
    }
    else {
        if (onStdIn) {
            ML::set_file_flag(task.stdInFd, O_NONBLOCK);
            addFdOneShot(task.stdInFd, &task.stdInFd, true);
        }
        if (onStdOut) {
            ML::set_file_flag(task.stdOutFd, O_NONBLOCK);
            addFdOneShot(task.stdOutFd, &task.stdOutFd);
        }
        if (onStdErr) {
            ML::set_file_flag(task.stdErrFd, O_NONBLOCK);
            addFdOneShot(task.stdErrFd, &task.stdErrFd);
        }
        ML::set_file_flag(task.statusFd, O_NONBLOCK);
        addFdOneShot(task.statusFd, &task.statusFd);

        childFds.close();
    }
}

void
AsyncRunner::
kill(int signum)
{
    if (!task_)
        throw ML::Exception("subprocess has already terminated");

    ::kill(task_->childPid, signum);
    waitTermination();
}

void
AsyncRunner::
waitTermination()
{
    while (running_) {
        ML::futex_wait(running_, true);
    }
}

/* ASYNCRUNNER::TASK */

void
AsyncRunner::
Task::
postTerminate(AsyncRunner & runner, const RunResult & runResult)
{
    // cerr << "postTerminate\n";

    waitpid(wrapperPid, NULL, 0);

    auto unregisterFd = [&] (int & fd)  {
        if (fd > -1) {
            runner.removeFd(fd);
            ::close(fd);
            fd = -1;
        }
    };
    unregisterFd(stdInFd);
    unregisterFd(stdOutFd);
    unregisterFd(stdErrFd);
    unregisterFd(statusFd);

    if (onTerminate) {
        onTerminate(runResult);
    }
}

/* ASYNCRUNNER::RUNRESULT */

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

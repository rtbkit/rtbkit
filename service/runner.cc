/* runner.cc                                                       -*- C++ -*-
   Wolfgang Sourdeau, September 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.

   A command runner class that hides the specifics of the underlying unix
   system calls and can intercept input and output.
*/

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/epoll.h>
#include <sys/prctl.h>
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
#include "sink.h"

#include "runner.h"

using namespace std;
using namespace Datacratic;


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
                ::close(fd);
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

    // Set up the arguments before we fork, as we don't want to call malloc()
    // from the fork, and it can be called from c_str() in theory.
    size_t len = command.size();
    char * argv[len + 1];
    for (int i = 0; i < len; i++) {
        argv[i] = (char *) command[i].c_str();
    }
    argv[len] = NULL;

    int childPid = fork();
    if (childPid == -1) {
        throw ML::Exception(errno, "RunWrapper fork");
    }
    else if (childPid == 0) {
        signal(SIGCHLD, SIG_DFL);
        signal(SIGPIPE, SIG_DFL);

        ::prctl(PR_SET_PDEATHSIG, SIGTERM);
        ::close(fds.statusFd);
        int res = execv(command[0].c_str(), argv);
        if (res == -1) {
            throw ML::Exception(errno, "RunWrapper exec");
        }

        /* there is no possible way this code could be executed */
        throw ML::Exception("The Alpha became the Omega.");
    }
    else {
        // Undo any SIGCHLD block from the parent process so it can
        // properly wait for the signal
        signal(SIGCHLD, SIG_DFL);
        signal(SIGPIPE, SIG_DFL);

        // FILE * terminal = ::fopen("/dev/tty", "a");

        // ::fprintf(terminal, "wrapper: real child pid: %d\n", childPid);
        ChildStatus status;

        status.running = true;
        status.pid = childPid;
        if (::write(fds.statusFd, &status, sizeof(status)) == -1) {
            throw ML::Exception(errno, "RunWrapper write");
        }

        // ::fprintf(terminal, "wrapper: waiting child...\n");

        int res = ::waitpid(childPid, &status.status, 0);
        if (res == -1)
            throw ML::Exception(errno, "waitpid");
        if (res != childPid)
            throw ML::Exception("waitpid has not returned the childPid");

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

Runner::
Runner()
    : wakeup_(EFD_NONBLOCK | EFD_CLOEXEC),
      running_(false)
{
    Epoller::init(4);

    addFdOneShot(wakeup_.fd(), &wakeup_.fd_);

    handleEvent = [&] (const struct epoll_event & event) {
        return this->handleEpollEvent(event);
    };
}

Runner::
~Runner()
{
    waitTermination();
}

bool
Runner::
handleEpollEvent(const struct epoll_event & event)
{
    if (task_) {
        Task & task = *task_;
        if (event.data.ptr == &task.statusFd) {
            // fprintf(stderr, "parent: handle child status input\n");
            handleChildStatus(event, task.statusFd, task);
        }
        else if (event.data.ptr == &task.stdOutFd) {
            // fprintf(stderr, "parent: handle child output from stdout\n");
            handleOutputStatus(event, task.stdOutFd, *stdOutSink_);
        }
        else if (event.data.ptr == &task.stdErrFd) {
            // fprintf(stderr, "parent: handle child output from stderr\n");
            handleOutputStatus(event, task.stdErrFd, *stdErrSink_);
        }
        else if (event.data.ptr == &wakeup_.fd_) {
            handleTaskTermination(event);
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
Runner::
handleChildStatus(const struct epoll_event & event, int statusFd, Task & task)
{
    ChildStatus status;

    // cerr << "handleChildStatus\n";
    ssize_t s = ::read(statusFd, &status, sizeof(status));
    if (s == -1) {
        throw ML::Exception(errno, "Runner::handleChildStatus read");
    }

    if (status.running) {
        task.childPid = status.pid;
        restartFdOneShot(statusFd, event.data.ptr);
    }
    else {
        if (stdInSink_) {
            closeStdInSink();
        }
        task.runResult.updateFromStatus(status.status);
        wakeup_.signal();
    }
}

void
Runner::
handleOutputStatus(const struct epoll_event & event,
                   int outputFd, InputSink & sink)
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
                                        "Runner::handleOutputStatus read");
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
            sink.notifyReceived(move(data));
        }
        else {
            cerr << "ignoring child output due to size == 0\n";
        }
    }

    // cerr << "closedFd: " << closedFd << endl;
    if (closedFd || (event.events & EPOLLHUP) == 0) {
        sink.notifyClosed();
    }
    else {
        JML_TRACE_EXCEPTIONS(false);
        try {
            restartFdOneShot(outputFd, event.data.ptr);
        }
        catch (const ML::Exception & exc) {
            sink.notifyClosed();
        }
    }
}

void
Runner::
handleTaskTermination(const struct epoll_event & event)
{
    wakeup_.read();

    if (stdInSink_) {
        closeStdInSink();
        parent_->removeSource(stdInSink_.get());
        stdInSink_.reset();
    }
    if (stdOutSink_) {
        stdOutSink_->notifyClosed();
        stdOutSink_.reset();
    }
    if (stdErrSink_) {
        stdErrSink_->notifyClosed();
        stdErrSink_.reset();
    }

    task_->postTerminate(*this);
    task_.reset(nullptr);

    restartFdOneShot(wakeup_.fd(), event.data.ptr);

    running_ = false;
    ML::futex_wake(running_);
}

OutputSink &
Runner::
getStdInSink()
{
    if (running_)
        throw ML::Exception("already running");
    if (stdInSink_)
        throw ML::Exception("stdin sink already set");

    auto onHangup = [&] () {
        this->closeStdInSink();
    };
    auto onClose = [&] () {
        this->closeStdInSink();
    };
    stdInSink_.reset(new AsyncFdOutputSink(onHangup, onClose));

    return *stdInSink_;
}

void
Runner::
run(const vector<string> & command,
    const OnTerminate & onTerminate,
    const shared_ptr<InputSink> & stdOutSink,
    const shared_ptr<InputSink> & stdErrSink)
{
    if (task_)
        throw ML::Exception("already running");

    running_ = true;
    task_.reset(new Task());
    Task & task = *task_;

    task.onTerminate = onTerminate;

    ChildFds childFds;
    tie(task.statusFd, childFds.statusFd) = CreateStdPipe(false);

    if (stdInSink_) {
        tie(task.stdInFd, childFds.stdIn) = CreateStdPipe(true);
    }
    if (stdOutSink) {
        stdOutSink_ = stdOutSink;
        tie(task.stdOutFd, childFds.stdOut) = CreateStdPipe(false);
    }
    if (stdErrSink) {
        stdErrSink_ = stdErrSink;
        tie(task.stdErrFd, childFds.stdErr) = CreateStdPipe(false);
    }

    ::flockfile(stdout);
    ::flockfile(stderr);
    ::fflush(NULL);
    task.wrapperPid = fork();
    if (task.wrapperPid == -1) {
        ::funlockfile(stderr);
        ::funlockfile(stdout);
        throw ML::Exception(errno, "Runner::run fork");
    }
    else if (task.wrapperPid == 0) {
        ::funlockfile(stderr);
        ::funlockfile(stdout);
        RunWrapper(command, childFds);
    }
    else {
        ::funlockfile(stderr);
        ::funlockfile(stdout);
        ML::set_file_flag(task.statusFd, O_NONBLOCK);
        if (stdInSink_) {
            ML::set_file_flag(task.stdInFd, O_NONBLOCK);
            stdInSink_->init(task.stdInFd);
            parent_->addSource("stdInSink", stdInSink_);
        }
        addFdOneShot(task.statusFd, &task.statusFd);
        if (stdOutSink) {
            ML::set_file_flag(task.stdOutFd, O_NONBLOCK);
            addFdOneShot(task.stdOutFd, &task.stdOutFd);
        }
        if (stdErrSink) {
            ML::set_file_flag(task.stdErrFd, O_NONBLOCK);
            addFdOneShot(task.stdErrFd, &task.stdErrFd);
        }

        childFds.close();
    }
}

void
Runner::
kill(int signum)
{
    if (!task_)
        throw ML::Exception("subprocess has already terminated");

    ::kill(task_->childPid, signum);
    waitTermination();
}

void
Runner::
waitTermination()
{
    while (running_) {
        ML::futex_wait(running_, true);
    }
}

void
Runner::
closeStdInSink()
{
    if (task_) {
        if (task_->stdInFd != -1) {
            ::close(task_->stdInFd);
            task_->stdInFd = -1;
        }
    }
    else {
        throw ML::Exception("task dead but stdin sink alive\n");
    }

    stdInSink_->state = OutputSink::CLOSED;
    parent_->removeSource(stdInSink_.get());
    stdInSink_.reset();
}


/* ASYNCRUNNER::TASK */

void
Runner::
Task::
postTerminate(Runner & runner)
{
    // cerr << "postTerminate\n";

    int res = ::waitpid(wrapperPid, NULL, 0);
    if (res == -1)
        throw ML::Exception(errno, "waitpid");
    if (res != wrapperPid)
        throw ML::Exception("waitpid has not returned the wrappedPid");

    if (stdInFd != -1) {
        ::close(stdInFd);
        stdInFd = -1;
    }

    auto unregisterFd = [&] (int & fd) {
        if (fd > -1) {
            runner.removeFd(fd);
            ::close(fd);
            fd = -1;
        }
    };
    unregisterFd(stdOutFd);
    unregisterFd(stdErrFd);
    unregisterFd(statusFd);

    if (onTerminate) {
        onTerminate(runResult);
    }
}


/* ASYNCRUNNER::RUNRESULT */

void
Runner::
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


/* EXECUTE */

Runner::RunResult
Datacratic::
execute(MessageLoop & loop,
        const vector<string> & command,
        const shared_ptr<InputSink> & stdOutSink,
        const shared_ptr<InputSink> & stdErrSink,
        const string & stdInData)
{
    Runner::RunResult result;
    auto onTerminate = [&](const Runner::RunResult & runResult) {
        result = runResult;
    };

    Runner runner;

    loop.addSource("runner", runner);
    loop.start();

    if (stdInData.size() > 0) {
        auto & sink = runner.getStdInSink();
        runner.run(command, onTerminate, stdOutSink, stdErrSink);
        sink.write(stdInData);
        sink.requestClose();
    }
    else {
        runner.run(command, onTerminate, stdOutSink, stdErrSink);
    }

    runner.waitTermination();
    loop.removeSource(&runner);

    return result;
}

Runner::RunResult
Datacratic::
execute(const vector<string> & command,
        const shared_ptr<InputSink> & stdOutSink,
        const shared_ptr<InputSink> & stdErrSink,
        const string & stdInData)
{
    MessageLoop loop;

    Runner::RunResult result = execute(loop, command, stdOutSink, stdErrSink,
                                       stdInData);

    loop.shutdown();

    return result;
}

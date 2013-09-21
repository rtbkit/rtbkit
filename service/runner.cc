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

} // namespace


/* ASYNCRUNNER */

Runner::
Runner()
    : running_(false), childPid_(-1),
      statusRemaining_(sizeof(Task::ChildStatus))

{
    Epoller::init(4);

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
    if (event.data.ptr == &task_.statusFd) {
        // fprintf(stderr, "parent: handle child status input\n");
        handleChildStatus(event);
    }
    else if (event.data.ptr == &task_.stdOutFd) {
        // fprintf(stderr, "parent: handle child output from stdout\n");
        handleOutputStatus(event, task_.stdOutFd, stdOutSink_);
    }
    else if (event.data.ptr == &task_.stdErrFd) {
        // fprintf(stderr, "parent: handle child output from stderr\n");
        handleOutputStatus(event, task_.stdErrFd, stdErrSink_);
    }
    else {
        throw ML::Exception("this should never occur");
    }

    return false;
}

void
Runner::
handleChildStatus(const struct epoll_event & event)
{
    // cerr << "handleChildStatus\n";
    Task::ChildStatus status;

    if ((event.events & EPOLLIN) != 0) {
        while (1) {
            char * current = (statusBuffer_ + sizeof(Task::ChildStatus)
                              - statusRemaining_);
            ssize_t s = ::read(task_.statusFd, current, statusRemaining_);
            if (s == -1) {
                if (errno == EWOULDBLOCK) {
                    break;
                }
                else if (errno == EBADF || errno == EINVAL) {
                    // cerr << "badf\n";
                    break;
                }
                throw ML::Exception(errno, "Runner::handleChildStatus read");
            }
            else if (s == 0) {
                break;
            }

            statusRemaining_ -= s;
            if (statusRemaining_ > 0) {
                cerr << "warning: reading status fd in multiple chunks\n";
            }
            else if (statusRemaining_ == 0) {
                memcpy(&status, statusBuffer_, sizeof(status));
                if (task_.statusState == Task::StatusState::START) {
                    childPid_ = status.pid;
                    ML::futex_wake(childPid_);
                    // cerr << "child now running: " + to_string(childPid_) + "\n";
                    task_.statusState = Task::StatusState::STOP;
                }
                else if (task_.statusState == Task::StatusState::STOP) {
                    // cerr << "child now stopped: " + to_string(childPid_) + "\n";
                    childPid_ = -1;
                    task_.runResult.updateFromStatus(status.status);
                    task_.statusState = Task::StatusState::DONE;
                    attemptTaskTermination();
                }
                else {
                    throw ML::Exception("unexpected status when DONE");
                }
                statusRemaining_ = sizeof(statusBuffer_);
            }
        }
    }

    if ((event.events & EPOLLHUP) != 0) {
        if (task_.statusState != Task::StatusState::DONE) {
            throw ML::Exception("statusfd hup but not done: "
                                + task_.statusStateAsString());
        }

        removeFd(task_.statusFd);
        ::close(task_.statusFd);
        task_.statusFd = -1;
    }
    else {
        restartFdOneShot(task_.statusFd, event.data.ptr);
    }

    // cerr << "handleChildStatus done\n";
}

void
Runner::
handleOutputStatus(const struct epoll_event & event,
                   int outputFd, shared_ptr<InputSink> & sink)
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
                else if (errno == EBADF || errno == EINVAL) {
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
            sink->notifyReceived(move(data));
        }
        else {
            cerr << "ignoring child output due to size == 0\n";
        }
    }

    if (closedFd || (event.events & EPOLLHUP) != 0) {
        sink->notifyClosed();
        sink.reset();
        attemptTaskTermination();
    }
    else {
        JML_TRACE_EXCEPTIONS(false);
        try {
            restartFdOneShot(outputFd, event.data.ptr);
        }
        catch (const ML::Exception & exc) {
            cerr << "closing sink due to bad fd\n";
            sink->notifyClosed();
            sink.reset(); 
            attemptTaskTermination();
        }
    }
}

void
Runner::
attemptTaskTermination()
{
    /* for a task to be considered done:
       - stdout and stderr must have been closed, provided we redirected them
       - the closing child status must have been returned */
    if (!stdOutSink_ && !stdErrSink_ && childPid_ == -1) {
        // cerr << to_string(childPid()) + ": really terminated\n";
        if (stdInSink_) {
            closeStdInSink();
        }

        task_.postTerminate(*this);

        running_ = false;
        ML::futex_wake(running_);
    }
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
    if (running_)
        throw ML::Exception("already running");

    running_ = true;

    task_.statusState = Task::StatusState::START;
    task_.onTerminate = onTerminate;

    Task::ChildFds childFds;
    tie(task_.statusFd, childFds.statusFd) = CreateStdPipe(false);

    if (stdInSink_) {
        tie(task_.stdInFd, childFds.stdIn) = CreateStdPipe(true);
    }
    if (stdOutSink) {
        stdOutSink_ = stdOutSink;
        tie(task_.stdOutFd, childFds.stdOut) = CreateStdPipe(false);
    }
    if (stdErrSink) {
        stdErrSink_ = stdErrSink;
        tie(task_.stdErrFd, childFds.stdErr) = CreateStdPipe(false);
    }

    ::flockfile(stdout);
    ::flockfile(stderr);
    ::fflush_unlocked(NULL);
    task_.wrapperPid = fork();
    ::funlockfile(stderr);
    ::funlockfile(stdout);
    if (task_.wrapperPid == -1) {
        throw ML::Exception(errno, "Runner::run fork");
    }
    else if (task_.wrapperPid == 0) {
        task_.RunWrapper(command, childFds);
    }
    else {
        ML::set_file_flag(task_.statusFd, O_NONBLOCK);
        if (stdInSink_) {
            ML::set_file_flag(task_.stdInFd, O_NONBLOCK);
            stdInSink_->init(task_.stdInFd);
            parent_->addSource("stdInSink", stdInSink_);
        }
        addFdOneShot(task_.statusFd, &task_.statusFd);
        if (stdOutSink) {
            ML::set_file_flag(task_.stdOutFd, O_NONBLOCK);
            addFdOneShot(task_.stdOutFd, &task_.stdOutFd);
        }
        if (stdErrSink) {
            ML::set_file_flag(task_.stdErrFd, O_NONBLOCK);
            addFdOneShot(task_.stdErrFd, &task_.stdErrFd);
        }

        childFds.close();
    }
}

void
Runner::
kill(int signum)
    const
{
    if (!childPid_)
        throw ML::Exception("subprocess not available");

    ::kill(childPid_, signum);
    waitTermination();
}

void
Runner::
waitStart()
    const
{
    while (childPid_ == -1) {
        ML::futex_wait(childPid_, -1);
    }
}

void
Runner::
waitTermination()
    const
{
    while (running_) {
        ML::futex_wait(running_, true);
    }
}

void
Runner::
closeStdInSink()
{
    if (task_.stdInFd != -1) {
        ::close(task_.stdInFd);
        task_.stdInFd = -1;
    }

    stdInSink_->state = OutputSink::CLOSED;
    parent_->removeSource(stdInSink_.get());
    stdInSink_.reset();
}


/* ASYNCRUNNER::TASK */

void
Runner::
Task::
RunWrapper(const vector<string> & command, ChildFds & fds)
{
    // Undo any SIGCHLD block from the parent process so it can
    // properly wait for the signal
    ::signal(SIGCHLD, SIG_DFL);
    ::signal(SIGPIPE, SIG_DFL);

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
        ::signal(SIGQUIT, SIG_DFL);
        ::signal(SIGTERM, SIG_DFL);
        ::signal(SIGINT, SIG_DFL);

        ::prctl(PR_SET_PDEATHSIG, SIGTERM);
        ::close(fds.statusFd);
        int res = ::execv(command[0].c_str(), argv);
        if (res == -1) {
            throw ML::Exception(errno, "RunWrapper exec");
        }

        /* there is no possible way this code could be executed */
        throw ML::Exception("The Alpha became the Omega.");
    }
    else {
        // FILE * terminal = ::fopen("/dev/tty", "a");

        // ::fprintf(terminal, "wrapper: real child pid: %d\n", childPid);
        ChildStatus status;

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

        if (::write(fds.statusFd, &status, sizeof(status)) == -1) {
            throw ML::Exception(errno, "RunWrapper write");
        }

        fds.close();

        ::_exit(0);
    }
}

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
    wrapperPid = -1;

    if (stdInFd != -1) {
        ::close(stdInFd);
        stdInFd = -1;
    }

    auto unregisterFd = [&] (int & fd) {
        if (fd > -1) {
            JML_TRACE_EXCEPTIONS(false);
            try {
                runner.removeFd(fd);
            }
            catch (const ML::Exception & exc) {
            }
            ::close(fd);
            fd = -1;
        }
    };
    unregisterFd(stdOutFd);
    unregisterFd(stdErrFd);

    command.clear();

    if (onTerminate) {
        onTerminate(runResult);
        onTerminate = nullptr;
    }
    runResult.signaled = false;
    runResult.returnCode = -1;
}


/* CHILD::CHILDFDS */

Runner::
Task::
ChildFds::
ChildFds()
    : stdIn(::fileno(stdin)),
      stdOut(::fileno(stdout)),
      stdErr(::fileno(stderr)),
      statusFd(-1)
{
}

/* child api */
void
Runner::
Task::
ChildFds::
closeRemainingFds()
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

void
Runner::
Task::
ChildFds::
dupToStdStreams()
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
void
Runner::
Task::
ChildFds::
close()
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
    ML::sleep(0.5);

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

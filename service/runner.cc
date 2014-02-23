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

const string stdbufCmd = "/usr/bin/stdbuf";

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

namespace Datacratic {

/* ASYNCRUNNER */

Runner::
Runner()
    : running_(false), childPid_(-1), wakeup_(EFD_NONBLOCK),
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
    else if (event.data.ptr == nullptr) {
        // stdInSink cleanup for now...
        handleWakeup(event);
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
            else if (statusRemaining_ != 0)
                continue;

            memcpy(&status, statusBuffer_, sizeof(status));

            // Set up for next message
            statusRemaining_ = sizeof(statusBuffer_);

#if 0
            cerr << "got status " << status.state
                 << " " << Task::statusStateAsString(status.state)
                 << " " << status.pid << " " << status.childStatus
                 << " " << status.launchErrno << " "
                 << strerror(status.launchErrno) << " "
                 << Task::strLaunchError(status.launchErrorCode)
                 << endl;
#endif

            task_.statusState = status.state;

            if (status.launchErrno
                || status.launchErrorCode) {
                //cerr << "*** launch error" << endl;
                // Error
                childPid_ = -1;
                task_.runResult.updateFromLaunchError
                    (status.launchErrno,
                     Task::strLaunchError(status.launchErrorCode));
                attemptTaskTermination();
                break;
            }

            switch (status.state) {
            case Task::LAUNCHING:
                childPid_ = status.pid;
                break;
            case Task::RUNNING:
                childPid_ = status.pid;
                ML::futex_wake(childPid_);
                break;
            case Task::STOPPED:
                childPid_ = -1;
                task_.runResult.updateFromStatus(status.childStatus);
                task_.statusState = Task::StatusState::DONE;
                if (stdInSink_) {
                    stdInSink_->requestClose();
                }
                attemptTaskTermination();
                break;
            case Task::DONE:
                throw ML::Exception("unexpected status DONE");
            case Task::ST_UNKNOWN:
                throw ML::Exception("unexpected status UNKNOWN");
            }
        }
    }

    if ((event.events & EPOLLHUP) != 0) {
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
handleWakeup(const struct epoll_event & event)
{
    if ((event.events & EPOLLIN) != 0) {
        if (stdInSink_) {
            if (stdInSink_->connectionState_
                == AsyncEventSource::DISCONNECTED) {
                stdInSink_.reset();
                attemptTaskTermination();
            }
        }
        while (!wakeup_.tryRead());
        removeFd(wakeup_.fd());
    }
}

void
Runner::
attemptTaskTermination()
{
    /* for a task to be considered done:
       - stdout and stderr must have been closed, provided we redirected them
       - the closing child status must have been returned */
    if (!stdInSink_ && !stdOutSink_ && !stdErrSink_ && childPid_ == -1) {
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

    auto onClose = [&] () {
        if (task_.stdInFd != -1) {
            ::close(task_.stdInFd);
            task_.stdInFd = -1;
        }
        parent_->removeSource(stdInSink_.get());
        wakeup_.signal();
    };
    stdInSink_.reset(new AsyncFdOutputSink(onClose, onClose));

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

    task_.statusState = Task::StatusState::ST_UNKNOWN;
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
            addFdOneShot(wakeup_.fd());
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

    ::kill(-childPid_, signum);
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

    vector<string> stdbufCommand; //{stdbufCmd, "-o0"};
    stdbufCommand.insert(stdbufCommand.end(), command.begin(), command.end());

    // Set up the arguments before we fork, as we don't want to call malloc()
    // from the fork, and it can be called from c_str() in theory.
    size_t len = stdbufCommand.size();
    char * argv[len + 1];
    for (int i = 0; i < len; i++) {
        argv[i] = (char *) stdbufCommand[i].c_str();
    }
    argv[len] = NULL;

    // Create a pipe for the child to accurately report launch errors back
    // to the parent.  We set the close-on-exit so that when the new
    // process has finished launching, the pipe will be completely closed
    // and we can use this to know that it has properly started.

    int childLaunchStatusFd[2];
    int res = ::pipe2(childLaunchStatusFd, O_CLOEXEC);
    if (res == -1)
        throw ML::Exception(errno, "pipe() for status");

    int childPid = fork();
    if (childPid == -1) {
        throw ML::Exception(errno, "fork() in RunWrapper");
    }
    else if (childPid == 0) {
        ::close(childLaunchStatusFd[0]);

        ::setpgid(0, 0);

        ::signal(SIGQUIT, SIG_DFL);
        ::signal(SIGTERM, SIG_DFL);
        ::signal(SIGINT, SIG_DFL);

        ::prctl(PR_SET_PDEATHSIG, SIGHUP);
        if (getppid() == 1)
            ::kill(getpid(), SIGHUP);
        ::close(fds.statusFd);
        int res = ::execv(stdbufCommand[0].c_str(), argv);
        if (res == -1) {
            // Report back that we couldn't launch
            int err = errno;
            int res = ::write(childLaunchStatusFd[1], &err, sizeof(err));
            if (res == -1)
                _exit(126);
            else _exit(127);
        }

        /* there is no possible way this code could be executed */
        throw ML::Exception("The Alpha became the Omega.");
    }
    else {
        ::close(childLaunchStatusFd[1]);
        // FILE * terminal = ::fopen("/dev/tty", "a");
        // ::fprintf(terminal, "wrapper: real child pid: %d\n", childPid);
        ChildStatus status;

        // Write an update to the current status
        auto writeStatus = [&] ()
            {
                int res = ::write(fds.statusFd, &status, sizeof(status));
                if (res == -1)
                    throw ML::Exception(errno, "RunWrapper write status");
                else if (res != sizeof(status))
                    throw ML::Exception("didn't completely write status");
            };

        // Write that there was an error to the calling process, and then
        // exit
        auto writeError = [&] (int launchErrno, LaunchErrorCode errorCode,
                               int exitCode)
            {
                status.launchErrno = launchErrno;
                status.launchErrorCode = errorCode;
                
                //cerr << "sending error " << strerror(launchErrno)
                //<< " " << strLaunchError(errorCode) << endl;

                int res = ::write(fds.statusFd, &status, sizeof(status));
                if (res == -1)
                    throw ML::Exception(errno, "RunWrapper write status");
                else if (res != sizeof(status))
                    throw ML::Exception("didn't completely write status");

                fds.close();
                
                _exit(exitCode);
            };

        status.state = LAUNCHING;
        status.pid = childPid;

        writeStatus();

        // ::fprintf(terminal, "wrapper: waiting child...\n");

        // Read from the launch status pipe to know that the launch has
        // finished.
        int launchErrno;
        int bytes = ::read(childLaunchStatusFd[0], &launchErrno,
                           sizeof(launchErrno));
        
        if (bytes == 0) {
            // Launch happened successfully (pipe was closed on exec)
            status.state = RUNNING;
            writeStatus();
        }
        else if (bytes == -1) {
            // Problem reading
            writeError(errno, E_READ_STATUS_PIPE, 127);
        }
        else if (bytes != sizeof(launchErrno)) {
            // Wrong size of message
            writeError(0, E_STATUS_PIPE_WRONG_LENGTH, 127);
        }
        else {
            // Launch was unsuccessful; we have the errno.  Return it and
            // exit.
            writeError(launchErrno, E_SUBTASK_LAUNCH, 126);
        }

        int childStatus;

        int res = ::waitpid(childPid, &childStatus, 0);
        if (res == -1) {
            writeError(errno, E_SUBTASK_WAITPID, 127);
        }
        else if (res != childPid) {
            writeError(0, E_WRONG_CHILD, 127);
        }

        status.state = STOPPED;
        status.childStatus = childStatus;

        writeStatus();

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

    //cerr << "waiting for wrapper" << endl;
    int res = ::waitpid(wrapperPid, NULL, 0);
    if (res == -1)
        throw ML::Exception(errno, "waitpid");
    if (res != wrapperPid)
        throw ML::Exception("waitpid has not returned the wrappedPid");
    wrapperPid = -1;

    //cerr << "finished waiting for wrapper" << endl;

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
    runResult = RunResult();
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


/* RUNRESULT */

void
RunResult::
updateFromStatus(int status)
{
    if (WIFEXITED(status)) {
        state = RETURNED;
        returnCode = WEXITSTATUS(status);
    }
    else if (WIFSIGNALED(status)) {
        state = SIGNALED;
        signum = WTERMSIG(status);
    }
}

std::string to_string(const RunResult::State & state)
{
    switch (state) {
    case RunResult::UNKNOWN: return "UNKNOWN";
    case RunResult::LAUNCH_ERROR: return "LAUNCH_ERROR";
    case RunResult::RETURNED: return "RETURNED";
    case RunResult::SIGNALED: return "SIGNALED";
    }

    return ML::format("RunResult::State(%d)", state);
}

std::ostream &
operator << (std::ostream & stream, const RunResult::State & state)
{
    return stream << to_string(state);
}


/* EXECUTE */

RunResult
execute(MessageLoop & loop,
        const vector<string> & command,
        const shared_ptr<InputSink> & stdOutSink,
        const shared_ptr<InputSink> & stdErrSink,
        const string & stdInData)
{
    RunResult result;
    auto onTerminate = [&](const RunResult & runResult) {
        result = runResult;
    };

    Runner runner;

    loop.addSource("runner", runner);

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
    runner.waitConnectionState(AsyncEventSource::DISCONNECTED);

    return result;
}

RunResult
execute(const vector<string> & command,
        const shared_ptr<InputSink> & stdOutSink,
        const shared_ptr<InputSink> & stdErrSink,
        const string & stdInData)
{
    MessageLoop loop;

    loop.start();
    RunResult result = execute(loop, command, stdOutSink, stdErrSink,
                               stdInData);

    loop.shutdown();

    return result;
}

} // namespace Datacratic

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
#include "jml/utils/guard.h"
#include "jml/utils/file_functions.h"

#include "logs.h"
#include "message_loop.h"
#include "sink.h"

#include "runner.h"

#include "soa/types/basic_value_descriptions.h"

using namespace std;
using namespace Datacratic;


timevalDescription::
timevalDescription()
{
    addField("tv_sec", &timeval::tv_sec, "seconds");
    addField("tv_usec", &timeval::tv_usec, "micro seconds");
}

rusageDescription::
rusageDescription()
{
    addField("utime", &rusage::ru_utime, "user CPU time used");
    addField("stime", &rusage::ru_stime, "system CPU time used");
    addField("maxrss", &rusage::ru_maxrss, "maximum resident set size");
    addField("ixrss", &rusage::ru_ixrss, "integral shared memory size");
    addField("idrss", &rusage::ru_idrss, "integral unshared data size");
    addField("isrss", &rusage::ru_isrss, "integral unshared stack size");
    addField("minflt", &rusage::ru_minflt, "page reclaims (soft page faults)");
    addField("majflt", &rusage::ru_majflt, "page faults (hard page faults)");
    addField("nswap", &rusage::ru_nswap, "swaps");
    addField("inblock", &rusage::ru_inblock, "block input operations");
    addField("oublock", &rusage::ru_oublock, "block output operations");
    addField("msgsnd", &rusage::ru_msgsnd, "IPC messages sent");
    addField("msgrcv", &rusage::ru_msgrcv, "IPC messages received");
    addField("nsignals", &rusage::ru_nsignals, "signals received");
    addField("nvcsw", &rusage::ru_nvcsw, "voluntary context switches");
    addField("nivcsw", &rusage::ru_nivcsw, "involuntary context switches");
}


namespace {



Logging::Category warnings("Runner::warning");


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
    : closeStdin(false), running_(false), childPid_(-1),
      wakeup_(EFD_NONBLOCK | EFD_CLOEXEC),
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

Epoller::HandleEventResult
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

    return Epoller::DONE;
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
                continue;
            }

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
            task_.runResult.usage = status.usage;

            if (status.launchErrno || status.launchErrorCode) {
                //cerr << "*** launch error" << endl;
                // Error
                task_.runResult.updateFromLaunchError
                    (status.launchErrno,
                     Task::strLaunchError(status.launchErrorCode));

                task_.postTerminate(*this);

                childPid_ = -2;
                ML::futex_wake(childPid_);

                running_ = false;
                ML::futex_wake(running_);
                break;
            }

            switch (status.state) {
            case Task::LAUNCHING:
                childPid_ = status.pid;
                // cerr << " childPid_ = status.pid (launching)\n";
                break;
            case Task::RUNNING:
                childPid_ = status.pid;
                // cerr << " childPid_ = status.pid (running)\n";
                ML::futex_wake(childPid_);
                break;
            case Task::STOPPED:
                childPid_ = -3;
                // cerr << " childPid_ = -3 (stopped)\n";
                ML::futex_wake(childPid_);
                task_.runResult.updateFromStatus(status.childStatus);
                task_.statusState = Task::StatusState::DONE;
                if (stdInSink_ && stdInSink_->state != OutputSink::CLOSED) {
                    stdInSink_->requestClose();
                }
                attemptTaskTermination();
                break;
            case Task::DONE:
                throw ML::Exception("unexpected status DONE");
            case Task::ST_UNKNOWN:
                throw ML::Exception("unexpected status UNKNOWN");
            }

            if (status.launchErrno || status.launchErrorCode)
                break;
        }
    }

    if ((event.events & EPOLLHUP) != 0) {
        //cerr << "*** hangup" << endl;
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
    // cerr << "handleWakup\n";
    while (!wakeup_.tryRead());

    if ((event.events & EPOLLIN) != 0) {
        if (stdInSink_) {
            if (stdInSink_->connectionState_
                == AsyncEventSource::DISCONNECTED) {
                attemptTaskTermination();
                removeFd(wakeup_.fd());
            }
            else {
                wakeup_.signal();
                restartFdOneShot(wakeup_.fd(), event.data.ptr);
            }
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
    if ((!stdInSink_ || stdInSink_->state == OutputSink::CLOSED)
        && !stdOutSink_ && !stdErrSink_ && childPid_ < 0
        && (task_.statusState == Task::StatusState::STOPPED
            || task_.statusState == Task::StatusState::DONE)) {
        task_.postTerminate(*this);

        if (stdInSink_) {
            stdInSink_.reset();
        }

        // cerr << "terminated task\n";
        running_ = false;
        ML::futex_wake(running_);
    }
#if 0
    else {
        cerr << "cannot terminate yet because:\n";
        if ((stdInSink_ && stdInSink_->state != OutputSink::CLOSED)) {
            cerr << "stdin sink active\n";
        }
        if (stdOutSink_) {
            cerr << "stdout sink active\n";
        }
        if (stdErrSink_) {
            cerr << "stderr sink active\n";
        }
        if (childPid_ >= 0) {
            cerr << "childPid_ >= 0\n";
        }
        if (!(task_.statusState == Task::StatusState::STOPPED
              || task_.statusState == Task::StatusState::DONE)) {
            cerr << "task status != stopped/done\n";
        }
    }
#endif
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
    if (parent_ == nullptr) {
        LOG(warnings)
            << ML::format("Runner %p is not connected to any MessageLoop\n", this);
    }

    if (running_)
        throw ML::Exception("already running");

    running_ = true;
    ML::futex_wake(running_);

    task_.statusState = Task::StatusState::ST_UNKNOWN;
    task_.onTerminate = onTerminate;

    Task::ChildFds childFds;
    tie(task_.statusFd, childFds.statusFd) = CreateStdPipe(false);

    if (stdInSink_) {
        tie(task_.stdInFd, childFds.stdIn) = CreateStdPipe(true);
    }
    else if (closeStdin) {
        childFds.stdIn = -1;
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
        task_.runWrapper(command, childFds);
    }
    else {
        task_.statusState = Task::StatusState::LAUNCHING;

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

bool
Runner::
kill(int signum, bool mustSucceed) const
{
    if (childPid_ <= 0) {
        if (mustSucceed)
            throw ML::Exception("subprocess not available");
        else return false;
    }

    ::kill(-childPid_, signum);
    waitTermination();
    return true;
}

bool
Runner::
signal(int signum, bool mustSucceed)
{
    if (childPid_ <= 0) {
        if (mustSucceed)
            throw ML::Exception("subprocess not available");
        else return false;
    }
    
    ::kill(childPid_, signum);
    return true;
}

bool
Runner::
waitStart(double secondsToWait) const
{
    Date deadline = Date::now().plusSeconds(secondsToWait);

    while (childPid_ == -1) {
        //cerr << "waitStart childPid_ = " << childPid_ << endl;
        double timeToWait = Date::now().secondsUntil(deadline);
        if (timeToWait < 0)
            break;
        if (isfinite(timeToWait))
            ML::futex_wait(childPid_, -1, timeToWait);
        else ML::futex_wait(childPid_, -1);
        //cerr << "waitStart childPid_ now = " << childPid_ << endl;
    }

    return childPid_ > 0;
}

void
Runner::
waitTermination() const
{
    while (running_) {
        ML::futex_wait(running_, true);
    }
}

/* RUNNER::TASK */

Runner::Task::
Task()
    : wrapperPid(-1),
      stdInFd(-1),
      stdOutFd(-1),
      stdErrFd(-1),
      statusFd(-1),
      statusState(ST_UNKNOWN)
{}

std::string
Runner::Task::
strLaunchError(LaunchErrorCode error)
{
    switch (error) {
    case E_NONE: return "no error";
    case E_READ_STATUS_PIPE: return "read() on status pipe";
    case E_STATUS_PIPE_WRONG_LENGTH:
        return "wrong message size reading launch pipe";
    case E_SUBTASK_LAUNCH: return "exec() launching subtask";
    case E_SUBTASK_WAITPID: return "waitpid waiting for subtask";
    case E_WRONG_CHILD: return "waitpid() returned the wrong child";
    }
    throw ML::Exception("unknown error launch error code %d",
                        error);
}

std::string
Runner::Task::
statusStateAsString(StatusState statusState)
{
    switch (statusState) {
    case ST_UNKNOWN: return "UNKNOWN";
    case LAUNCHING: return "LAUNCHING";
    case RUNNING: return "RUNNING";
    case STOPPED: return "STOPPED";
    case DONE: return "DONE";
    }
    throw ML::Exception("unknown status %d", statusState);
}

void
Runner::Task::
runWrapper(const vector<string> & command, ChildFds & fds)
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

    // Create a pipe for the child to accurately report launch errors back
    // to the parent.  We set the close-on-exit so that when the new
    // process has finished launching, the pipe will be completely closed
    // and we can use this to know that it has properly started.

    int childLaunchStatusFd[2] = { -1, -1 };

    // Arrange for them to be closed in the case of an exception.
    ML::Call_Guard guard([&] ()
                         {
                             if (childLaunchStatusFd[0] != -1)
                                 ::close(childLaunchStatusFd[0]);
                             if (childLaunchStatusFd[1] != -1)
                                 ::close(childLaunchStatusFd[1]);

                         });
    int res = ::pipe2(childLaunchStatusFd, O_CLOEXEC);
    if (res == -1)
        throw ML::Exception(errno, "pipe() for status");

    int childPid = fork();
    if (childPid == -1) {
        throw ML::Exception(errno, "fork() in runWrapper");
    }
    else if (childPid == 0) {
        ::close(childLaunchStatusFd[0]);

        ::setsid();

        ::signal(SIGQUIT, SIG_DFL);
        ::signal(SIGTERM, SIG_DFL);
        ::signal(SIGINT, SIG_DFL);

        ::prctl(PR_SET_PDEATHSIG, SIGHUP);
        if (getppid() == 1) {
            cerr << "runner: parent process already dead\n";
            ::kill(getpid(), SIGHUP);
        }
        ::close(fds.statusFd);
        int res = ::execv(command[0].c_str(), argv);
        if (res == -1) {
            // Report back that we couldn't launch
            int err = errno;
            int res = ::write(childLaunchStatusFd[1], &err, sizeof(err));
            if (res == -1)
                _exit(124);
            else _exit(125);
        }

        // No need to close the FDs because this fork won't last long

        /* there is no possible way this code could be executed */
        throw ML::Exception("The Alpha became the Omega.");
    }
    else {
        ::close(childLaunchStatusFd[1]);
        childLaunchStatusFd[1] = -1;
        // FILE * terminal = ::fopen("/dev/tty", "a");
        // ::fprintf(terminal, "wrapper: real child pid: %d\n", childPid);
        ChildStatus status;

        // Write an update to the current status
        auto writeStatus = [&] ()
            {
                int res = ::write(fds.statusFd, &status, sizeof(status));
                if (res == -1)
                    throw ML::Exception(errno, "runWrapper write status");
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
                //<< " " << strLaunchError(errorCode) << " and exiting with "
                //<< exitCode << endl;

                int res = ::write(fds.statusFd, &status, sizeof(status));
                if (res == -1)
                    throw ML::Exception(errno, "runWrapper write status");
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
        else {
            // Error launching

            //cerr << "got launch error" << endl;
            int childStatus;
            // We ignore the error code for this... there is nothing we
            // can do if we can't waitpid
            while (::waitpid(childPid, &childStatus, 0) == -1 && errno == EINTR) ;

            //cerr << "waitpid on " << childPid << " returned "
            //     << res << " with childStatus "
            //     << childStatus << endl;

            //cerr << "done with an error; first wait for the child to exit"
            //     << endl;

            if (bytes == -1) {
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
        }

        int childStatus;
        int res;
        while ((res = ::waitpid(childPid, &childStatus, 0)) == -1
               && errno == EINTR);
        if (res == -1) {
            writeError(errno, E_SUBTASK_WAITPID, 127);
        }
        else if (res != childPid) {
            writeError(0, E_WRONG_CHILD, 127);
        }

        status.state = STOPPED;
        status.childStatus = childStatus;
        getrusage(RUSAGE_CHILDREN, &status.usage);

        writeStatus();

        fds.close();

        ::_exit(0);
    }
}

void
Runner::Task::
postTerminate(Runner & runner)
{
    // cerr << "postTerminate\n";

    if (wrapperPid <= 0) {
        throw ML::Exception("wrapperPid <= 0, has postTerminate been executed before?");
    }

    // cerr << "waiting for wrapper pid: " << wrapperPid << endl;
    int wrapperPidStatus;
    while (true) {
        int res = ::waitpid(wrapperPid, &wrapperPidStatus, 0);
        if (res == wrapperPid) {
            break;
        }
        else if (res == -1) {
            if (errno != EINTR) {
                throw ML::Exception(errno, "waitpid");
            }
        }
        else {
            throw ML::Exception("waitpid has not returned the wrappedPid");
        }
    }
    wrapperPid = -1;

    //cerr << "finished waiting for wrapper with status " << wrapperPidStatus
    //     << endl;

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

Runner::Task::ChildFds::
ChildFds()
    : stdIn(::fileno(stdin)),
      stdOut(::fileno(stdout)),
      stdErr(::fileno(stderr)),
      statusFd(-1)
{
}

/* child api */
void
Runner::Task::ChildFds::
closeRemainingFds()
{
    struct rlimit limits;
    ::getrlimit(RLIMIT_NOFILE, &limits);

    for (int fd = 0; fd < limits.rlim_cur; fd++) {
        if ((fd != STDIN_FILENO || stdIn == -1)
            && fd != STDOUT_FILENO && fd != STDERR_FILENO
            && fd != statusFd) {
            ::close(fd);
        }
    }
}

void
Runner::Task::ChildFds::
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
    if (stdIn != -1) {
        dupToStdStream(stdIn, STDIN_FILENO);
    }
    dupToStdStream(stdOut, STDOUT_FILENO);
    dupToStdStream(stdErr, STDERR_FILENO);
}

/* parent & child api */
void
Runner::Task::ChildFds::
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


/* RUNNER TASK CHILDSTATUS */

Runner::Task::ChildStatus::
ChildStatus()
{
    // Doing it this way keeps ValGrind happy
    ::memset(this, 0, sizeof(*this));

    state = ST_UNKNOWN;
    pid = -1;
    childStatus = -1;
    launchErrno = 0;
    launchErrorCode = E_NONE;
}


/* RUNRESULT */

RunResult::
RunResult()
    : state(UNKNOWN), signum(-1), returnCode(-1), launchErrno(0)
{
    ::memset(&usage, 0, sizeof(usage));
}

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

int
RunResult::
processStatus()
    const
{
    int status;

    if (state == RETURNED)
        status = returnCode;
    else if (state == SIGNALED)
        status = 128 + signum;
    else if (state == LAUNCH_ERROR) {
        if (launchErrno == EPERM) {
            status = 126;
        }
        else if (launchErrno == ENOENT) {
            status = 127;
        }
        else {
            status = 1;
        }
    }
    else
        throw ML::Exception("unhandled state");

    return status;
}

void
RunResult::
updateFromLaunchError(int launchErrno,
                      const std::string & launchError)
{
    this->state = LAUNCH_ERROR;
    this->launchErrno = launchErrno;
    if (!launchError.empty()) {
        this->launchError = launchError;
        if (launchErrno)
            this->launchError += std::string(": ")
                + strerror(launchErrno);
    }
    else {
        this->launchError = strerror(launchErrno);
    }
}

std::string
to_string(const RunResult::State & state)
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

RunResultDescription::
RunResultDescription()
{
    addField("state", &RunResult::state, "State of run command");
    addField("signum", &RunResult::signum,
             "Signal number that it exited with", -1);
    addField("returnCode", &RunResult::returnCode,
             "Return code of command", -1);
    addField("launchErrno", &RunResult::launchErrno,
             "Errno for launch error", 0);
    addField("launchError", &RunResult::launchError,
             "Error message for launch error");
    addField("usage", &RunResult::usage,
             "Process statistics as returned by getrusage()");
}

RunResultStateDescription::
RunResultStateDescription()
{
    addValue("UNKNOWN", RunResult::UNKNOWN,
             "State is unknown or uninitialized");
    addValue("LAUNCH_ERROR", RunResult::LAUNCH_ERROR,
             "Command was unable to be launched");
    addValue("RETURNED", RunResult::RETURNED, "Command returned");
    addValue("SIGNALED", RunResult::SIGNALED, "Command exited with a signal");
}


/* EXECUTE */

RunResult
execute(MessageLoop & loop,
        const vector<string> & command,
        const shared_ptr<InputSink> & stdOutSink,
        const shared_ptr<InputSink> & stdErrSink,
        const string & stdInData,
        bool closeStdin)
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
        runner.closeStdin = closeStdin;
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
        const string & stdInData,
        bool closeStdin)
{
    MessageLoop loop(1, 0, -1);

    loop.start();
    RunResult result = execute(loop, command, stdOutSink, stdErrSink,
                               stdInData, closeStdin);
    loop.shutdown();

    return result;
}

} // namespace Datacratic

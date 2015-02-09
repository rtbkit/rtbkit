/* runner_common.h                                                   -*-C++-*-
   Wolfgang Sourdeau, 10 December 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.
*/

#include <sys/time.h>
#include <sys/resource.h>

#include <string>


namespace Datacratic {

/****************************************************************************/
/* LAUNCH ERROR CODE                                                        */
/****************************************************************************/

/** Possible errors that could happen in launching.  These are
    enumerated here so that they can be passed back as an int
    rather than as a variable length string (or a const char *
    to memory which we could have to ensure was available in
    both the launcher process and the calling process).
*/
enum struct LaunchError {
    NONE,                     ///< No launch error
    READ_STATUS_PIPE,         ///< Error reading status pipe
    STATUS_PIPE_WRONG_LENGTH, ///< Status msg wrong length
    SUBTASK_LAUNCH,           ///< Error launching subtask
    SUBTASK_WAITPID,          ///< Error calling waitpid
    WRONG_CHILD               ///< Wrong child was reaped
};

/** Turn a launch error code into a descriptive string. */
std::string strLaunchError(LaunchError error);
            

/****************************************************************************/
/* PROCESS STATE                                                            */
/****************************************************************************/

/** State of the process. */
enum struct ProcessState {
    UNKNOWN,    ///< Unknown status
    LAUNCHING,     ///< Being launched
    RUNNING,       ///< Currently running
    STOPPED,       ///< No longer running
    DONE           ///< Completely stopped
};

std::string statusStateAsString(ProcessState statusState);


/****************************************************************************/
/* PROCESS STATUS                                                           */
/****************************************************************************/

/** Structure passed back and forth between the launcher and the monitor to
    know the current state of the running process.
*/
struct ProcessStatus {
    ProcessStatus();

    void setErrorCodes(int newLaunchErrno, LaunchError newErrorCode);

    ProcessState state;
    pid_t pid;
    int childStatus;
    int launchErrno;
    LaunchError launchErrorCode;
    rusage usage;
};


/****************************************************************************/
/* PROCESS FDS                                                              */
/****************************************************************************/

struct ProcessFds {
    ProcessFds();

    void closeRemainingFds();
    void dupToStdStreams();
    void close();

    void encodeToBuffer(char * buffer, size_t bufferSize) const;
    void decodeFromBuffer(const char * buffer);

    void writeStatus(const ProcessStatus & status) const;

    int stdIn;
    int stdOut;
    int stdErr;
    int statusFd;
};

}

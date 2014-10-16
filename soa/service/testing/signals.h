#pragma once

#include <signal.h>

namespace Datacratic {

struct BlockedSignals
{
    BlockedSignals(const sigset_t & newSet)
    {
        blockMask(newSet);
    }

    ~BlockedSignals()
    {
        // Clear the pending signals before we unblock them
        // This avoids us getting spurious signals, especially SIGCHLD from
        // grandchildren, etc.
        struct timespec timeout = { 0, 0 };
        siginfo_t info;
        while (::sigtimedwait(&newSet_, &info, &timeout) != -1) ;

        // Now unblock
        ::pthread_sigmask(SIG_UNBLOCK, &oldSet_, NULL);
    }

    BlockedSignals(int signum)
    {
        sigemptyset(&newSet_);
        sigaddset(&newSet_, signum);

        blockMask(newSet_);
    }

    void blockMask(const sigset_t & newSet)
    {
        newSet_ = newSet;
        ::pthread_sigmask(SIG_BLOCK, &newSet, &oldSet_);
    }

    sigset_t newSet_;
    sigset_t oldSet_;
};

}

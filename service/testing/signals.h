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
        sigprocmask(SIG_UNBLOCK, &oldSet_, NULL);
    }

    BlockedSignals(int signum)
    {
        sigset_t newSet;

        sigemptyset(&newSet);
        sigaddset(&newSet, signum);

        blockMask(newSet);
    }

    void blockMask(const sigset_t & newSet)
    {
        ::sigprocmask(SIG_BLOCK, &newSet, &oldSet_);
    }

    sigset_t oldSet_;
};

}

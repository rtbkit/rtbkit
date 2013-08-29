#include <signal.h>

struct DisabledSignal
{
    DisabledSignal(int signum)
    : signum_(signum)
    {
        oldHandler_ = ::signal(signum, SIG_DFL);
    }

    ~DisabledSignal()
    {
        ::signal(signum_, oldHandler_);
    }

    int signum_;
    sighandler_t oldHandler_;
};

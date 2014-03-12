/** watchdog.h                                                     -*- C++ -*-
    Jeremy Barnes, 16 May 2011
    Copyright (c) 2011 Datacratic.  All rights reserved.

    Watchdog timer class.
*/

#ifndef __jml_testing__watchdog_h__
#define __jml_testing__watchdog_h__

#include <boost/thread/thread.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>
#include <iostream>
#include <signal.h>

namespace ML {

struct Watchdog {
    bool finished;
    double seconds;
    boost::thread_group tg;
    boost::function<void ()> timeoutFunction;
    
    static void abortProcess()
    {
        using namespace std;

        cerr << "**** WATCHDOG TIMEOUT; KILLING HUNG TEST ****"
             << endl;
        abort();
        kill(0, SIGKILL);
    }
    
    void runThread()
    {
        struct timespec ts = { 0, 10000000 };
        struct timespec rem;
        for (unsigned i = 0;  i != int(seconds * 100) && !finished;
             ++i) {
            nanosleep(&ts, &rem);
        }
        
        if (!finished)
            timeoutFunction();
    }
    
    /** Create a watchdog timer that will time out after the given number
        of seconds.
    */
    Watchdog(double seconds = 2.0,
             boost::function<void ()> timeoutFunction = abortProcess)
        : finished(false), seconds(seconds), timeoutFunction(timeoutFunction)
    {
        //return;
        tg.create_thread(boost::bind(&Watchdog::runThread,
                                     this));
    }

    ~Watchdog()
    {
        finished = true;
        tg.join_all();
    }
};

} // namespace ML

#endif /* __jml_testing__watchdog_h__ */


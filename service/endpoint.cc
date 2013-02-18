/* endpoint.cc
   Jeremy Barnes, 21 February 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

*/

#include "soa/service//endpoint.h"

#include "soa/service//http_endpoint.h"
#include "jml/arch/cmp_xchg.h"
#include "jml/arch/atomic_ops.h"
#include "jml/arch/format.h"
#include "jml/arch/exception.h"
#include "jml/arch/demangle.h"
#include "jml/arch/backtrace.h"
#include "jml/arch/timers.h"
#include "jml/arch/futex.h"
#include "jml/utils/set_utils.h"
#include "jml/utils/guard.h"
#include "jml/utils/vector_utils.h"
#include "jml/utils/smart_ptr_utils.h"
#include "jml/utils/exc_assert.h"
#include "jml/arch/rt.h"
#include <sys/prctl.h>
#include <sys/epoll.h>
#include <poll.h>


using namespace std;
using namespace ML;


namespace Datacratic {

/*****************************************************************************/
/* ENDPOINT BASE                                                             */
/*****************************************************************************/

EndpointBase::
EndpointBase(const std::string & name)
    : idle(1), modifyIdle(true),
      name_(name),
      threadsActive_(0),
      numTransports(0), shutdown_(false)
{
    Epoller::init(16384);
    Epoller::addFd(wakeup.fd());
    Epoller::handleEvent = std::bind(&EndpointBase::handleEpollEvent,
                                     this,
                                     std::placeholders::_1);
}

EndpointBase::
~EndpointBase()
{
}

void
EndpointBase::
spinup(int num_threads, bool synchronous)
{
    shutdown_ = false;

    if (eventThreads)
        throw Exception("spinup with threads already up");
    eventThreads.reset(new boost::thread_group());

    threadsActive_ = 0;

    for (unsigned i = 0;  i < num_threads;  ++i) {
        boost::thread * thread
            = eventThreads->create_thread
            ([=] ()
             {
                 this->runEventThread(i, num_threads);
             });
        eventThreadList.push_back(thread);
    }

    if (synchronous) {
        for (;;) {
            int oldValue = threadsActive_;
            if (oldValue >= num_threads) break;
            //cerr << "threadsActive_ " << threadsActive_
            //     << " of " << num_threads << endl;
            futex_wait(threadsActive_, oldValue);
            //ML::sleep(0.001);
        }
    }
}

void
EndpointBase::
makeRealTime(int priority)
{
    for (unsigned i = 0;  i < eventThreadList.size();  ++i)
        makeThreadRealTime(*eventThreadList[i], priority);
}

void
EndpointBase::
shutdown()
{
    //cerr << "Endpoint shutdown" << endl;
    //cerr << "numTransports = " << numTransports << endl;

    closePeer();

    {
        Guard guard(lock);
        //cerr << "sending shutdown to " << alive.size() << " transports"
        //     << endl;

        for (auto it = alive.begin(), end = alive.end();  it != end;  ++it) {
            auto transport = it->get();
            //cerr << "shutting down transport " << transport->status() << endl;
            transport->doAsync([=] ()
                               {
                                   //cerr << "killing transport " << transport
                                   //     << endl;
                                   transport->closeWhenHandlerFinished();
                               },
                               "killtransport");
        }
    }

    //cerr << "eventThreads = " << eventThreads.get() << endl;
    //cerr << "eventThreadList.size() = " << eventThreadList.size() << endl;

    //cerr << "numTransports = " << numTransports << endl;

    sleepUntilIdle();

    //cerr << "idle" << endl;

    while (numTransports != 0) {
        //cerr << "shutdown " << this << ": numTransports = "
        //     << numTransports << endl;
        ML::sleep(0.1);
    }

    //cerr << "numTransports = " << numTransports << endl;

    shutdown_ = true;
    ML::memory_barrier();
    wakeup.signal();

    if (eventThreads) {
        eventThreads->join_all();
        eventThreads.reset();
    }
    eventThreadList.clear();

    // Now undo the signal
    wakeup.read();

}

void
EndpointBase::
useThisThread()
{
    runEventThread(-1, -1);
}

void
EndpointBase::
notifyNewTransport(const std::shared_ptr<TransportBase> & transport)
{
    Guard guard(lock);

    //cerr << "new transport " << transport << endl;

    if (alive.count(transport))
        throw ML::Exception("active set already contains connection");
    alive.insert(transport);

    int fd = transport->getHandle();
    if (fd < 0)
        throw Exception("notifyNewTransport: fd %d out of range");

    startPolling(transport.get());

    if (numTransports++ == 0 && modifyIdle)
        idle.acquire();
    futex_wake(numTransports);

    int & ntr = numTransportsByHost[transport->getPeerName()];
    ++ntr;

    //cerr << "host " << transport->getPeerName() << " has "
    //     << ntr << " connections" << endl;


    if (onTransportOpen)
        onTransportOpen(transport.get());
}

void
EndpointBase::
startPolling(TransportBase * transport)
{
    addFdOneShot(transport->epollFd_, transport);
}

void
EndpointBase::
stopPolling(TransportBase * transport)
{
    removeFd(transport->epollFd_);
}

void
EndpointBase::
restartPolling(TransportBase * transport)
{
    restartFdOneShot(transport->epollFd_, transport);
}

void
EndpointBase::
notifyCloseTransport(const std::shared_ptr<TransportBase> & transport)
{
#if 0
    cerr << "closed transport " << transport << " with fd "
         << transport->getHandle() << " with " << transport.use_count()
         << " references" << " and " << transport->hasAsync() << " async"
         << endl;
#endif

    if (onTransportClose)
        onTransportClose(transport.get());

    stopPolling(transport.get());

    transport->zombie_ = true;
    transport->closePeer();

    Guard guard(lock);
    if (!alive.count(transport)) {
        cerr << "closed transport " << transport << " with fd "
             << transport->getHandle() << " with " << transport.use_count()
             << " references" << " and " << transport->hasAsync() << " async"
             << endl;
        cerr << "activities: " << endl;
        transport->activities.dump();
        cerr << endl << endl;

        throw ML::Exception("active set didn't contain connection");
    }
    alive.erase(transport);

    int & ntr = numTransportsByHost[transport->getPeerName()];
    --numTransports;
    futex_wake(numTransports);
    --ntr;
    if (ntr <= 0)
        numTransportsByHost.erase(transport->getPeerName());
    if (numTransports == 0 && modifyIdle)
        idle.release();
}

void
EndpointBase::
notifyRecycleTransport(const std::shared_ptr<TransportBase> & transport)
{
    notifyCloseTransport(transport);
}

void
EndpointBase::
sleepUntilIdle() const
{
    for (;;) {
        //cerr << "sleepUntilIdle " << this << ": numTransports = "
        //     << numTransports << endl;
        ACE_Time_Value time(0, 100000);
        time += ACE_OS::gettimeofday();
        int res = idle.acquire(time);
        if (res != -1) {
            idle.release();
            return;
        }

        Guard guard(lock);
        cerr << alive.size() << " transports" << endl;

        for (auto it = alive.begin(), end = alive.end();  it != end;  ++it) {
            auto transport = it->get();
            cerr << "transport " << transport->status() << endl;
        }

        dumpState();
    }
}

void
EndpointBase::
dumpState() const
{
    Guard guard(lock);
    cerr << "----------------------------------------------" << endl;
    cerr << "Endpoint of type " << type_name(*this)
         << " with " << numTransports << " transports"
         << endl;

}

int
EndpointBase::
numConnections() const
{
    return numTransports;
}

std::map<std::string, int>
EndpointBase::
numConnectionsByHost() const
{
    Guard guard(lock);
    return numTransportsByHost;
}

/** Handle a single ePoll event */
bool
EndpointBase::
handleEpollEvent(epoll_event & event)
{
    bool debug = false;

    if (debug) {
        cerr << "handleEvent" << endl;
        int mask = event.events;
                
        cerr << "events " 
             << (mask & EPOLLIN ? "I" : "")
             << (mask & EPOLLOUT ? "O" : "")
             << (mask & EPOLLPRI ? "P" : "")
             << (mask & EPOLLERR ? "E" : "")
             << (mask & EPOLLHUP ? "H" : "")
             << (mask & EPOLLRDHUP ? "R" : "")
             << endl;
    }            
            
    TransportBase * transport_
        = reinterpret_cast<TransportBase *>(event.data.ptr);
    
    //cerr << "transport_ = " << transport_ << endl; 

    if (transport_ == 0) return true;  // wakeup for shutdown

    if (debug)
        cerr << "transport status = " << transport_->status() << endl;

    // Pin it so that it can't be destroyed whilst handling messages
    std::shared_ptr<TransportBase> transport
        = transport_->shared_from_this();

    transport->handleEvents();

    if (!transport->isZombie())
        this->restartPolling(transport.get());

    return false;
}

void
EndpointBase::
runEventThread(int threadNum, int numThreads)
{
    //cerr << "runEventThread" << endl;

    prctl(PR_SET_NAME,"EptCtrl",0,0,0);

    bool debug = false;
    //debug = name() == "Backchannel";
    //debug = threadNum == 7;

    ML::Duty_Cycle_Timer duty;

    Date lastCheck = Date::now();

    ML::atomic_inc(threadsActive_);
    futex_wake(threadsActive_);
    //cerr << "threadsActive_ " << threadsActive_ << endl;

    Epoller::OnEvent beforeSleep = [&] ()
        {
            duty.notifyBeforeSleep();
        };

    Epoller::OnEvent afterSleep = [&] ()
        {
            duty.notifyAfterSleep();
        };


    // Where does my timeslice start?
    double timesliceUs = 1000.0 / numThreads;
    int myStartUs = timesliceUs * threadNum;
    int myEndUs   = timesliceUs * (threadNum + 1);

    if (debug) {
        static ML::Spinlock lock;
        lock.acquire();
        cerr << "threadNum = " << threadNum << " of " << name()
             << " numThreads = " << numThreads
             << " myStartUs = " << myStartUs
             << " myEndUs = " << myEndUs
             << endl;
        lock.release();
    }

    bool forceInSlice = false;

    while (!shutdown_) {

        Date now = Date::now();
        
        if (now.secondsSince(lastCheck) > 1.0 && debug) {
            ML::Duty_Cycle_Timer::Stats stats = duty.stats();
            string msg = format("control thread for %s: "
                                "events %lld sleeping %lld "
                                "processing %lld duty %.2f%%",
                                name().c_str(),
                                (long long)stats.numWakeups,
                                (long long)stats.usAsleep,
                                (long long)stats.usAwake,
                                stats.duty_cycle() * 100.0);
            cerr << msg << flush;
            duty.clear();
            lastCheck = now;
        }

        int us = now.fractionalSeconds() * 1000000;
        int fracms = us % 1000;  // fractional part of the millisecond
        
        if (debug && false) {
            cerr << "now = " << now.print(6) << " us = " << us
                 << " fracms = " << fracms << " myStartUs = "
                 << myStartUs << " myEndUs = " << myEndUs
                 << endl;
        }

        // Are we in our timeslice?
        if (/* forceInSlice
               || */(fracms >= myStartUs && fracms < myEndUs)) {
            // Yes... then sleep in epoll_wait...
            int usToWait = myEndUs - fracms;
            if (usToWait < 0 || usToWait > timesliceUs)
                usToWait = timesliceUs;
            
            int numHandled = handleEvents(usToWait, 4, handleEvent,
                                          beforeSleep, afterSleep);
            if (debug && false)
                cerr << "  in slice: handled " << numHandled << " events "
                     << "for " << usToWait << " microseconds "
                     << " taken " << Date::now().secondsSince(now) * 1000000
                     << "us" << endl;
            if (numHandled == -1) break;
            forceInSlice = false;
        }
        else {
            // No... try to handle something and then sleep if we don't
            // find anything to do
            int numHandled = handleEvents(0, 1, handleEvent,
                                          beforeSleep, afterSleep);
            if (debug && false)
                cerr << "  out of slice: handled " << numHandled << " events"
                     << endl;
            if (numHandled == -1) break;
            if (numHandled == 0) {
                // Sleep until our timeslice
                duty.notifyBeforeSleep();
                int usToSleep = myStartUs - fracms;
                if (usToSleep < 0)
                    usToSleep += 1000;
                ExcAssertGreaterEqual(usToSleep, 0);
                if (debug && false)
                    cerr << "sleeping for " << usToSleep << " micros" << endl;
                ML::sleep(usToSleep / 1000000.0);
                duty.notifyAfterSleep();
                forceInSlice = true;

                if (debug && false)
                    cerr << " slept for "
                         << Date::now().secondsSince(now) * 1000000
                         << "us when " << usToSleep << " requested"
                         << endl;
            }
        }
    }

    cerr << "thread shutting down" << endl;

    ML::atomic_dec(threadsActive_);
    futex_wake(threadsActive_);
}

} // namespace Datacratic

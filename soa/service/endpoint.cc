/* endpoint.cc
   Jeremy Barnes, 21 February 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

*/

#include <sys/timerfd.h>

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
      numTransports(0), shutdown_(false), disallowTimers_(false),
      pollingMode_(MIN_CONTEXT_SWITCH_POLLING)
{
    Epoller::init(16384);
    auto wakeupData = make_shared<EpollData>(EpollData::EpollDataType::WAKEUP,
                                             wakeup.fd());
    epollDataSet.insert(wakeupData);
    Epoller::addFd(wakeupData->fd, wakeupData.get());
    Epoller::handleEvent = [&] (epoll_event & event) {
        return this->handleEpollEvent(event);
    };
}

EndpointBase::
~EndpointBase()
{
    shutdown();
}

void
EndpointBase::
setPollingMode(enum PollingMode mode)
{
    pollingMode_ = mode;

    int timeout = modePollTimeout(mode);
    setPollTimeout(timeout);
}

void
EndpointBase::
addPeriodic(double timePeriodSeconds, OnTimer toRun)
{
    if (!toRun)
        throw ML::Exception("'toRun' cannot be nil");

    int timerFd = timerfd_create(CLOCK_MONOTONIC, TFD_NONBLOCK);
    if (timerFd == -1)
        throw ML::Exception(errno, "timerfd_create");

    itimerspec spec;
    
    int res = clock_gettime(CLOCK_MONOTONIC, &spec.it_value);
    if (res == -1)
        throw ML::Exception(errno, "clock_gettime");
    uint64_t seconds, nanoseconds;
    seconds = timePeriodSeconds;
    nanoseconds = (timePeriodSeconds - seconds) * 1000000000;

    spec.it_interval.tv_sec = spec.it_value.tv_sec = seconds;
    spec.it_interval.tv_nsec = spec.it_value.tv_nsec = nanoseconds;

    res = timerfd_settime(timerFd, 0, &spec, 0);
    if (res == -1)
        throw ML::Exception(errno, "timerfd_settime");

    auto timerData = make_shared<EpollData>(EpollData::EpollDataType::TIMER,
                                            timerFd);
    timerData->onTimer = toRun;
    startPolling(timerData);
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

    resourceUsage.resize(num_threads);

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
    // First, close the peer so that we don't get any more connections
    // while shutting down
    closePeer();

    //cerr << "Endpoint shutdown" << endl;
    //cerr << "numTransports = " << numTransports << endl;

    /* we pin all EpollDataSet instances to avoid freeing them whilst handling
       messages */
    EpollDataSet dataSetCopy = epollDataSet;
    
    {
        Guard guard(lock);
        //cerr << "sending shutdown to " << transportMapping.size()
        //<< " transports" << endl;

        for (const auto & it: transportMapping) {
            auto transport = it.first.get();
            // cerr << "shutting down transport " << transport->status() << endl;
            transport->closeAsync();
        }
    }

    disallowTimers_ = true;
    ML::memory_barrier();
    {
        /* we remove timer infos separately, as transport infos will be
           removed via notifyCloseTransport */
        for (const auto & it: dataSetCopy) {
            if (it->fdType == EpollData::EpollDataType::TIMER) {
                stopPolling(it);
            }
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
        int oldValue = numTransports;
        ML::futex_wait(numTransports, oldValue, 0.1);
    }

    //cerr << "numTransports = " << numTransports << endl;

    shutdown_ = true;
    ML::memory_barrier();
    wakeup.signal();

    while (threadsActive_ > 0) {
        int oldValue = threadsActive_;
        ML::futex_wait(threadsActive_, oldValue);
    }

    {
        /* we can now close the timer fds as we now that they will no longer
           be listened to */
        MutexGuard guard(dataSetLock);
        for (const auto & it: dataSetCopy) {
            if (it->fdType == EpollData::EpollDataType::TIMER) {
                ::close(it->fd);
            }
        }
    }

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

    if (transportMapping.count(transport))
        throw ML::Exception("active set already contains connection");
    auto epollData
        = make_shared<EpollData>(EpollData::EpollDataType::TRANSPORT,
                                 transport->epollFd_);
    epollData->transport = transport;
    transportMapping.insert({transport, epollData});

    int fd = transport->getHandle();
    if (fd < 0)
        throw Exception("notifyNewTransport: fd %d out of range", fd);

    startPolling(epollData);

    ML::atomic_inc(numTransports);
    if (numTransports == 1 && modifyIdle)
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
startPolling(const shared_ptr<EpollData> & epollData)
{
    MutexGuard guard(dataSetLock);
    auto inserted = epollDataSet.insert(epollData);
    if (!inserted.second)
        throw ML::Exception("epollData already present");
    addFdOneShot(epollData->fd, epollData.get());
}

void
EndpointBase::
stopPolling(const shared_ptr<EpollData> & epollData)
{ 
    removeFd(epollData->fd);
    MutexGuard guard(dataSetLock);
    epollDataSet.erase(epollData);
}

void
EndpointBase::
restartPolling(EpollData * epollDataPtr)
{
    restartFdOneShot(epollDataPtr->fd, epollDataPtr);
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

    Guard guard(lock);
    if (!transportMapping.count(transport)) {
        cerr << "closed transport " << transport << " with fd "
             << transport->getHandle() << " with " << transport.use_count()
             << " references" << " and " << transport->hasAsync() << " async"
             << endl;
        cerr << "activities: " << endl;
        transport->activities.dump();
        cerr << endl << endl;

        throw ML::Exception("transportMapping didn't contain connection");
    }

    auto epollData = transportMapping.at(transport);
    stopPolling(epollData);
    transportMapping.erase(transport); 
    
    transport->zombie_ = true;
    transport->closePeer();

    int & ntr = numTransportsByHost[transport->getPeerName()];
    ML::atomic_dec(numTransports);
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
        cerr << transportMapping.size() << " transports" << endl;

        for (auto & it: transportMapping) {
            auto transport = it.first;
            transport->dumpActivities();
            cerr << "transport " << transport->status() << " zombie "
                << transport->isZombie() << endl;
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

Epoller::HandleEventResult
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

    EpollData * epollDataPtr = reinterpret_cast<EpollData *>(event.data.ptr);
    switch (epollDataPtr->fdType) {
    case EpollData::EpollDataType::TRANSPORT: {
        shared_ptr<TransportBase> transport = epollDataPtr->transport;
        pollStart_ = Date::now();
        handleTransportEvent(transport);
        if (!transport->isZombie()) {
            this->restartPolling(epollDataPtr);
        }
        break;
    }
    case EpollData::EpollDataType::TIMER: {
        handleTimerEvent(epollDataPtr->fd, epollDataPtr->onTimer);
        if (!disallowTimers_) {
            this->restartPolling(epollDataPtr);
        }
        break;
    }
    case EpollData::EpollDataType::WAKEUP:
        // wakeup for shutdown
        return Epoller::SHUTDOWN;
    default:
        throw ML::Exception("unrecognized fd type");
    }


    return Epoller::DONE;
}

void
EndpointBase::
handleTransportEvent(const shared_ptr<TransportBase> & transport)
{
    bool debug = false;

    //cerr << "transport_ = " << transport_.get() << endl; 

    if (debug)
        cerr << "transport status = " << transport->status() << endl;

    int res = transport->handleEvents();
    if (res == -1)
        transport->closePeer();
}

void
EndpointBase::
handleTimerEvent(int fd, OnTimer toRun)
{
    uint64_t numWakeups = 0;
    for (;;) {
        int res = ::read(fd, &numWakeups, 8);
        if (res == -1 && errno == EINTR) continue;
        if (res == -1 && (errno == EAGAIN || errno == EWOULDBLOCK))
            break;
        if (res == -1) {
            throw ML::Exception(errno, "timerfd read");
        }
        else if (res != 8)
            throw ML::Exception("timerfd read: wrong number of bytes: %d",
                                res);
        toRun(numWakeups);
        break;
    }
}

void
EndpointBase::
runEventThread(int threadNum, int numThreads)
{
    prctl(PR_SET_NAME,"EptCtrl",0,0,0);

    ML::atomic_inc(threadsActive_);
    futex_wake(threadsActive_);
    //cerr << "threadsActive_ " << threadsActive_ << endl;

    while (!shutdown_) {
        switch (pollingMode_) {
        case MIN_CONTEXT_SWITCH_POLLING:
            doMinCtxSwitchPolling(threadNum, numThreads);
            break;
        case MIN_LATENCY_POLLING:
            doMinLatencyPolling(threadNum, numThreads);
            break;
        case MIN_CPU_POLLING:
            doMinCpuPolling(threadNum, numThreads);
            break;
        default:
            throw ML::Exception("unhandled polling mode");
        }
    }

    // cerr << "thread shutting down" << endl;

    ML::atomic_dec(threadsActive_);
    futex_wake(threadsActive_);
}

void
EndpointBase::
doMinCtxSwitchPolling(int threadNum, int numThreads)
{
    bool debug = false;
    int epoch = 0;
    //debug = name() == "Backchannel";
    //debug = threadNum == 7;

    ML::Duty_Cycle_Timer duty;

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

    Date lastCheck = Date::now();
    // bool forceInSlice = false;

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

        // sync with the loop monitor request
        int i = resourceEpoch;
        if(i != epoch) {
            // query the kernel for performance metrics
            rusage now;
            getrusage(RUSAGE_THREAD, &now);

            // if we're just started, assume we know nothing and don't update the usage
            long s = now.ru_utime.tv_sec+now.ru_stime.tv_sec;
            if(s > 1) {
                std::lock_guard<std::mutex> guard(usageLock);
                resourceUsage[threadNum] = now;
            }

            epoch = i;
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
            // forceInSlice = false;
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

                double secToSleep = double(usToSleep) / 1000000.0;

                ML::sleep(secToSleep);
                duty.notifyAfterSleep();
                // forceInSlice = true;

                if (debug && false)
                    cerr << " slept for "
                         << Date::now().secondsSince(now) * 1000000
                         << "us when " << usToSleep << " requested"
                         << endl;
            }
        }
    }
}

void
EndpointBase::
doMinLatencyPolling(int threadNum, int numThreads)
{
    int epoch = 0;

    while (!shutdown_) {
        // sync with the loop monitor request
        int i = resourceEpoch;
        if(i != epoch) {
            // query the kernel for performance metrics
            rusage now;
            getrusage(RUSAGE_THREAD, &now);

            // if we're just started, assume we know nothing and don't update the usage
            long s = now.ru_utime.tv_sec+now.ru_stime.tv_sec;
            if(s > 1) {
                std::lock_guard<std::mutex> guard(usageLock);
                resourceUsage[threadNum] = now;
            }

            epoch = i;
        }

        handleEvents(0, 1, handleEvent);
    }
}

void
EndpointBase::
doMinCpuPolling(int threadNum, int numThreads)
{
    bool debug = false;

    ML::Duty_Cycle_Timer duty;

    auto beforeSleep = [&] () {
        duty.notifyBeforeSleep();
    };

    auto afterSleep = [&] () {
        duty.notifyAfterSleep();
    };

    if (debug) {
        cerr << ("threadNum = " + to_string(threadNum)
                 + " of " + name()
                 + " numThreads = " + to_string(numThreads)
                 + "\n");
    }

    Date lastCheck = Date::now();

    while (!shutdown_) {
        handleEvents(0, -1, handleEvent, beforeSleep, afterSleep);

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
    }
}

int
EndpointBase::
modePollTimeout(enum PollingMode mode)
    const
{
    return (mode == MIN_CPU_POLLING) ? 1000 : 0;
}

} // namespace Datacratic

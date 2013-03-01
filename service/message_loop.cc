/** message_loop.cc
    Jeremy Barnes, 1 June 2012
    Copyright (c) 2012 Datacratic.  All rights reserved.

*/

#include "soa/service//message_loop.h"
#include "jml/utils/smart_ptr_utils.h"
#include <boost/make_shared.hpp>
#include <iostream>
#include "jml/arch/exception.h"
#include "jml/arch/timers.h"
#include "jml/arch/demangle.h"
#include <time.h>
#include <limits.h>
#include "jml/arch/futex.h"
#include "soa/types/date.h"
#include <sys/epoll.h>
#include "jml/arch/backtrace.h"
#include <thread>


using namespace std;
using namespace ML;


namespace Datacratic {


/*****************************************************************************/
/* ASYNC EVENT SOURCE                                                        */
/*****************************************************************************/



/*****************************************************************************/
/* MESSAGE LOOP                                                              */
/*****************************************************************************/

MessageLoop::
MessageLoop(int numThreads, double maxAddedLatency)
    : numThreadsCreated(0)
{
    init(numThreads, maxAddedLatency);
}

MessageLoop::
~MessageLoop()
{
    shutdown();
}

void
MessageLoop::
init(int numThreads, double maxAddedLatency)
{
    if (maxAddedLatency == 0)
        cerr << "warning: MessageLoop with maxAddedLatency of zero will busy wait" << endl;

    Epoller::init(16384);
    this->shutdown_ = false;
    this->maxAddedLatency_ = maxAddedLatency;
    this->handleEvent = std::bind(&MessageLoop::handleEpollEvent,
                                   this,
                                   std::placeholders::_1);
    debug_ = false;
}

void
MessageLoop::
start(std::function<void ()> onStop)
{
    if (numThreadsCreated)
        throw ML::Exception("already have started message loop");

    //cerr << "starting thread from " << this << endl;
    //ML::backtrace();

    auto runfn = [=] ()
        {
            this->runWorkerThread();
            if (onStop) onStop();
        };

    threads.create_thread(runfn);

    ++numThreadsCreated;
}
    
void
MessageLoop::
startSync(std::function<void ()> onStop)
{
    if (numThreadsCreated)
        throw ML::Exception("already have started message loop");

    ++numThreadsCreated;

    runWorkerThread();
    if (onStop) onStop();
}
    
void
MessageLoop::
shutdown()
{
    if (shutdown_)
        return;

    shutdown_ = true;
    futex_wake((int &)shutdown_);

    threads.join_all();

    numThreadsCreated = 0;
}

void
MessageLoop::
addSource(const std::string & name,
          AsyncEventSource & source, int priority)
{
    addSource(name, make_unowned_std_sp(source), priority);
}

void
MessageLoop::
addSource(const std::string & name,
          std::shared_ptr<AsyncEventSource> source, int priority)
{
    Guard guard(lock);
    addSourceImpl(name, source, priority);
}

void
MessageLoop::
addPeriodic(const std::string & name,
            double timePeriodSeconds,
            std::function<void (uint64_t)> toRun,
            int priority)
{
    addSource(name,
              std::make_shared<PeriodicEventSource>(timePeriodSeconds,
                                                      toRun),
              priority);
}

void
MessageLoop::
addSourceDeferred(const std::string & name,
                  AsyncEventSource & source, int priority)
{
    //cerr << "addSourceDeferred" << endl;
    // TODO: assert that lock is held by this thread
    deferredSources.push_back(make_tuple(name, make_unowned_std_sp(source), priority));
}

void
MessageLoop::
addSourceDeferred(const std::string & name,
                  std::shared_ptr<AsyncEventSource> source, int priority)
{
    //cerr << "addSourceDeferred" << endl;
    // TODO: assert that lock is held by this thread
    deferredSources.push_back(make_tuple(name, source, priority));
}

void
MessageLoop::
addPeriodicDeferred(const std::string & name,
                    double timePeriodSeconds,
                    std::function<void (uint64_t)> toRun,
                    int priority)
{
    addSourceDeferred(name,
                      std::make_shared<PeriodicEventSource>(timePeriodSeconds,
                                                              toRun),
                      priority);
}

void
MessageLoop::
removeSource(AsyncEventSource * source)
{
    // When we free the elements in our destructor, they will try to remove themselves.
    // we just make it a nop.
    if (shutdown_)
        return;

    Guard guard(lock);
    for (unsigned i = 0;  i < sources.size();  ++i) {
        if (sources[i].second.get() != source)
            continue;
        int fd = source->selectFd();
        if (fd != -1) {
            removeFd(fd);

            // Make sure that our and our parent's value of needsPoll is up to date
            bool sourceNeedsPoll = source->needsPoll;
            if (needsPoll && sourceNeedsPoll) {
                bool oldNeedsPoll = needsPoll;
                checkNeedsPoll();
                if (oldNeedsPoll != needsPoll && parent_)
                    parent_->checkNeedsPoll();
            }
        }
        sources.erase(sources.begin() + i);
        return;
    }

    throw ML::Exception("couldn't remove message loop source");
}

void
MessageLoop::
wakeupMainThread()
{
    // TODO: do
}

void
MessageLoop::
startSubordinateThread(const SubordinateThreadFn & thread)
{
    Guard guard(lock);
    int64_t id = 0;
    threads.create_thread(std::bind(thread, std::ref(shutdown_),
                                    id));
}

void
MessageLoop::
runWorkerThread()
{
    Date lastCheck = Date::now();

    ML::Duty_Cycle_Timer duty;

    while (!shutdown_) {


        Date start = Date::now();

        bool more = true;

        if (debug_) {
            cerr << "handling events from " << sources.size()
                 << " sources with needsPoll " << needsPoll << endl;
            for (unsigned i = 0;  i < sources.size();  ++i)
                cerr << sources[i].first << " " << sources[i].second->needsPoll << endl;
        }

        while (more) {
            more = processOne();
        }
        
        Date end = Date::now();

        double elapsed = end.secondsSince(start);
        double sleepTime = maxAddedLatency_ - elapsed;

        duty.notifyBeforeSleep();
        if (sleepTime > 0) {
            ML::sleep(sleepTime);
        }
        duty.notifyAfterSleep();

        if (lastCheck.secondsUntil(end) > 10.0) {
            auto stats = duty.stats();
            //cerr << "message loop: wakeups " << stats.numWakeups
            //     << " duty " << stats.duty_cycle() << endl;
            lastCheck = end;
            duty.clear();
        }
    }
}

bool
MessageLoop::
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
    
    AsyncEventSource * source
        = reinterpret_cast<AsyncEventSource *>(event.data.ptr);
    
    //cerr << "source = " << source << " of type "
    //     << ML::type_name(*source) << endl;

    if (source == 0) return true;  // wakeup for shutdown

    source->processOne();

    return false;
}

bool
MessageLoop::
poll() const
{
    if (needsPoll) {
        Guard guard(lock);
        for (auto & s: sources)
            if (s.second->poll())
                return true;
        return false;
    }
    else return Epoller::poll();
}

bool
MessageLoop::
processOne()
{
    bool more = false;

    if (needsPoll) {
        Guard guard(lock);
            
        for (unsigned i = 0;  i < sources.size();  ++i) {
            try {
                bool hasMore = sources[i].second->processOne();
                if (debug_) {
                    cerr << "source " << sources[i].first << " has " << hasMore << endl;
                }
                //if (hasMore)
                //    cerr << "source " << sources[i].first << " has more" << endl;
                more = more || hasMore;
            } catch (...) {
                cerr << "exception processing source " << sources[i].first
                     << endl;
                throw;
            }
        }
    }
    else more = Epoller::processOne();

    // Add in any deferred sources that have been queued by the event handlers
    if (!deferredSources.empty()) {
        // Taking this lock after calling empty() instead of before is dodgy,
        // but looking at the g++ STL it should never crash and it saves us
        // from a spurious lock on every iteration through.
        // TODO: a better scheme, maybe with a boolean for hasDeferredSources
        Guard guard(lock);

        //cerr << "adding " << deferredSources.size() << " deferred sources"
        //     << endl;

        for (auto s: deferredSources)
            addSourceImpl(std::get<0>(s), std::get<1>(s), std::get<2>(s));
        
        deferredSources.clear();
    }

    return more;
}

void
MessageLoop::
addSourceImpl(const std::string & name,
              std::shared_ptr<AsyncEventSource> source, int priority)
{
    if (debug_) {
        cerr << "adding source " << name << "; we now have " << sources.size()
             << " with needsPoll " << needsPoll;
        if (parent_)
            cerr << " parent needsPoll = " << parent_->needsPoll << endl;
        cerr << endl;
    }

    if (source->parent_)
        throw ML::Exception("adding a source that already has a parent");
    source->parent_ = this;

    sources.push_back(make_pair(name, source));
    int fd = source->selectFd();
    if (fd != -1)
        addFd(fd, source.get());
    bool oldNeedsPoll = needsPoll;
    if (source->needsPoll && !oldNeedsPoll) {
        needsPoll = true;

        // Deadlock?
        if (parent_)
            parent_->checkNeedsPoll();
    }

    if (debug_) {
        cerr << "finished adding source " << name << "; we now have " << sources.size()
             << " with needsPoll " << needsPoll;
        if (parent_)
            cerr << " parent needsPoll = " << parent_->needsPoll << endl;
        cerr << endl;
    }

    wakeupMainThread();
}

void
MessageLoop::
debug(bool debugOn)
{
    debug_ = debugOn;
    Guard guard(lock);
    for (unsigned i = 0;  i < sources.size();  ++i) {
        sources[i].second->debug(debugOn);
    }
}

void
MessageLoop::
checkNeedsPoll()
{
    Guard guard(lock);
    bool newNeedsPoll = false;
    for (unsigned i = 0;  i < sources.size() && !newNeedsPoll;  ++i)
        newNeedsPoll = sources[i].second->needsPoll;
    needsPoll = newNeedsPoll;
}

} // namespace Datacratic

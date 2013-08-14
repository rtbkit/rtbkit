/** message_loop.cc
    Jeremy Barnes, 1 June 2012
    Copyright (c) 2012 Datacratic.  All rights reserved.

*/

#include "soa/service//message_loop.h"
#include "soa/types/date.h"
#include "jml/arch/exception.h"
#include "jml/arch/timers.h"
#include "jml/arch/demangle.h"
#include "jml/arch/futex.h"
#include "jml/arch/backtrace.h"
#include "jml/utils/smart_ptr_utils.h"
#include "jml/utils/exc_assert.h"

#include <boost/make_shared.hpp>
#include <iostream>
#include <thread>
#include <time.h>
#include <limits.h>
#include <sys/epoll.h>


using namespace std;
using namespace ML;


namespace Datacratic {

/*****************************************************************************/
/* MESSAGE LOOP                                                              */
/*****************************************************************************/

MessageLoop::
MessageLoop(int numThreads, double maxAddedLatency)
    : queueFd(EFD_NONBLOCK),
      sourceQueueFlag(false),
      numThreadsCreated(0),
      totalSleepTime_(0.0)
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

    // See the comments on processOne below for more details on this assertion.
    ExcAssertEqual(numThreads, 1);

    Epoller::init(16384);
    this->shutdown_ = false;
    this->maxAddedLatency_ = maxAddedLatency;
    this->handleEvent = std::bind(&MessageLoop::handleEpollEvent,
                                   this,
                                   std::placeholders::_1);
    addFd(queueFd.fd());
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
startSync()
{
    if (numThreadsCreated)
        throw ML::Exception("already have started message loop");

    ++numThreadsCreated;

    runWorkerThread();
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
              std::make_shared<PeriodicEventSource>(timePeriodSeconds, toRun),
              priority);
}

void
MessageLoop::
addSourceImpl(const std::string & name,
              std::shared_ptr<AsyncEventSource> source,
              int priority)
{
    ExcCheck(!source->parent_, "source already has a parent: " + name);
    source->parent_ = this;

    Guard guard(queueLock);
    addSourceQueue.emplace_back(name, source, priority);
    sourceQueueFlag = true;
    queueFd.signal();
}

void
MessageLoop::
removeSource(AsyncEventSource * source)
{
    // When we free the elements in our destructor, they will try to remove themselves.
    // we just make it a nop.
    if (shutdown_) return;

    Guard guard(queueLock);
    removeSourceQueue.emplace_back(source);
    sourceQueueFlag = true;
    queueFd.signal();
}

void
MessageLoop::
processSourceQueue()
{
    if (!sourceQueueFlag) return;

    std::vector<SourceEntry> sourcesToAdd;
    std::vector<AsyncEventSource*> sourcesToRemove;

    {
        Guard guard(queueLock);

        sourcesToAdd = std::move(addSourceQueue);
        sourcesToRemove = std::move(removeSourceQueue);

        sourceQueueFlag = false;
        while (!queueFd.tryRead());
    }

    for (auto& entry : sourcesToAdd)
        processAddSource(entry);

    for (AsyncEventSource* source : sourcesToRemove)
        processRemoveSource(source);
}

void
MessageLoop::
processAddSource(const SourceEntry& entry)
{
    int fd = entry.source->selectFd();
    if (fd != -1)
        addFd(fd, entry.source.get());

    if (!needsPoll && entry.source->needsPoll) {
        needsPoll = true;
        if (parent_) parent_->checkNeedsPoll();
    }

    if (debug_) entry.source->debug(true);
    sources.push_back(entry);
}

void
MessageLoop::
processRemoveSource(AsyncEventSource* source)
{
    auto pred = [&] (const SourceEntry& entry) {
        return entry.source.get() == source;
    };
    auto it = find_if(sources.begin(), sources.end(), pred);

    ExcCheck(it != sources.end(), "couldn't remove source");

    SourceEntry entry = *it;
    sources.erase(it);

    int fd = source->selectFd();
    if (fd == -1) return;
    removeFd(fd);

    // Make sure that our and our parent's value of needsPoll is up to date
    bool sourceNeedsPoll = entry.source->needsPoll;
    if (needsPoll && sourceNeedsPoll) {
        bool oldNeedsPoll = needsPoll;
        checkNeedsPoll();
        if (oldNeedsPoll != needsPoll && parent_)
            parent_->checkNeedsPoll();
    }
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
    Guard guard(threadsLock);
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
                cerr << sources[i].name << " " << sources[i].source->needsPoll << endl;
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
            totalSleepTime_ += sleepTime;
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
        for (auto & s: sources)
            if (s.source->poll())
                return true;
        return false;
    }
    else return Epoller::poll();
}

/** This function assumes that it's called by only a single thread. This
    contradicts the AsyncEventSource documentation for processOne. If
    multi-threading is ever required then accesses to the sources array will
    need to be synchronized in some way.

    Note, that this synchronization mechanism should not hold a lock while
    calling a child's processOne() function. This can easily lead to
    deadlocks. This is the main reason for the existence of the sources queues.

 */
bool
MessageLoop::
processOne()
{
    bool more = false;

    processSourceQueue();

    if (needsPoll) {
        for (unsigned i = 0;  i < sources.size();  ++i) {
            try {
                bool hasMore = sources[i].source->processOne();
                if (debug_)
                    cerr << "source " << sources[i].name << " has " << hasMore << endl;
                more = more || hasMore;
            } catch (...) {
                cerr << "exception processing source " << sources[i].name
                     << endl;
                throw;
            }
        }
    }
    else more = Epoller::processOne();

    return more;
}

void
MessageLoop::
debug(bool debugOn)
{
    debug_ = debugOn;
}

void
MessageLoop::
checkNeedsPoll()
{
    bool newNeedsPoll = false;
    for (unsigned i = 0;  i < sources.size() && !newNeedsPoll;  ++i)
        newNeedsPoll = sources[i].source->needsPoll;

    if (newNeedsPoll == needsPoll) return;

    needsPoll = newNeedsPoll;
    if (parent_) parent_->checkNeedsPoll();
}

} // namespace Datacratic

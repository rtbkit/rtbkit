/** message_loop.cc
    Jeremy Barnes, 1 June 2012
    Copyright (c) 2012 Datacratic.  All rights reserved.

*/

#include <thread>
#include <time.h>
#include <limits.h>
#include <sys/epoll.h>

#include "jml/arch/exception.h"
#include "jml/arch/timers.h"
#include "jml/arch/demangle.h"
#include "jml/arch/futex.h"
#include "jml/arch/backtrace.h"
#include "jml/utils/smart_ptr_utils.h"
#include "jml/utils/exc_assert.h"
#include "soa/types/date.h"
#include "soa/service/logs.h"

#include "message_loop.h"

using namespace std;


namespace Datacratic {

Logging::Category MessageLoopLogs::print("Message Loop");
Logging::Category MessageLoopLogs::warning("Message Loop Warning", print);
Logging::Category MessageLoopLogs::error("Message Loop Error", print);
Logging::Category MessageLoopLogs::trace("Message Loop Trace", print);

typedef MessageLoopLogs Logs;

/*****************************************************************************/
/* MESSAGE LOOP                                                              */
/*****************************************************************************/

MessageLoop::
MessageLoop(int numThreads, double maxAddedLatency, int epollTimeout)
    : sourceActions_([&] () { handleSourceActions(); }),
      numThreadsCreated(0),
      shutdown_(true),
      totalSleepTime_(0.0)
{
    init(numThreads, maxAddedLatency, epollTimeout);
}

MessageLoop::
~MessageLoop()
{
    shutdown();
}

void
MessageLoop::
init(int numThreads, double maxAddedLatency, int epollTimeout)
{
    // std::cerr << "msgloop init: " << this << "\n";
    if (maxAddedLatency == 0 && epollTimeout != -1)
        LOG(Logs::warning)
            << "MessageLoop with maxAddedLatency of zero and "
            << "epollTeimout != -1 will busy wait" << endl;
    
    // See the comments on processOne below for more details on this assertion.
    ExcAssertEqual(numThreads, 1);

    Epoller::init(16384, epollTimeout);
    maxAddedLatency_ = maxAddedLatency;
    handleEvent = std::bind(&MessageLoop::handleEpollEvent,
                            this,
                            std::placeholders::_1);

    /* Our source action queue is a source in itself, which enables us to
       handle source operations from the same epoll mechanism as the rest.

       Adding a special source named "_shutdown" triggers shutdown-related
       events, without requiring the use of an additional signal fd. */
    addFd(sourceActions_.selectFd(), &sourceActions_);

    debug_ = false;
}

void
MessageLoop::
start(const OnStop & onStop)
{
    if (numThreadsCreated)
        throw ML::Exception("already have started message loop");

    shutdown_ = false;

    //cerr << "starting thread from " << this << endl;
    //ML::backtrace();

    auto runfn = [&, onStop] () {
        this->runWorkerThread();
        if (onStop) onStop();
    };

    threads.emplace_back(runfn);

    ++numThreadsCreated;
}
    
void
MessageLoop::
startSync()
{
    if (numThreadsCreated)
        throw ML::Exception("already have started message loop");

    ++numThreadsCreated;

    shutdown_ = false;
    runWorkerThread();
}
    
void
MessageLoop::
shutdown()
{
    if (shutdown_)
        return;

    shutdown_ = true;

    // We could be asleep (in which case we sleep on the shutdown_ futex and
    // will be woken by the futex_wake) or blocked in epoll (in which case
    // we will get the addSource event to wake us up).
    ML::futex_wake(shutdown_);
    addSource("_shutdown", nullptr);

    for (auto & t: threads)
        t.join();
    threads.clear();

    numThreadsCreated = 0;
}

bool
MessageLoop::
addSource(const std::string & name,
          AsyncEventSource & source, int priority)
{
    return addSource(name, ML::make_unowned_std_sp(source), priority);
}

bool
MessageLoop::
addSource(const std::string & name,
          const std::shared_ptr<AsyncEventSource> & source,
          int priority)
{
    if (name != "_shutdown") {
        ExcCheck(!source->parent_, "source already has a parent: " + name);
        source->parent_ = this;
    }

    // cerr << "addSource: " << source.get()
    //      << " (" << ML::type_name(*source) << ")"
    //      << " needsPoll: " << source->needsPoll
    //      << " in msg loop: " << this
    //      << " needsPoll: " << needsPoll
    //      << endl;

    SourceEntry entry(name, source, priority);
    SourceAction newAction(SourceAction::ADD, move(entry));

    return sourceActions_.push_back(move(newAction));
}

bool
MessageLoop::
addPeriodic(const std::string & name,
            double timePeriodSeconds,
            std::function<void (uint64_t)> toRun,
            int priority)
{
    auto newPeriodic
        = make_shared<PeriodicEventSource>(timePeriodSeconds, toRun);
    return addSource(name, newPeriodic, priority);
}

bool
MessageLoop::
removeSource(AsyncEventSource * source)
{
    // When we free the elements in our destructor, they will try to remove themselves.
    // we just make it a nop.
    if (shutdown_) return true;

    SourceEntry entry("", ML::make_unowned_std_sp(*source), 0);
    SourceAction newAction(SourceAction::REMOVE, move(entry));
    return sourceActions_.push_back(move(newAction));
}

bool
MessageLoop::
removeSourceSync(AsyncEventSource * source)
{
    bool r = removeSource(source);
    if (!r) return false;

    while(source->connectionState_ != AsyncEventSource::DISCONNECTED) {
        ML::futex_wait(source->connectionState_, AsyncEventSource::CONNECTED);
    }

    return true;
}

bool
MessageLoop::
runInMessageLoopThread(std::function<void ()> toRun)
{
    SourceEntry entry("", toRun, 0);
    SourceAction newAction(SourceAction::RUN, move(entry));
    return sourceActions_.push_back(move(newAction));
}

void
MessageLoop::
wakeupMainThread()
{
    ML::futex_wake(shutdown_);
}

void
MessageLoop::
startSubordinateThread(const SubordinateThreadFn & thread)
{
    Guard guard(threadsLock);
    int64_t id = 0;
    threads.emplace_back(std::bind(thread, std::ref(shutdown_), id));
}

void
MessageLoop::
runWorkerThread()
{
    Date lastCheck = Date::now();

    ML::Duty_Cycle_Timer duty;

    while (!shutdown_) {
        Date start = Date::now();

        if (debug_) {
            cerr << "handling events from " << sources.size()
                 << " sources with needsPoll " << needsPoll << endl;
            for (unsigned i = 0;  i < sources.size();  ++i)
                cerr << sources[i].name << " " << sources[i].source->needsPoll << endl;
        }

        if (!needsPoll) {
            Date beforeSleepTime;

            // Now we've processed what we can, let's allow a sleep
            auto beforeSleep = [&] ()
                {
                    duty.notifyBeforeSleep();
                    beforeSleepTime = Date::now();
                };

            auto afterSleep = [&] ()
                {
                    double delta  = Date::now().secondsSince(beforeSleepTime);
                    totalSleepTime_ += delta;
                    duty.notifyAfterSleep();
                };

            // Maximum number of events to handle in handleEvents.
            int maxEventsToHandle = 512;

            // First time, we sleep for up to one second waiting for events to come
            // in to the event loop, and handle as many as we can until we hit the
            // limit or we're idle.
            int res JML_UNUSED = handleEvents(999999 /* microseconds */, maxEventsToHandle,
                                              nullptr, beforeSleep, afterSleep);
            //cerr << "handleEvents returned " << res << endl;

#if 0
            while (res != 0) {
                if (shutdown_)
                    return;
                
                // Now we busy loop handling events while there is still more work to do
                res = handleEvents(0 /* microseconds */, maxEventsToHandle,
                                   nullptr, beforeSleep, afterSleep);
            }
#endif
        }

        if (shutdown_)
            return;

        // Do any outstanding work now
        int i = 0;
        while (processOne()) {
            if (shutdown_)
                return;

            if (i >= 50) {
                getrusage(RUSAGE_THREAD, &resourceUsage);
                i = 0;
            }
            i++;
        }

        getrusage(RUSAGE_THREAD, &resourceUsage);

        // At this point, we've done as much work as we can (there is no more
        // work to do).  We will now sleep for the maximum allowable delay
        // time minus the time we spent working.  This allows us to batch up
        // work to be done next time we wake up, rather then waking up all the
        // time to do a little bit of work.  The busier we get, the less time
        // we will sleep for until when we're completely busy we don't sleep
        // at all.
        Date end = Date::now();

        double elapsed = end.secondsSince(start);
        double sleepTime = maxAddedLatency_ - elapsed;

        duty.notifyBeforeSleep();
        if (sleepTime > 0) {
            ML::futex_wait(shutdown_, 0, sleepTime);
            totalSleepTime_ += sleepTime;
        }
        duty.notifyAfterSleep();
        
        if (lastCheck.secondsUntil(end) > 10.0) {
            // auto stats = duty.stats();
            //cerr << "message loop: wakeups " << stats.numWakeups
            //     << " duty " << stats.duty_cycle() << endl;
            lastCheck = end;
            duty.clear();
        }
    }
}

Epoller::HandleEventResult
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
    
    if (debug) {
        ExcAssert(source->poll());
        cerr << "message loop " << this << " with parent " << parent_
             << " handing source " << ML::type_name(*source) << " poll result "
             << Epoller::poll() << endl;
    }

    int res = source->processOne();

    if (debug) {
        cerr << "source " << ML::type_name(*source) << " had processOne() result " << res << endl;
        cerr << "poll() is now " << Epoller::poll() << endl;
    }

    return Epoller::DONE;
}

void
MessageLoop::
handleSourceActions()
{
    vector<SourceAction> actions = sourceActions_.pop_front(0);
    for (auto & action: actions) {
        if (action.action_ == SourceAction::ADD) {
            processAddSource(action.entry_);
        }
        else if (action.action_ == SourceAction::REMOVE) {
            processRemoveSource(action.entry_);
        }
        else if (action.action_ == SourceAction::RUN) {
            processRunAction(action.entry_);
        }
    }
}

void
MessageLoop::
processAddSource(const SourceEntry & entry)
{
    if (entry.name == "_shutdown")
        return;

    // cerr << "processAddSource: " << entry.source.get()
    //      << " (" << ML::type_name(*entry.source) << ")"
    //      << " needsPoll: " << entry.source->needsPoll
    //      << " in msg loop: " << this
    //      << " needsPoll: " << needsPoll
    //      << endl;
    int fd = entry.source->selectFd();
    if (fd != -1)
        addFd(fd, entry.source.get());

    if (!needsPoll && entry.source->needsPoll) {
        needsPoll = true;
        if (parent_) parent_->checkNeedsPoll();
    }

    if (debug_) entry.source->debug(true);
    sources.push_back(entry);

    if (needsPoll) {
        string pollingSources;
        
        for (auto & s: sources) {
            if (s.source->needsPoll) {
                if (!pollingSources.empty())
                    pollingSources += ", ";
                pollingSources += s.name;
            }
        }
        
        double wakeupsPerSecond = 1.0 / maxAddedLatency_;
        
        LOG(Logs::warning)
            << "message loop in polling mode will cause " << wakeupsPerSecond
            << " context switches per second due to polling on sources "
            << pollingSources << endl;
    }

    entry.source->connectionState_ = AsyncEventSource::CONNECTED;
    ML::futex_wake(entry.source->connectionState_);
}

void
MessageLoop::
processRemoveSource(const SourceEntry & rmEntry)
{
    auto pred = [&] (const SourceEntry & entry) {
        return entry.source.get() == rmEntry.source.get();
    };
    auto it = find_if(sources.begin(), sources.end(), pred);

    ExcCheck(it != sources.end(), "couldn't remove source");

    SourceEntry entry = *it;
    sources.erase(it);

    entry.source->parent_ = nullptr;
    int fd = entry.source->selectFd();
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

    entry.source->connectionState_ = AsyncEventSource::DISCONNECTED;
    ML::futex_wake(entry.source->connectionState_);
}

void
MessageLoop::
processRunAction(const SourceEntry & entry)
{
    entry.run();
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

    // NOTE: this is required for some buggy sources that don't have a reliable FD to
    // sleep on.  It shouldn't be substantially less efficient.
    if (needsPoll || true) {
        more = sourceActions_.processOne();

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

/* message_loop.h                                                  -*- C++ -*-
   Jeremy Barnes, 31 May 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Base class for loop that listens for various types of messages.
*/

#pragma once

#include "epoller.h"
#include "async_event_source.h"
#include "jml/arch/wakeup_fd.h"
#include "jml/arch/spinlock.h"

#include <boost/thread/thread.hpp>
#include <functional>
#include <mutex>
#include <atomic>


namespace Datacratic {


/*****************************************************************************/
/* MESSAGE LOOP                                                              */
/*****************************************************************************/

struct MessageLoop : public Epoller {
    
    MessageLoop(int numThreads = 1, double maxAddedLatency = 0.0005,
                int epollTimeout = 0);
    ~MessageLoop();

    void init(int numThreads = 1, double maxAddedLatency = 0.0005,
              int epollTimeout = 0);

    void start(std::function<void ()> onStop = std::function<void ()>());

    void startSync();
    
    //void sleepUntilIdle();

    void shutdown();

    /** Add the given source of asynchronous wakeups with the given
        callback to be run when they trigger.

        Note that this function call will not take effect immediately. All work
        is deferred to the main message loop thread.
    */
    void addSource(const std::string & name,
                   AsyncEventSource & source,
                   int priority = 0);

    /** Add the given source of asynchronous wakeups with the given
        callback to be run when they trigger.

        Note that this function call will not take effect immediately. All work
        is deferred to the main message loop thread.
    */
    void addSource(const std::string & name,
                   std::shared_ptr<AsyncEventSource> source,
                   int priority = 0);

    /** Add a periodic job to be performed by the loop.  The number passed
        to the toRun function is the number of timeouts that have elapsed
        since the last call; this is useful to know if something has
        got behind.  It will normally be 1.

        Note that this function call will not take effect immediately. All work
        is deferred to the main message loop thread.
    */
    void addPeriodic(const std::string & name,
                     double timePeriodSeconds,
                     std::function<void (uint64_t)> toRun,
                     int priority = 0);
    
    typedef std::function<void (volatile int & shutdown_,
                                int64_t threadId)> SubordinateThreadFn;

    /** Start a subordinate thread that runs the given function,
        returning when the passed parameter is non-zero, and manage
        its lifecycle with this thread.
    */
    void startSubordinateThread(const SubordinateThreadFn & mainFn);

    virtual bool processOne();

    virtual bool poll() const;

    /** Remove the given source from the list of active sources.

        Note that this function call will not take effect immediately. All work
        is deferred to the main message loop thread.
     */
    void removeSource(AsyncEventSource * source);

    /** Re-check if anything needs to poll. */
    void checkNeedsPoll();

    /** Total number of seconds that this message loop has spent sleeping.
        Can be polled regularly to determine the duty cycle of the loop.
     */
    double totalSleepSeconds() const { return totalSleepTime_; }

    void debug(bool debugOn);
    
private:
    void runWorkerThread();
    
    void wakeupMainThread();

    /** Implementation of addSource that runs without taking the lock. */
    void addSourceImpl(const std::string & name,
                       std::shared_ptr<AsyncEventSource> source,
                       int priority);

    typedef ML::Spinlock Lock;
    typedef std::lock_guard<Lock> Guard;

    struct SourceEntry
    {
        SourceEntry(
                const std::string& name,
                std::shared_ptr<AsyncEventSource> source,
                int priority) :
            name(name), source(source), priority(priority)
        {}

        std::string name;
        std::shared_ptr<AsyncEventSource> source;
        int priority;
    };

    std::vector<SourceEntry> sources;

    mutable Lock queueLock;
    std::vector<SourceEntry> addSourceQueue;
    std::vector<AsyncEventSource*> removeSourceQueue;

    // Notifies the main loop that a new source event needs processing.
    ML::Wakeup_Fd queueFd;

    /** Flag used to notify the main thread that one of the source queues has an
        event. Thread-safe on reads but writes must occur while hold queueLock
        to avoid lost notifications.
     */
    std::atomic<bool> sourceQueueFlag;

    void processRemoveSource(AsyncEventSource* source);
    void processAddSource(const SourceEntry& entry);
    void processSourceQueue();

    Lock threadsLock;
    int numThreadsCreated;
    boost::thread_group threads;
    
    /** Global flag to shutdown. */
    volatile int shutdown_;

    /** Global flag for idle. */
    volatile int idle_;

    /** Do we debug? */
    bool debug_;

    /** Number of secs that the message loop has spent sleeping. */
    double totalSleepTime_;

    /** Number of seconds of latency we're allowed to add in order to reduce
        the number of context switches.
    */
    double maxAddedLatency_;

    bool handleEpollEvent(epoll_event & event);
};

} // namespace Datacratic

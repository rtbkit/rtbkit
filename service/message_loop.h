/* message_loop.h                                                  -*- C++ -*-
   Jeremy Barnes, 31 May 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Base class for loop that listens for various types of messages.
*/

#pragma once

#include "async_event_source.h"
#include <boost/thread/thread.hpp>
#include <mutex>
#include <functional>
#include <mutex>
#include "soa/service/epoller.h"


namespace Datacratic {



/*****************************************************************************/
/* MESSAGE LOOP                                                              */
/*****************************************************************************/



struct MessageLoop : public Epoller {
    
    MessageLoop(int numThreads = 1, double maxAddedLatency = 0.0005);
    ~MessageLoop();

    void init(int numThreads = 1, double maxAddedLatency = 0.0005);

    void start(std::function<void ()> onStop = std::function<void ()>());

    void startSync();
    
    //void sleepUntilIdle();

    void shutdown();

    /** Add the given source of asynchronous wakeups with the given
        callback to be run when they trigger.

        This method cannot be called from within an event processed by the
        message loop (that would deadlock); instead use addPeriodicDeferred().
    */
    void addSource(const std::string & name,
                   AsyncEventSource & source,
                   int priority = 0);

    /** Add the given source of asynchronous wakeups with the given
        callback to be run when they trigger.

        This method cannot be called from within an event processed by the
        message loop (that would deadlock); instead use addPeriodicDeferred().
    */
    void addSource(const std::string & name,
                   std::shared_ptr<AsyncEventSource> source,
                   int priority = 0);

    /** Add a periodic job to be performed by the loop.  The number passed
        to the toRun function is the number of timeouts that have elapsed
        since the last call; this is useful to know if something has
        got behind.  It will normally be 1.

        This method cannot be called from within an event processed by the
        message loop (that would deadlock); instead use addPeriodicDeferred().
    */
    void addPeriodic(const std::string & name,
                     double timePeriodSeconds,
                     std::function<void (uint64_t)> toRun,
                     int priority = 0);

    /** Add the given source of asynchronous wakeups with the given
        callback to be run when they trigger.

        This method can only be called from within an event processed by the
        message loop; the handler will actually be added after the next
        event finishes processing.
    */
    void addSourceDeferred(const std::string & name,
                           AsyncEventSource & source,
                           int priority = 0);

    /** Add the given source of asynchronous wakeups with the given
        callback to be run when they trigger.

        This method can only be called from within an event processed by the
        message loop; the handler will actually be added after the next
        event finishes processing.
    */
    void addSourceDeferred(const std::string & name,
                           std::shared_ptr<AsyncEventSource> source,
                           int priority = 0);

    /** Add a periodic job to be performed by the loop.  The number passed
        to the toRun function is the number of timeouts that have elapsed
        since the last call; this is useful to know if something has
        got behind.  It will normally be 1.

        This method can only be called from within an event processed by the
        message loop; the handler will actually be added after the next
        event finishes processing.
    */
    void addPeriodicDeferred(const std::string & name,
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

    void debug(bool debugOn);

    /** Remove the given source from the list of active sources. */
    void removeSource(AsyncEventSource * source);

    /** Re-check if anything needs to poll. */
    void checkNeedsPoll();

    /** Total number of seconds that this message loop has spent sleeping.
        Can be polled regularly to determine the duty cycle of the loop.
     */
    double totalSleepSeconds() const { return totalSleepTime_; }
    
private:
    void runWorkerThread();
    
    void wakeupMainThread();

    /** Implementation of addSource that runs without taking the lock. */
    void addSourceImpl(const std::string & name,
                       std::shared_ptr<AsyncEventSource> source, int priority);
    
    //ML::Wakeup_Fd wakeup;
    
    std::vector<std::pair<std::string, std::shared_ptr<AsyncEventSource> > > sources;

    /** Set of deferred sources.  These will be added once the current events
        have finished processing.
    */
    std::vector<std::tuple<std::string, std::shared_ptr<AsyncEventSource>, int > > deferredSources;

    int numThreadsCreated;
    boost::thread_group threads;
    
    // TODO: bad, bad, bad... make an API whereby we can ask this to happen later
    // so we don't need a recursive mutex
    //typedef std::recursive_mutex Lock;

    typedef std::mutex Lock;
    typedef std::unique_lock<Lock> Guard;
    mutable Lock lock;
    
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

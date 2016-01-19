/* carbon_connector.h                                            -*- C++ -*-
   Jeremy Barnes, 3 August 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Class to accumulate operational stats and connect to carbon (directly).
*/

#pragma once

#include "soa/service/stat_aggregator.h"
#include "soa/service/stats_events.h"
#include "ace/INET_Addr.h"
#include "jml/stats/distribution.h"
#include "soa/types/date.h"
#include <unordered_map>
#include <map>
#include <memory>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <boost/thread.hpp>


namespace Datacratic {


/*****************************************************************************/
/* MULTI AGGREGATOR                                                          */
/*****************************************************************************/

/** Aggregates multiple stats together. */

struct MultiAggregator {
    MultiAggregator();

    typedef std::function<void (const std::vector<StatReading> &)>
        OutputFn;

    MultiAggregator(const std::string & path,
                    const OutputFn & output = OutputFn(),
                    double dumpInterval = 1.0,
                    std::function<void ()> onStop
                        = std::function<void ()>());
    
    ~MultiAggregator();

    void open(const std::string & path,
              const OutputFn & output = OutputFn(),
              double dumpInterval = 1.0,
              std::function<void ()> onStop
                  = std::function<void ()>());

    OutputFn outputFn;

    /** Function to be called when the stat is to be done.  Default will
        call the OutputFn.
    */
    virtual void doStat(const std::vector<StatReading> & value) const;

    /** Record, generic version. */
    void record(const std::string & stat,
                StatEventType type = ET_COUNT,
                float value = 1.0,
                std::initializer_list<int> extra = DefaultOutcomePercentiles);

    /** Simplest interface: record that a particular event occurred.  The
        stat will record the total count for each second.  Lock-free and
        thread safe.
    */
    void recordHit(const std::string & stat);

    /** Record that something happened.  The stat will record the total amount
        in each second.
    */
    void recordCount(const std::string & stat, float quantity);

    /** Record the value of a something. THe stat will record the mean of that
        value over a second.

        Lock-free (except the first time it's called for each name) and thread
        safe.
     */
    void recordStableLevel(const std::string & stat, float value);

    /** Record the level of a something.  The stat will record the mean, minimum
        and maximum level over the second.

        Lock-free (except the first time it's called for each name) and
        thread safe.
    */
    void recordLevel(const std::string & stat, float value);
    
    /** Record that a given value from an independent process. The stat will
        record the mean, mininum, maxixmum outcome over the second as well as
        the percentiles specified by the last argument, defaulting to the
        90th, 95th and 98th percentiles and the number of outcomes.

        Lock-free (except the first time it's called for each name) and thread
        safe.
    */
    void recordOutcome(const std::string & stat, float value,
            const std::vector<int>& percentiles = DefaultOutcomePercentiles);

    /** Dump synchronously (taking the lock).  This should only be used in
        testing or debugging, not when connected to Carbon.
    */
    void dumpSync(std::ostream & stream) const;

    /** Wake up and dump the data.  Asynchronous; this signals the dumping
        thread to do the actual dump.
    */
    void dump();

    /** Stop the dumping. */
    void stop();
    
protected:
    // Prefix to add to each stat to put it in its namespace
    std::string prefix;

    // Function to call when it's stopped/shutdown
    std::function<void ()> onStop;

    // Functions to implement the shutdown
    std::function<void ()> onPreShutdown, onPostShutdown;

private:
    // This map can only have things removed from it, never added to
    typedef std::map<std::string, std::shared_ptr<StatAggregator> > Stats;
    Stats stats;

    // R/W mutex for reading/writing stats.  Read to look up, write to
    // add a new stat.
    typedef std::mutex Lock;
    mutable Lock lock;

    typedef std::unordered_map<std::string, Stats::iterator> LookupCache;

    // Cache of lookups for each thread to avoid needing to acquire a lock
    // very much.
    boost::thread_specific_ptr<LookupCache> lookupCache;

    /** Thread that's started up to start dumping. */
    void runDumpingThread();

    /** Shutdown everything. */
    void shutdown();

    /** Look for the aggregator for this given stat.  If it doesn't exist,
        then initialize it from the given function.
    */
    template<typename... Args>
    StatAggregator & getAggregator(const std::string & stat,
                                   StatAggregator * (*createFn) (Args...),
                                   Args&&... args)
    {
        if (!lookupCache.get())
            lookupCache.reset(new LookupCache());

        auto found = lookupCache->find(stat);
        if (found != lookupCache->end())
            return *found->second->second;

        // Get the read lock to look for the aggregator
        std::unique_lock<Lock> guard(lock);

        auto found2 = stats.find(stat);

        if (found2 != stats.end()) {
            guard.unlock();

            (*lookupCache)[stat] = found2;

            return *found2->second;
        }

        guard.unlock();

        // Get the write lock to add it to the aggregator
        std::unique_lock<Lock> guard2(lock);

        // Add it in
        found2 = stats.insert(
                make_pair(stat, std::shared_ptr<StatAggregator>(createFn(std::forward<Args>(args)...)))).first;

        guard2.unlock();
        (*lookupCache)[stat] = found2;
        return *found2->second;
    }
    
    std::unique_ptr<std::thread> dumpingThread;

    std::condition_variable cond;  // to wake up dumping thread
    std::mutex m;
    std::atomic<bool> doShutdown;                 // thread woken up to shutdown
    std::atomic<bool> doDump;                     // thread woken up to dump

    /** How many seconds to wait before we dump.  If set to zero, dumping
        is only done on demand.
    */
    double dumpInterval;
};


/*****************************************************************************/
/* CARBON CONNECTOR                                                        */
/*****************************************************************************/

/** A connector to carbon that pings it once/second and accumulates
    statistics apart from that.
*/

struct CarbonConnector : public MultiAggregator {
    CarbonConnector();

    CarbonConnector(const std::string & carbonAddr,
                    const std::string & path,
                    double dumpInterval = 1.0,
                    std::function<void ()> onStop
                        = std::function<void ()>());

    CarbonConnector(const std::vector<std::string> & carbonAddrs,
                    const std::string & path,
                    double dumpInterval = 1.0,
                    std::function<void ()> onStop
                        = std::function<void ()>());

    ~CarbonConnector();

    void open(const std::string & carbonAddr,
              const std::string & path,
              double dumpInterval = 1.0,
              std::function<void ()> onStop
                  = std::function<void ()>());

    void open(const std::vector<std::string> & carbonAddrs,
              const std::string & path,
              double dumpInterval = 1.0,
              std::function<void ()> onStop
                  = std::function<void ()>());

    /** Override for doStat to send it over to Carbon. */
    virtual void doStat(const std::vector<StatReading> & value) const;

private:
    // Do our own internal shutdown
    void doShutdown();

    struct Connection {

        Connection(const std::string & addr)
            : fd(-1), addr(addr), shutdown(0)
        {
        }

        ~Connection();

        /** Attempt to connect.  If after this call fd is -1, then it failed
            and the error message is returned.  Otherwise it succeeded and
            the return value should be ignored.
        */
        std::string connect();
        void reconnect();

        void send(const std::string & message);

        void close();

        /** Function that runs in a thread when we're trying to reconenct. */
        void runReconnectionThread();

        // Networky things
        int fd;
        std::string addr;
        ACE_INET_Addr ip;

        // Reconnection thread if we are trying to reconnect
        std::shared_ptr<std::thread> reconnectionThread;
        bool reconnectionThreadActive;
        bool reconnectionThreadJoinable;
        int shutdown;
    };

    std::vector<std::shared_ptr<Connection> > connections;
};


} // namespace Datacratic

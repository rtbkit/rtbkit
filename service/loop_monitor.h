 /** loop_monitor.h                                 -*- C++ -*-
    Rémi Attab, 06 May 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Monitor for message loop that can be used to track the load of a set of
    message loops.
*/

#pragma once

#include "message_loop.h"
#include "service_base.h"
#include "jml/arch/spinlock.h"
#include "jml/utils/rng.h"

#include <map>

namespace Datacratic {


/******************************************************************************/
/* LOOP MONITOR                                                               */
/******************************************************************************/

struct LoopMonitor : public ServiceBase, public MessageLoop
{
    LoopMonitor(
            const std::shared_ptr<ServiceProxies> services
                = std::make_shared<ServiceProxies>(),
            const std::string& name = "loopMonitor");

    LoopMonitor(
            ServiceBase& parent, const std::string& name = "loopMonitor");

    void init(double updatePeriod = 1.0);


    /** Update frequency of the monitor. */
    double getUpdatePeriod() const { return updatePeriod; }


    /** Called every updatePeriod seconds to sample the load factor of a loop
        where 0 is completely idle and 1 is always processing.
     */
    typedef std::function<double(double elapsedTime)> SampleLoadFn;

    /** Adds a sampling function for a MessageLoop which will be called every
        updatePeriod. Thread-safe.
     */
    void addMessageLoop(const std::string& name, const MessageLoop* loop);

    /** Adds a load sampling function which will be called every updatePeriod.
        Thread-safe.
     */
    void addCallback(const std::string& name, const SampleLoadFn& cb);

    /** Remove the monitoring of a loop associated with the given name.
        Thread-safe.
     */
    void remove(const std::string& name);


    /** Atomically writable and readable container for the load. */
    struct LoadSample
    {
        LoadSample() : load(0.0), sequence(0) {}
        explicit LoadSample(uint64_t packed) : packed(packed) {}

        union {
            struct {
                float load;        ///< load factor of a loop
                uint32_t sequence; ///< incremented everytime load changes
            } JML_PACKED;

            uint64_t packed;       ///< used for atomic rw the struct
        };
    };

    /** Returns the maximum load factor sampled of all the loops. Since this is
        sample periodically, the sequence number in the Load struct will be
        incremented everytime the load is updated.
     */
    LoadSample sampleLoad() const { return LoadSample(curLoad.packed); }

    /** Called whenever the value returned by sampleLoad changes. */
    std::function<void(double load)> onLoadChange;

private:

    void doLoops(uint64_t numTimeouts);

    double updatePeriod;

    mutable ML::Spinlock lock;
    std::map<std::string, SampleLoadFn> loops;

    LoadSample curLoad;
};


/******************************************************************************/
/* LOAD STABILIZER                                                            */
/******************************************************************************/

/** Simple load shedding that attempts to stabilize the load of the system at
    around a threshold by randomly dropping messages with a probability that
    is adjusted over time with the system's load.
 */
struct LoadStabilizer
{
    LoadStabilizer(const LoopMonitor& loopMonitor);

    void setLoadThreshold(double val = 0.9) { loadThreshold = val; }

    /** Returns true if a message should be dropped to help the system deal with
        excessive load.

        Thread-safe and lock-free.
     */
    bool shedMessage()
    {
        double prob = shedProbability();
        return prob == 0.0 ? false : rng.random01() < shedProb;
    }

    /** Returns the probability at which a message should be dropped to help the
        system deal with excessive load.

        Thread-safe and lock-free.
     */
    double shedProbability()
    {
        auto sample = loopMonitor.sampleLoad();
        if (sample.sequence != lastSample.sequence)
            updateProb(sample);

        return shedProb <= 0.01 ? 0.0 : shedProb;
    }

private:

    void updateProb(LoopMonitor::LoadSample load);

    const LoopMonitor& loopMonitor;

    double loadThreshold;
    double shedProb;
    ML::RNG rng;

    LoopMonitor::LoadSample lastSample;
};


} // namespace Datacratic

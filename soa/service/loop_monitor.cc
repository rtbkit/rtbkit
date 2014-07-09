/** loop_monitor.cc                                 -*- C++ -*-
    Rémi Attab, 06 May 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Implementation details of the message loop monitor.

*/

#include "loop_monitor.h"
#include "jml/arch/cmp_xchg.h"

#include <mutex>
#include <functional>

using namespace std;
using namespace ML;

namespace Datacratic {


/******************************************************************************/
/* LOOP MONITOR                                                               */
/******************************************************************************/

LoopMonitor::
LoopMonitor(const shared_ptr<ServiceProxies> services, const string& name) :
    ServiceBase(name, services)
{}

LoopMonitor::
LoopMonitor(ServiceBase& parent, const string& name) :
    ServiceBase(name, parent)
{}

void
LoopMonitor::
init(double updatePeriod)
{
    this->updatePeriod = updatePeriod;
    addPeriodic("LoopMonitor", updatePeriod,
            std::bind(&LoopMonitor::doLoops, this, placeholders::_1));
}

void
LoopMonitor::
doLoops(uint64_t numTimeouts)
{
    std::lock_guard<ML::Spinlock> guard(lock);

    LoadSample maxLoad;
    maxLoad.sequence = curLoad.sequence + 1;
    for (auto& loop : loops) {
        double load = loop.second(updatePeriod * numTimeouts);
        if (load < 0.0 || load > 1.0) {
            stringstream ss;
            ss << "WARNING: LoopMonitor." << loop.first << ": "  << load
                << " - ignoring the value"
                << endl;
            cerr << ss.str();
            continue;
        }

        if (load > maxLoad.load) maxLoad.load = load;
        recordLevel(load, loop.first);
    }

    curLoad.packed = maxLoad.packed;
    if (onLoadChange) onLoadChange(maxLoad.load);
}

void
LoopMonitor::
addMessageLoop(const string& name, const MessageLoop* loop)
{
    // acts as a private member variable for sampleFn.
    double lastTimeSlept = 0.0;

    auto sampleFn = [=] (double elapsedTime) mutable {
        double timeSlept = loop->totalSleepSeconds();
        double delta = std::min(timeSlept - lastTimeSlept, 1.0);
        lastTimeSlept = timeSlept;

        return 1.0 - (delta / elapsedTime);
    };

    addCallback(name, sampleFn);
}

void
LoopMonitor::
addCallback(const string& name, const SampleLoadFn& cb)
{
    std::lock_guard<ML::Spinlock> guard(lock);

    auto ret = loops.insert(make_pair(name, cb));
    ExcCheck(ret.second, "loop already being monitored: " + name);
}

void
LoopMonitor::
remove(const string& name)
{
    std::lock_guard<ML::Spinlock> guard(lock);

    size_t ret = loops.erase(name);
    ExcCheckEqual(ret, 1, "loop is not monitored: " + name);
}


/******************************************************************************/
/* LOAD STABILIZER                                                            */
/******************************************************************************/

LoadStabilizer::
LoadStabilizer(const LoopMonitor& loopMonitor) :
    loopMonitor(loopMonitor),
    loadThreshold(0.9),
    shedProb(0.0)
{}

void
LoadStabilizer::
updateProb(LoopMonitor::LoadSample sample)
{
    // Ensures that only the first thread to get past this point will be able to
    // update the shedProb.
    auto oldSample = lastSample;
    if (sample.sequence == oldSample.sequence) return;
    if (!ML::cmp_xchg(lastSample.packed, oldSample.packed, sample.packed))
        return;

    // Don't drop too low otherwise it'll take forever to raise the prob.
    if (sample.load < loadThreshold)
        shedProb = std::max(0.01, shedProb - 0.01);

    // Should drop faster then it raises so that we're responsive to load spikes
    else shedProb = std::min(1.0, shedProb + 0.05);
}


} // namepsace Datacratic

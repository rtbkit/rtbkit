/** thread_test.h                                 -*- C++ -*-
    Rémi Attab, 26 Jul 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Multithreaded test implementation.

*/

#include "threaded_test.h"
#include "future_fix.h"
#include "jml/arch/timers.h"
#include "jml/utils/exc_check.h"

#include <atomic>

using namespace std;
using namespace ML;

namespace Datacratic {


/******************************************************************************/
/* THREADED TEST                                                              */
/******************************************************************************/

void
ThreadedTest::
start(const TestFn& testFn, int threadCount, int group)
{
    {
        lock_guard<mutex> guard(lock);
        promises[group].resize(threadCount);
    }

    for (int id = 0;  id < threadCount;  ++id) {
        threads[group].create_thread([=] {
                    int r = testFn(id);
                    lock_guard<mutex> guard(this->lock);
                    this->promises[group][id].set_value(r);
                });
    }

}

int
ThreadedTest::
join(int group, uint64_t timeout)
{
    chrono::duration<int, milli> dur(timeout);

    int r = 0;
    for (int id = 0; id < promises[group].size(); ++id) {
        future<int> future = promises[group][id].get_future();
        ExcCheck(wait_for(future, dur), "Test thread is timed out");
        r += future.get();
    }

    threads[group].join_all();
    return r;
}

int
ThreadedTest::
joinAll(uint64_t timeout)
{
    int r = 0;
    for (auto it = threads.begin(), end = threads.end(); it != end; ++it)
        r += join(it->first, timeout);
    return r;
}



/******************************************************************************/
/* TIMED THREADED TEST                                                        */
/******************************************************************************/


void
ThreadedTimedTest::
run(unsigned durationMs)
{
    latencies.resize(configs.size());
    throughputs.resize(configs.size());

    typedef pair<size_t, double> RetValue;
    vector< vector< future< RetValue > > > futures;
    futures.resize(configs.size());
    for (size_t th = 0; th < futures.size(); ++th)
        futures[th].reserve(get<1>(configs[th]));


    atomic<bool> isDone(false);

    auto doTask = [&] (TestFn fn, unsigned id) -> RetValue {
        Timer tm;

        size_t ops = 0;
        for (unsigned it = 0; !isDone; ++it)
            ops += fn(id, it);

        return make_pair(ops, tm.elapsed_wall());
    };

    Timer tm;

    // Launch the threads round-robin style.
    size_t id = 0;
    for (size_t th = 0; th < configs.size(); ++th) {
        const TestFn& fn  = get<0>(configs[th]);
        unsigned& thCount = get<1>(configs[th]);

        for (size_t i = 0; i < thCount; ++i) {
            packaged_task<RetValue(TestFn, unsigned)> task(doTask);
            // I hope this does some actual multi-threading...
            futures[th].emplace_back(async(launch::async, doTask, fn, id));
            id++;
        }
    }


    this_thread::sleep_for(chrono::milliseconds(durationMs));
    isDone = true;

    // Wait for each thread to finish and gather the time stats.
    size_t remaining;
    do {
        remaining = 0;
        unsigned processed = 0;

        for (size_t i = 0; i < futures.size(); ++i) {
            for (size_t j = 0; j < futures[i].size(); ++j) {
                auto& future = futures[i][j];

                if (!future.valid()) continue;
                if (!Datacratic::wait_for(future, chrono::seconds(0))) {
                    remaining++;
                    continue;
                }

                double elapsed = tm.elapsed_wall();
                double latency;
                int ops;
                tie(ops, latency) = future.get();

                latencies[i].push_back(latency / ops);
                throughputs[i].push_back(ops / elapsed);

                processed++;
            }
        }

        if (!processed) this_thread::yield();

    } while (remaining > 0);
}


} // namepsace Datacratic

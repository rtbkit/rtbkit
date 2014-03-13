/** threaded_test.h                                 -*- C++ -*-
    Rémi Attab, 26 Jul 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Multithreaded test utility.

*/

#pragma once

#include "jml/stats/distribution.h"

#include <boost/ptr_container/ptr_map.hpp>
#include <boost/ptr_container/ptr_vector.hpp>
#include <boost/thread.hpp>
#include <functional>
#include <future>
#include <mutex>
#include <string>
#include <chrono>


namespace Datacratic {

/*****************************************************************************/
/* THREADED TEST                                                             */
/*****************************************************************************/

/** Multithreading test helper.

    Used to quickly and simply launch a functor/lambda on multiple threads and
    join on them with a timeout.
*/
struct ThreadedTest
{

    // definition of the functor/lambda.
    typedef std::function<int(int)> TestFn;

    ThreadedTest() {}
    virtual ~ThreadedTest () {}

    ThreadedTest(const ThreadedTest&) = delete;
    ThreadedTest& operator=(const ThreadedTest&) = delete;


    /** Starts the provided lambda on the given number of threads.

        If multiple functions are going to be called during the same test, a
        unique group identifier should be passed in the group parameter.  This
        id will be used during the join operation (explicitly with join() or
        implicity with joinAll()).
    */
    void start(const TestFn& testFn, int threadCount, int group = 1);

    /** Joins the threads represented by the given group with a timeout in
        milliseconds.

        Must be called from the same thread that called start().
    */
    int join(int group = 1, uint64_t timeout = 1000);


    /** Joins the threads represented by the given group with a timeout in
        milliseconds.

        Must be called from the same thread that called start().
    */
    int joinAll(uint64_t timeout = 1000);

private:

    std::mutex lock;

    // This ptr silliness is required to avoid calling copy constructors
    // and still get RAII
    boost::ptr_map<int, boost::ptr_vector<std::promise<int> > > promises;
    boost::ptr_map<int, boost::thread_group> threads;

};



/******************************************************************************/
/* THREADED TIMED TEST                                                        */
/******************************************************************************/

struct ThreadedTimedTest
{
    typedef std::function<size_t(unsigned, unsigned)> TestFn;

    unsigned add(const TestFn& fn, unsigned threadCount)
    {
        configs.emplace_back(fn, threadCount);
        return configs.size() - 1;
    }

    typedef ML::distribution<double> Dist;

    std::pair<Dist, Dist> distributions(unsigned gr)
    {
        return std::make_pair(latencies[gr], throughputs[gr]);
    }

    void run(unsigned durationMs);

private:
    std::vector< std::tuple<TestFn, unsigned> > configs;

    std::vector<Dist> latencies;
    std::vector<Dist> throughputs;

};


} // namespace Datacratic

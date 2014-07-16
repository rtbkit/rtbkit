/** benchmarks.h                                                   -*- C++ -*-
    Wolfgang Sourdeau, 13 April 2014
    Copyright (c) 2014 Datacratic Inc.  All rights reserved.

    Simple utility class to benchmark operations easily.

    Possible improvements:
    - statistics
*/

#pragma once

#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "soa/types/date.h"


namespace Datacratic {

struct Benchmark;


/****************************************************************************/
/* BENCHMARKS                                                               */
/****************************************************************************/

/* A simple collector class that registers a set of tags and accumulate time
   deltas. */

struct Benchmarks {
    void collectBenchmark(const std::vector<std::string> & tags,
                          double delta) noexcept;

    void dumpTotals(std::ostream & ostream = std::cerr);
    void clear();

    typedef std::mutex Lock;
    typedef std::unique_lock<Lock> Guard;

    Lock dataLock_;
    std::map<std::string, double> data_;
};


/****************************************************************************/
/* BENCHMARK                                                                */
/****************************************************************************/

/* A helper class that collects the time delta for a specific task, and
   registers that delta for the associated tags to a central "Benchmarks"
   instance.
   It makes use of RAII, and thus starts counting time at instantiation and
   stops when destroyed. */

struct Benchmark {
    Benchmark(Benchmarks & bInstance, const std::string & tag)
        : bInstance_(bInstance), tags_({tag}), start_(Date::now())
    {}

    Benchmark(Benchmarks & bInstance, const std::vector<std::string> & tags)
        : bInstance_(bInstance), tags_(tags), start_(Date::now())
    {}

    Benchmark(Benchmarks & bInstance,
              const std::initializer_list<std::string> & tags)
        : bInstance_(bInstance), tags_(tags), start_(Date::now())
    {}

    ~Benchmark()
    {
        reportBm();
    }

    void reportBm() noexcept
    {
        double delta = Date::now() - start_;
        bInstance_.collectBenchmark(tags_, delta);
    }

    Benchmarks & bInstance_;
    std::vector<std::string> tags_;
    Date start_;
};

} // namespace Datacratic

#include "benchmarks.h"

using namespace std;
using namespace Datacratic;


/* BENCHMARKS */

void
Benchmarks::
collectBenchmark(const vector<string> & tags, double delta)
    noexcept
{
    Guard lock(dataLock_);
    // fprintf(stderr, "benchmark: %s took %f s.\n",
    //         label.c_str(), delta);
    for (const string & tag: tags) {
        data_[tag] += delta;
    }
}

void
Benchmarks::
dumpTotals(ostream & out)
{
    Guard lock(dataLock_);

    string result("Benchmark totals:\n");
    for (const auto & entry: data_) {
        result += ("  " + entry.first
                   + ": " + to_string(entry.second)
                   + " s.\n");
    }

    out << result;
}

void
Benchmarks::
clear()
{
    Guard lock(dataLock_);
    data_.clear();
}

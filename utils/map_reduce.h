/** map_reduce.h                                                   -*- C++ -*-
    Jeremy Barnes, 30 October 2012
    Copyright (c) 2012 Datacratic Inc.  All rights reserved.

    Functionality to perform map/reduce type operations.
*/


#ifndef __utils__map_reduce_h__
#define __utils__map_reduce_h__

#include "worker_task.h"

namespace ML {

template<typename MapFn, typename ReduceFn, typename It, typename It2>
void
parallelMapInOrderReduce(It first, It2 last, MapFn map, ReduceFn reduce)
{
    // Result type of map function (to be passed to reduce)
    typedef decltype(map(first)) MapResult;

    It next = first;
    std::map<It, MapResult> writeQueue;
    boost::mutex lock;

    auto drainWriteQueue = [&] ()
        {
            while (!writeQueue.empty()
                   && writeQueue.begin()->first == next) {
                reduce(writeQueue.begin()->first,
                       writeQueue.begin()->second);
                writeQueue.erase(writeQueue.begin());
                ++next;
            }
        };

    auto doMap = [&] (It it)
        {
            auto res = map(it);

            boost::unique_lock<boost::mutex> guard(lock);
            writeQueue[it] = std::move(res);
            drainWriteQueue();
        };

    ML::run_in_parallel_blocked(first, last, doMap);

    drainWriteQueue();
}

template<typename MapFn, typename ReduceFn>
void
parallelMapInOrderReduceChunked(size_t first, size_t last,
                                MapFn map, ReduceFn reduce,
                                size_t chunkSize)
{
    // Result type of map function (to be passed to reduce)
    typedef decltype(map(first)) MapResult;

    size_t range = last - first;
    size_t n = (range + chunkSize - 1) / chunkSize;

    auto mapChunk = [&] (size_t i) -> std::vector<MapResult>
        {
            std::vector<MapResult> result;

            size_t start = i * chunkSize;
            size_t end = std::min<size_t>(start + chunkSize, range);

            for (size_t j = start;  j < end;  ++j) {
                result.push_back(map(j));
            }

            return result;
        };

    auto reduceChunk = [&] (size_t i, std::vector<MapResult> & mapped)
        {
            for (size_t j = 0;  j < mapped.size();  ++j) {
                reduce(i * chunkSize + j, mapped[j]);
            }
        };

    parallelMapInOrderReduce(size_t(0), n, mapChunk, reduceChunk);
}

} // namespace ML

#endif

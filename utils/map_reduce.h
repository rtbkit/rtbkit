/** map_reduce.h                                                   -*- C++ -*-
    Jeremy Barnes, 30 October 2012
    Copyright (c) 2012 Datacratic Inc.  All rights reserved.

    Functionality to perform map/reduce type operations in parallel.
*/


#ifndef __utils__map_reduce_h__
#define __utils__map_reduce_h__

#include <utility>
#include <mutex>
#include "worker_task.h"
#include <boost/thread/mutex.hpp>

namespace ML {

template<typename MapFn, typename ReduceFn, typename It, typename It2>
void
parallelMapInOrderReduce(It first, It2 last, MapFn map, ReduceFn reduce)
{
    // Result type of map function (to be passed to reduce)
    typedef decltype(map(first)) MapResult;

    It next = first;
    std::map<It, MapResult> writeQueue;
    std::mutex lock;

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

            std::unique_lock<std::mutex> guard(lock);
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

            result.reserve(end - start);

            for (size_t j = start;  j < end;  ++j)
                result.emplace_back(std::move(map(j)));

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

template<typename MapFn, typename ReduceFn, typename It>
void
parallelMapInOrderReducePreChunked(const std::vector<std::pair<It, It> > & chunks,
                                   MapFn map, ReduceFn reduce)
{
    // Result type of map function (to be passed to reduce)
    typedef decltype(map(chunks[0].first)) MapResult;

    auto mapChunk = [&] (int chunkNum)
        {
            std::vector<MapResult> result;

            It start = chunks[chunkNum].first;
            It end = chunks[chunkNum].second;

            result.reserve(end - start);//std::distance(start, end));

            for (It j = start;  j != end;  ++j)
                result.emplace_back(std::move(map(j)));

            return result;
        };

    auto reduceChunk = [&] (size_t i, std::vector<MapResult> & mapped)
        {
            for (size_t j = 0;  j < mapped.size();  ++j)
                reduce(chunks[i].first + j, mapped[j]);
        };

    parallelMapInOrderReduce(0, chunks.size(), mapChunk, reduceChunk);
}


template<typename MapFn, typename ReduceFn, typename WorkFn,
         typename It, typename It2>
void
parallelMapInOrderReduceInEqualWorkChunks
    (It first, It2 last,
     MapFn map, ReduceFn reduce, WorkFn work,
     size_t targetNumChunks = 512)
{
    // How many total elements to process?
    size_t n = last - first;

    if (n == 0)
        return;

    // Find the work (amount of work to do) for each of the elements
    std::vector<size_t> elementWork;
    size_t totalWork = 0;
    elementWork.reserve(n);

    for (auto it = first;  it != last;  ++it) {
        size_t s = work(it);
        totalWork += s;
        elementWork.push_back(s);
    }
    
    if (totalWork == 0)
        throw ML::Exception("total work must not be zero");
    
    // Now group them into contiguous chunks (ranges of iterators to process)
    // such that each chunk gets a roughly even amount of work.
    std::vector<std::pair<It, It> > chunks;

    // How much work to put in each chunk to have exactly numChunks chunks
    size_t workPerChunk = totalWork / targetNumChunks;

    // Try to split them into batches of behaviours with roughly equal amounts of work
    It lastEndIt = first;
    int lastEndN = 0;

    while (lastEndN < n) {
        It curr = lastEndIt;
        It i = lastEndN;

        // The chunk has at least one work item in it
        size_t workInChunk = elementWork[i];
        ++i;
        ++curr;

        while (i < n) {
            size_t thisChunk = elementWork[i];
            if (workInChunk + thisChunk > workPerChunk)
                break;
            workInChunk += thisChunk;
            ++curr;
            ++i;
        }

        chunks.push_back(std::make_pair(lastEndIt, curr));
        lastEndIt = curr;
        lastEndN = i;
    }
    
    // Now run over the chunks
    parallelMapInOrderReducePreChunked(chunks, map, reduce);
}


} // namespace ML

#endif

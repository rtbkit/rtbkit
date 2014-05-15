/** post_auction_redis_bench.cc                                 -*- C++ -*-
    Rémi Attab, 16 Apr 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    Basic PoC and Benchmarks for the redis CaS functionality.

    The new PAL uses the WATCH ... MULTI ... EXEC pattern which turns caps out
    the default redis config at 10k ops/sec using ~20 connections. Was expecting
    more but still good enough for our current needs.

    Another finding is that we can't multiplex a connection while using this
    access pattern because that would mix multiple transaction which confuses
    redis. Instead we need to have one connection per live transaction.

 */

#include "soa/types/date.h"
#include "soa/service/redis.h"
#include "soa/utils/print_utils.h"

#include <thread>
#include <atomic>
#include <random>

using namespace ML;
using namespace Datacratic;


/******************************************************************************/
/* CONFIG                                                                     */
/******************************************************************************/

enum {
    Connections = 20,
    Duration = 10,
    MaxInFlights = 1,
};


/******************************************************************************/
/* CONNECTION                                                                 */
/******************************************************************************/

struct Connection
{
    Connection(const Redis::Address& addr) :
        processed(0), retries(0), errors(0), inFlights(0), conn(addr)
    {}

    bool send()
    {
        if (inFlights >= MaxInFlights) return false;
        inFlights++;

        query(pickKey());
        return true;
    }

    void close() { conn.close(); }

    std::atomic<size_t> processed;
    std::atomic<size_t> retries;
    std::atomic<size_t> errors;

private:

    uint64_t pickKey()
    {
        return std::uniform_int_distribution<uint64_t>(0, 1 << 16)(rng);
    }

    bool assertOk(uint64_t key, const Redis::Result& result)
    {
        if (result.ok()) return true;

        std::stringstream ss;
        ss << "Assert(" << key << "): " << result << std::endl;
        std::cerr << ss.str();

        errors++;
        inFlights--;
        return false;
    }

    void query(uint64_t key)
    {
        auto assertFn = [=] (const Redis::Result& r) { assertOk(key, r); };
        conn.queue(Redis::Command("WATCH", key), assertFn);

        auto mutateFn = [=] (const Redis::Result& r) { mutate(key, r); };
        conn.queue(Redis::Command("GET", key), mutateFn);
    }

    void mutate(uint64_t key, const Redis::Result& result)
    {
        if (!assertOk(key, result)) return;

        uint64_t newValue = 0;

        if (result.reply().type() != Redis::NIL) {
            std::string oldValue = result.reply().getString();
            newValue = std::stoull(oldValue) + 1;
        }

        auto postFn = [=] (const Redis::Result& r) { post(key, r); };
        auto assertFn = [=] (const Redis::Result& r) { assertOk(key, r); };

        conn.queue(Redis::MULTI, assertFn);
        conn.queue(Redis::Command("SET", key, std::to_string(newValue)), postFn);
        conn.queue(Redis::EXEC, assertFn);
    }

    void post(uint64_t key, const Redis::Result& result)
    {
        if (!assertOk(key, result)) return;

        if (result.reply().type() == Redis::NIL) {
            retries++;
            query(key);
            return;
        }

        inFlights--;
        processed++;
        return;
    }

    std::atomic<size_t> inFlights;
    Redis::AsyncConnection conn;
    std::mt19937 rng;
};


/******************************************************************************/
/* REPORT                                                                     */
/******************************************************************************/

void report(std::vector<Connection*>& connections, double duration)
{
    double processed = 0;
    double retries = 0;
    double errors = 0;

    for (auto& conn : connections) {
        processed += conn->processed;
        retries += conn->retries;
        errors += conn->errors;
    }

    double throughput = processed / duration;

    std::cerr << "\r"
        << "ops/sec=" << printValue(throughput)
        << ", retries=" << printPct(retries / processed)
        << ", errors=" << printPct(errors / processed);
}


/******************************************************************************/
/* RUN                                                                        */
/******************************************************************************/

void run(std::vector<Connection*>& connections)
{
    Date now = Date::now();
    Date start = now;
    Date print = now.plusSeconds(0.1);
    Date end = now.plusSeconds(Duration);

    std::cerr << "\n"
        << printElapsed(Duration) << " duration\n"
        << printValue(Connections) << " connections\n";


    while ((now = Date::now()) < end) {
        if (now > print) {
            report(connections, now.secondsSince(start));
            print = now.plusSeconds(0.1);
        }

        bool result = false;

        for (auto& conn : connections)
            result |= conn->send();

        if (!result)
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    std::cerr << std::endl;
}


/******************************************************************************/
/* MAIN                                                                       */
/******************************************************************************/

int main(int argc, char* argv[])
{
    Redis::Address addr("localhost:6379");

    std::vector<Connection*> connections;
    connections.reserve(Connections);

    while(connections.size() < Connections)
        connections.push_back(new Connection(addr));

    run(connections);
}

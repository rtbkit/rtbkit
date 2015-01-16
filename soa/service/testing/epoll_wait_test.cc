/* endpoint_periodic_test.cc
   Wolfgang Sourdeau, May 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.

   Tests for the handle of periodic events in endpoints.cc
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <time.h>
#include <atomic>
#include <boost/test/unit_test.hpp>
#include "soa/service/http_endpoint.h"
//#include <sys/socket.h>
#include "jml/utils/testing/watchdog.h"
//#include "jml/arch/timers.h"
//#include "jml/arch/futex.h"


using namespace std;
using namespace ML;
using namespace Datacratic;


/*
namespace {

struct MockEndpoint : public Datacratic::EndpointBase {
    MockEndpoint (const std::string & name)
        : EndpointBase(name)
    {
    }

    ~MockEndpoint()
    {
        shutdown();
    }

    virtual std::string hostname()
        const
    {
        return "mock-ep";
    }
    
    virtual int port()
        const
    {
        return -1;
    }

    virtual void closePeer()
    {}
};

}
*/

struct bench {
    template<typename F>
    bench(int times, F lambda) : k(times), f(lambda) {}

    void operator()() {
        timespec t0, t1;
        clock_gettime(CLOCK_REALTIME, &t0);

        long long sum = 0;
        for(int i = 0; i != k; i++) {
            sum += f();
        }

        clock_gettime(CLOCK_REALTIME, &t1);
        long long dt = (t1.tv_sec - t0.tv_sec) * 1000000000 + (t1.tv_nsec - t0.tv_nsec);
        printf("time: %lldns (%lldns/op) sum=%lld\n", dt, dt/k, sum);
    }

    int k;
    std::function<long long()> f;
};

template<typename F>
void load(int threads, F lambda) {
    std::vector<std::thread> workers;

    for(int i = 0; i < threads; i++) {
        workers.emplace_back(bench(32, lambda));
    }

    for(auto & item : workers) {
        item.join();
    }
}

template<typename F>
void work(int threads, F lambda) {
    HttpEndpoint endpoint("auctions");
    endpoint.realTimePolling(true);
    endpoint.init(PortRange(), "localhost", threads);
    lambda();
}

template<typename F>
void run_test(F lambda) {
    printf("baseline: 1 thread\n");
    load(1, lambda);

    printf("1 thread + 1 real-time polling thread\n");
    work(1, [=](){ load(1, lambda); });

    printf("10 threads + 1 real-time polling threads\n");
    work(1, [=](){ load(10, lambda); });

    printf("1 thread + 8 real-time polling threads\n");
    work(8, [=](){ load(1, lambda); });

    printf("10 threads + 8 real-time polling threads\n");
    work(8, [=](){ load(10, lambda); });

    printf("typical load: 15 threads + 8 real-time polling threads\n");
    work(8, [=](){ load(15, lambda); });
}

long long something_nice() {
    int n = 64*1014*1014;

    std::vector<int> data;
    data.resize(n);
    for(int i = 0; i != n; i++) {
        data[i] = i;
    }

    long long sum = 0;
    for(int i = 0; i != n; i++) {
        sum += data[i];
    }

    return sum;
}

long long something_that_allocates() {
    int n = 128*1024;
    int k = 0;

    std::vector<int *> allocs;

    long long sum = 0;
    for(int i = 0; i != n; i++) {
        k = unsigned(1013904223L+1664525L*(long long)k)%4096;
        if(i && (k%10)==0) {
            int * p = 0;
            int w = k%allocs.size();
            std::swap(p, allocs[w]);
            free(p);
        }

        int * data = (int *) malloc(sizeof(int)*k);
        allocs.push_back(data);

        for(int j = 0; j != k; j++) {
            data[j] = j;
        }

        for(int j = 0; j != k; j++) {
            sum += data[j];
        }
    }

    for(auto p : allocs) {
        free(p);
    }

    return sum;
}

std::atomic<int> global;

long long something_that_increments() {
    int n = 1024*1024;
    for(int i = 0; i != n; i++) {
        global++;
    }

    return global;
}

BOOST_AUTO_TEST_CASE( test_epoll_wait )
{
    // uses LLC a little bit but latency is hidden by L2.
    printf("linear memory access\n");
    run_test([](){ return something_nice(); });

    // uses LLC a lot more when using tcmalloc. This seems to represents best our situation.
    printf("linear memory access of small random segments\n");
    run_test([](){ return something_that_allocates(); });

    // overloads the LLC.
    printf("global atomic increment\n");
    run_test([](){ return something_that_increments(); });
}

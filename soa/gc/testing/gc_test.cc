/* gc_test.cc
   Jeremy Barnes, 23 February 2010
   Copyright (c) 2010 Datacratic.  All rights reserved.

   Test of the garbage collector locking.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "soa/gc/gc_lock.h"
#include "jml/utils/string_functions.h"
#include "jml/utils/exc_assert.h"
#include "jml/utils/guard.h"
#include "jml/arch/atomic_ops.h"
#include "jml/arch/thread_specific.h"
#include "jml/arch/rwlock.h"
#include "jml/arch/spinlock.h"
#include "jml/arch/tick_counter.h"
#include <boost/test/unit_test.hpp>
#include <boost/bind.hpp>
#include <iostream>
#include <atomic>

#include <boost/thread.hpp>
#include <boost/thread/barrier.hpp>


using namespace ML;
using namespace Datacratic;
using namespace std;

// Defined in gc_lock.cc
namespace Datacratic {
extern int32_t gcLockStartingEpoch;
};


#if 1

BOOST_AUTO_TEST_CASE ( test_gc )
{
    GcLock gc;
    gc.lockShared();

    BOOST_CHECK(gc.isLockedShared());

    bool deferred = false;

    cerr << endl << "before defer" << endl;
    gc.dump();

    gc.defer([&] () { deferred = true; memory_barrier(); });

    cerr << endl << "after defer" << endl;
    gc.dump();

    gc.unlockShared();

    cerr << endl << "after unlock shared" << endl;
    gc.dump();

    BOOST_CHECK(!gc.isLockedShared());
    BOOST_CHECK(deferred);
}

BOOST_AUTO_TEST_CASE(test_mutual_exclusion)
{
    cerr << "testing mutual exclusion" << endl;

    GcLock lock;
    volatile bool finished = false;
    volatile int numExclusive = 0;
    volatile int numShared = 0;
    int errors = 0;
    int multiShared = 0;
    uint64_t sharedIterations = 0;
    uint64_t exclusiveIterations = 0;

    auto sharedThread = [&] ()
        {
            while (!finished) {
                GcLock::SharedGuard guard(lock);
                ML::atomic_inc(numShared);

                if (numExclusive > 0) {
                    cerr << "exclusive and shared" << endl;
                    ML::atomic_inc(errors);
                }
                if (numShared > 1) {
                    ML::atomic_inc(multiShared);
                }

                ML::atomic_dec(numShared);
                ML::atomic_inc(sharedIterations);
                ML::memory_barrier();
            }
        };

    auto exclusiveThread = [&] ()
        {
            while (!finished) {
                GcLock::ExclusiveGuard guard(lock);
                ML::atomic_inc(numExclusive);

                if (numExclusive > 1) {
                    cerr << "more than one exclusive" << endl;
                    ML::atomic_inc(errors);
                }
                if (numShared > 0) {
                    cerr << "exclusive and shared" << endl;
                    ML::atomic_inc(multiShared);
                }

                ML::atomic_dec(numExclusive);
                ML::atomic_inc(exclusiveIterations);
                ML::memory_barrier();
            }
        };

    lock.getEntry();

    int nthreads = 4;

    {
        cerr << "single shared" << endl;
        sharedIterations = exclusiveIterations = multiShared = finished = 0;
        boost::thread_group tg;
        tg.create_thread(sharedThread);
        sleep(1);
        finished = true;
        tg.join_all();
        BOOST_CHECK_EQUAL(errors, 0);
        cerr << "iterations: shared " << sharedIterations
             << " exclusive " << exclusiveIterations << endl;
        cerr << "multiShared = " << multiShared << endl;
    }

    {
        cerr << "multi shared" << endl;
        sharedIterations = exclusiveIterations = multiShared = finished = 0;
        boost::thread_group tg;
        for (unsigned i = 0;  i < nthreads;  ++i)
            tg.create_thread(sharedThread);
        sleep(1);
        finished = true;
        tg.join_all();
        BOOST_CHECK_EQUAL(errors, 0);
        if (nthreads > 1)
            BOOST_CHECK_GE(multiShared, 0);
        cerr << "iterations: shared " << sharedIterations
             << " exclusive " << exclusiveIterations << endl;
        cerr << "multiShared = " << multiShared << endl;
    }

    {
        cerr << "single exclusive" << endl;
        sharedIterations = exclusiveIterations = multiShared = finished = 0;
        boost::thread_group tg;
        tg.create_thread(exclusiveThread);
        sleep(1);
        finished = true;
        tg.join_all();
        BOOST_CHECK_EQUAL(errors, 0);
        cerr << "iterations: shared " << sharedIterations
             << " exclusive " << exclusiveIterations << endl;
        cerr << "multiShared = " << multiShared << endl;
    }

    {
        cerr << "multi exclusive" << endl;
        sharedIterations = exclusiveIterations = multiShared = finished = 0;
        boost::thread_group tg;
        for (unsigned i = 0;  i < nthreads;  ++i)
            tg.create_thread(exclusiveThread);
        sleep(1);
        finished = true;
        tg.join_all();
        BOOST_CHECK_EQUAL(errors, 0);
        cerr << "iterations: shared " << sharedIterations
             << " exclusive " << exclusiveIterations << endl;
        cerr << "multiShared = " << multiShared << endl;
    }

    {
        cerr << "mixed shared and exclusive" << endl;
        sharedIterations = exclusiveIterations = multiShared = finished = 0;
        boost::thread_group tg;
        for (unsigned i = 0;  i < nthreads;  ++i)
            tg.create_thread(sharedThread);
        for (unsigned i = 0;  i < nthreads;  ++i)
            tg.create_thread(exclusiveThread);
        sleep(1);
        finished = true;
        tg.join_all();
        BOOST_CHECK_EQUAL(errors, 0);
        if (nthreads > 1)
            BOOST_CHECK_GE(multiShared, 0);
        cerr << "iterations: shared " << sharedIterations
             << " exclusive " << exclusiveIterations << endl;
        cerr << "multiShared = " << multiShared << endl;
    }

    {
        cerr << "overflow" << endl;
        gcLockStartingEpoch = 0xFFFFFFF0;
        sharedIterations = exclusiveIterations = multiShared = finished = 0;
        boost::thread_group tg;
        tg.create_thread(sharedThread);
        sleep(1);
        finished = true;
        tg.join_all();
        BOOST_CHECK_EQUAL(errors, 0);
        cerr << "iterations: shared " << sharedIterations
             << " exclusive " << exclusiveIterations << endl;
        cerr << "multiShared = " << multiShared << endl;
    }

    {
        cerr << "INT_MIN to INT_MAX" << endl;
        gcLockStartingEpoch = 0x7FFFFFF0;
        sharedIterations = exclusiveIterations = multiShared = finished = 0;
        boost::thread_group tg;
        tg.create_thread(sharedThread);
        sleep(1);
        finished = true;
        tg.join_all();
        BOOST_CHECK_EQUAL(errors, 0);
        cerr << "iterations: shared " << sharedIterations
             << " exclusive " << exclusiveIterations << endl;
        cerr << "multiShared = " << multiShared << endl;
    }

    {
        cerr << "benign overflow" << endl;
        gcLockStartingEpoch = 0xBFFFFFF0;
        sharedIterations = exclusiveIterations = multiShared = finished = 0;
        boost::thread_group tg;
        tg.create_thread(sharedThread);
        sleep(1);
        finished = true;
        tg.join_all();
        BOOST_CHECK_EQUAL(errors, 0);
        cerr << "iterations: shared " << sharedIterations
             << " exclusive " << exclusiveIterations << endl;
        cerr << "multiShared = " << multiShared << endl;
    }

}

#endif

#define USE_MALLOC 1

template<typename T>
struct Allocator {
    Allocator(int nblocks, T def = T())
        : def(def)
    {
        init(nblocks);
        highestAlloc = nallocs = ndeallocs = 0;
    }

    ~Allocator()
    {
#if ( ! USE_MALLOC )
        delete[] blocks;
        delete[] free;
#endif
    }

    T def;
    T * blocks;
    int * free;
    int nfree;
    int highestAlloc;
    int nallocs;
    int ndeallocs;
    ML::Spinlock lock;

    void init(int nblocks)
    {
#if ( ! USE_MALLOC )
        blocks = new T[nblocks];
        free = new int[nblocks];

        std::fill(blocks, blocks + nblocks, def);

        nfree = 0;
        for (int i = nblocks - 1;  i >= 0;  --i)
            free[nfree++] = i;
#endif
    }

    T * alloc()
    {
#if USE_MALLOC
        ML::atomic_inc(nallocs);
        ML::atomic_max(highestAlloc, nallocs - ndeallocs);
        return new T(def);
#else
        boost::lock_guard<ML::Spinlock> guard(lock);
        if (nfree == 0)
            throw ML::Exception("none free");
        int i = free[nfree - 1];
        highestAlloc = std::max(highestAlloc, i);
        T * result = blocks + i;
        --nfree;
        ++nallocs;
        return result;
#endif
    }

    void dealloc(T * value)
    {
        if (!value) return;
        *value = def;
#if USE_MALLOC
        delete value;
        ML::atomic_inc(ndeallocs);
        return;
#else
        boost::lock_guard<ML::Spinlock> guard(lock);
        int i = value - blocks;
        free[nfree++] = i;
        ++ndeallocs;
#endif
    }

    static void doDealloc(void * thisptr, void * blockPtr_, void * blockVar_)
    {
        int * & blockVar = *reinterpret_cast<int **>(blockVar_);
        int * blockPtr = reinterpret_cast<int *>(blockPtr_);
        ExcAssertNotEqual(blockVar, blockPtr);
        //blockVar = 0;
        //ML::memory_barrier();
        //cerr << "blockPtr = " << blockPtr << endl;
        //int * blockPtr = reinterpret_cast<int *>(block);
        reinterpret_cast<Allocator *>(thisptr)->dealloc(blockPtr);
    }

    static void doDeallocAll(void * thisptr, void * blocksPtr_, void * numBlocks_)
    {
        size_t numBlocks = reinterpret_cast<size_t>(numBlocks_);
        int ** blocksPtr = reinterpret_cast<int **>(blocksPtr_);
        Allocator * alloc = reinterpret_cast<Allocator *>(thisptr);

        for (unsigned i = 0;  i != numBlocks;  ++i) {
            if (blocksPtr[i])
                alloc->dealloc(blocksPtr[i]);
        }

        delete[] blocksPtr;
    }
};

template<typename Lock>
struct TestBase {
    TestBase(int nthreads, int nblocks, int nSpinThreads = 0)
        : finished(false),
          nthreads(nthreads),
          nblocks(nblocks),
          nSpinThreads(nSpinThreads),
          allocator(1024 * 1024, -1),
          nerrors(0),
          allBlocks(nthreads)
    {
        for (unsigned i = 0;  i < nthreads;  ++i) {
            allBlocks[i] = new int *[nblocks];
            std::fill(allBlocks[i].load(), allBlocks[i].load() + nblocks, (int *)0);
        }
    }

    ~TestBase()
    {
        for (unsigned i = 0;  i < nthreads;  ++i)
            delete[] allBlocks[i];
    }

    volatile bool finished;
    int nthreads;
    int nblocks;
    int nSpinThreads;
    Allocator<int> allocator;
    Lock gc;
    uint64_t nerrors;

    /* All of the blocks are published here.  Any pointer which is read from
       here by another thread should always refer to exactly the same
       value.
    */
    vector<atomic<int **>> allBlocks;

    void checkVisible(int threadNum, unsigned long long start)
    {
        // We're reading from someone else's pointers, so we need to lock here
        //gc.enterCS();
        gc.lockShared();

        for (unsigned i = 0;  i < nthreads;  ++i) {
            for (unsigned j = 0;  j < nblocks;  ++j) {
                //int * val = allBlocks[i][j];
                int * val = allBlocks[i].load()[j];
                if (val) {
                    int atVal = *val;
                    if (atVal != i) {
                        cerr << ML::format("%.6f thread %d: invalid value read "
                                "from thread %d block %d: %d\n",
                                (ticks() - start) / ticks_per_second, threadNum,
                                i, j, atVal);
                        ML::atomic_inc(nerrors);
                        //abort();
                    }
                }
            }
        }

        //gc.exitCS();
        gc.unlockShared();
    }

    void doReadThread(int threadNum)
    {
        gc.getEntry();
        unsigned long long start = ticks();
        while (!finished) {
            checkVisible(threadNum, start);
        }
    }

    void doSpinThread()
    {
        while (!finished) {
        }
    }

    void allocThreadDefer(int threadNum)
    {
        gc.getEntry();
        try {
            uint64_t nErrors = 0;

            int ** blocks = allBlocks[threadNum];

            while (!finished) {

                int ** oldBlocks = new int * [nblocks];

                //gc.enterCS();

                for (unsigned i = 0;  i < nblocks;  ++i) {
                    int * block = allocator.alloc();
                    if (*block != -1) {
                        cerr << "old block was allocated" << endl;
                        ++nErrors;
                    }
                    *block = threadNum;
                    ML::memory_barrier();
                    //rcu_set_pointer_sym((void **)&blocks[i], block);
                    int * oldBlock = blocks[i];
                    blocks[i] = block;
                    ML::memory_barrier();
                    oldBlocks[i] = oldBlock;
                }

                gc.defer(Allocator<int>::doDeallocAll, &allocator, oldBlocks,
                         (void *)(size_t)nblocks);

                //gc.exitCS();
            }


            int * oldBlocks[nblocks];

            for (unsigned i = 0;  i < nblocks;  ++i) {
                oldBlocks[i] = blocks[i];
                blocks[i] = 0;
            }

            gc.visibleBarrier();

            //cerr << "at end" << endl;

            for (unsigned i = 0;  i < nblocks;  ++i)
                allocator.dealloc(oldBlocks[i]);

            //cerr << "nErrors = " << nErrors << endl;
        } catch (...) {
            static ML::Spinlock lock;
            lock.acquire();
            //cerr << "threadnum " << threadNum << " inEpoch "
            //     << gc.getEntry().inEpoch << endl;
            gc.dump();
            abort();
        }
    }

    void allocThreadSync(int threadNum)
    {
        gc.getEntry();
        try {
            uint64_t nErrors = 0;

            int ** blocks = allBlocks[threadNum];
            int * oldBlocks[nblocks];

            while (!finished) {

                for (unsigned i = 0;  i < nblocks;  ++i) {
                    int * block = allocator.alloc();
                    if (*block != -1) {
                        cerr << "old block was allocated" << endl;
                        ++nErrors;
                    }
                    *block = threadNum;
                    int * oldBlock = blocks[i];
                    blocks[i] = block;
                    oldBlocks[i] = oldBlock;
                }

                ML::memory_barrier();
                gc.visibleBarrier();

                for (unsigned i = 0;  i < nblocks;  ++i)
                    if (oldBlocks[i]) *oldBlocks[i] = 1234;

                for (unsigned i = 0;  i < nblocks;  ++i)
                    if (oldBlocks[i]) allocator.dealloc(oldBlocks[i]);
            }

            for (unsigned i = 0;  i < nblocks;  ++i) {
                oldBlocks[i] = blocks[i];
                blocks[i] = 0;
            }

            gc.visibleBarrier();

            for (unsigned i = 0;  i < nblocks;  ++i)
                allocator.dealloc(oldBlocks[i]);

            //cerr << "nErrors = " << nErrors << endl;
        } catch (...) {
            static ML::Spinlock lock;
            lock.acquire();
            //cerr << "threadnum " << threadNum << " inEpoch "
            //     << gc.getEntry().inEpoch << endl;
            gc.dump();
            abort();
        }
    }

    void run(boost::function<void (int)> allocFn,
             int runTime = 1)
    {
        gc.getEntry();
        boost::thread_group tg;

        for (unsigned i = 0;  i < nthreads;  ++i)
            tg.create_thread(boost::bind<void>(&TestBase::doReadThread, this, i));

        for (unsigned i = 0;  i < nthreads;  ++i)
            tg.create_thread(boost::bind<void>(allocFn, i));

        for (unsigned i = 0;  i < nSpinThreads;  ++i)
            tg.create_thread(boost::bind<void>(&TestBase::doSpinThread, this));

        sleep(runTime);

        finished = true;

        tg.join_all();

        gc.deferBarrier();

        gc.dump();

        BOOST_CHECK_EQUAL(allocator.nallocs, allocator.ndeallocs);
        BOOST_CHECK_EQUAL(nerrors, 0);

        cerr << "allocs " << allocator.nallocs
             << " deallocs " << allocator.ndeallocs << endl;
        cerr << "highest " << allocator.highestAlloc << endl;

        cerr << "gc.currentEpoch() = " << gc.currentEpoch() << endl;
    }
};

#if 1
BOOST_AUTO_TEST_CASE ( test_gc_sync_many_threads_contention )
{
    cerr << "testing contention synchronized GcLock with many threads" << endl;

    int nthreads = 8;
    int nSpinThreads = 16;
    int nblocks = 2;

    TestBase<GcLock> test(nthreads, nblocks, nSpinThreads);
    test.run(boost::bind(&TestBase<GcLock>::allocThreadSync, &test, _1));
}
#endif

BOOST_AUTO_TEST_CASE ( test_gc_deferred_contention )
{
    cerr << "testing contended deferred GcLock" << endl;

    int nthreads = 8;
    int nSpinThreads = 0;//16;
    int nblocks = 2;

    TestBase<GcLock> test(nthreads, nblocks, nSpinThreads);
    test.run(boost::bind(&TestBase<GcLock>::allocThreadDefer, &test, _1));
}


#if 1

BOOST_AUTO_TEST_CASE ( test_gc_sync )
{
    cerr << "testing synchronized GcLock" << endl;

    int nthreads = 2;
    int nblocks = 2;

    TestBase<GcLock> test(nthreads, nblocks);
    test.run(boost::bind(&TestBase<GcLock>::allocThreadSync, &test, _1));
}

BOOST_AUTO_TEST_CASE ( test_gc_sync_many_threads )
{
    cerr << "testing synchronized GcLock with many threads" << endl;

    int nthreads = 8;
    int nblocks = 2;

    TestBase<GcLock> test(nthreads, nblocks);
    test.run(boost::bind(&TestBase<GcLock>::allocThreadSync, &test, _1));
}

BOOST_AUTO_TEST_CASE ( test_gc_deferred )
{
    cerr << "testing deferred GcLock" << endl;

    int nthreads = 2;
    int nblocks = 2;

    TestBase<GcLock> test(nthreads, nblocks);
    test.run(boost::bind(&TestBase<GcLock>::allocThreadDefer, &test, _1));
}


struct SharedGcLockProxy : public SharedGcLock {
    static const char* name;
    SharedGcLockProxy() :
        SharedGcLock(GC_OPEN, name)
    {}
};
const char* SharedGcLockProxy::name = "gc_test.dat";

BOOST_AUTO_TEST_CASE( test_shared_lock_sync )
{
    cerr << "testing contention synchronized GcLock with shared lock" << endl;

    SharedGcLock lockGuard(GC_CREATE, SharedGcLockProxy::name);
    Call_Guard unlink_guard([&] { lockGuard.unlink(); });

    int nthreads = 8;
    int nSpinThreads = 16;
    int nblocks = 2;

    TestBase<SharedGcLockProxy> test(nthreads, nblocks, nSpinThreads);
    test.run(boost::bind(
                    &TestBase<SharedGcLockProxy>::allocThreadSync, &test, _1));

}

BOOST_AUTO_TEST_CASE( test_shared_lock_defer )
{
    cerr << "testing contended deferred GcLock with shared lock" << endl;

    SharedGcLock lockGuard(GC_CREATE, SharedGcLockProxy::name);
    Call_Guard unlink_guard([&] { lockGuard.unlink(); });

    int nthreads = 8;
    int nSpinThreads = 16;
    int nblocks = 2;

    TestBase<SharedGcLockProxy> test(nthreads, nblocks, nSpinThreads);
    test.run(boost::bind(
                    &TestBase<SharedGcLockProxy>::allocThreadSync, &test, _1));
}

BOOST_AUTO_TEST_CASE ( test_defer_race )
{
    cerr << "testing defer race" << endl;
    GcLock gc;

    boost::thread_group tg;

    volatile bool finished = false;

    int nthreads = 0;

    volatile int numStarted = 0;

    auto doTestThread = [&] ()
        {
            while (!finished) {
                ML::atomic_inc(numStarted);
                while (numStarted != nthreads) ;

                gc.deferBarrier();

                ML::atomic_dec(numStarted);
                while (numStarted != 0) ;
            }
        };


    for (unsigned i = 0;  i < nthreads;  ++i)
        tg.create_thread(doTestThread);

    int runTime = 1;

    sleep(runTime);

    finished = true;

    tg.join_all();
}

#endif

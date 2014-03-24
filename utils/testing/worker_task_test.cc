/* worker_task_test.cc
   Jeremy Barnes, 5 May 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Test for the worker task code.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/thread.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/bind.hpp>
#include <vector>
#include <stdint.h>
#include <iostream>

#include "jml/utils/worker_task.h"
#include "jml/boosting/training_data.h"
#include "jml/boosting/dense_features.h"
#include "jml/boosting/feature_info.h"
#include "jml/utils/smart_ptr_utils.h"
#include "jml/utils/vector_utils.h"
#include "jml/arch/timers.h"
#include "jml/utils/guard.h"
#include "jml/arch/exception_handler.h"
#include "jml/arch/atomic_ops.h"
#include "jml/arch/demangle.h"

using namespace ML;
using namespace std;

using boost::unit_test::test_suite;

// With a semaphore created with a count of nthreads, there should never be a
// case where tryacquire() fails.
template<class Semaphore>
void test_semaphore_thread(Semaphore & sem, boost::barrier & barrier,
                           int & errors, int niter)
{
    barrier.wait();
    
    int my_errors = 0;

    for (unsigned i = 0;  i < niter;  ++i) {
        int res = sem.tryacquire();
        if (res == -1)
            ++my_errors;
        else sem.release();
    }

    atomic_add(errors, my_errors);
}

template<class Semaphore>
void test_semaphore(int nthreads, int niter)
{
    cerr << "testing type " << demangle(typeid(Semaphore).name())
         << " with " << nthreads << " threads and " << niter << " iter"
         << endl;
    boost::barrier barrier(nthreads);
    int errors = 0;
    Semaphore sem(nthreads);

    boost::thread_group tg;
    for (unsigned i = 0;  i < nthreads;  ++i)
        tg.create_thread(boost::bind(test_semaphore_thread<Semaphore>,
                                     boost::ref(sem),
                                     boost::ref(barrier),
                                     boost::ref(errors),
                                     niter));
    
    tg.join_all();

    BOOST_CHECK_EQUAL(errors, 0);
}

void null_job()
{
}

void test_overhead_job(int nthreads, int ntasks, bool verbose = true,
                       const Job & job = null_job)
{
    Worker_Task worker(nthreads - 1);
    
    /* We submit 1 million do-nothing tasks, and look at how long it takes
       to do them all */
    
    Timer timer;

    int group;
    {
        int parent = -1;  // no parent group
        group = worker.get_group(NO_JOB, "", parent);
        Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                     boost::ref(worker),
                                     group));
        
        for (unsigned i = 0;  i < ntasks;  ++i)
            worker.add(job, "", group);
    }
    
    worker.run_until_finished(group);

    if (verbose)
        cerr << "elapsed for " << ntasks << " null tasks in " << nthreads
             << " threads:" << timer.elapsed() << endl;
}

void exception_job()
{
    throw Exception("there was an exception");
}

#if 0
BOOST_AUTO_TEST_CASE(test_ace_semaphore)
{
    test_semaphore<ACE_Semaphore>(1, 1000000);
    test_semaphore<ACE_Semaphore>(10, 100000);
    test_semaphore<ACE_Semaphore>(100, 10000);
}
#endif

BOOST_AUTO_TEST_CASE(test_our_semaphore)
{
    test_semaphore<Semaphore>(1, 1000000);
    test_semaphore<Semaphore>(10, 100000);
    test_semaphore<Semaphore>(100, 10000);
}

BOOST_AUTO_TEST_CASE( test_create_destroy )
{
    int njobs = 1000;
    for (unsigned i = 0;  i < 100;  ++i)
        test_overhead_job(4, njobs, false /* verbose */);
}

BOOST_AUTO_TEST_CASE( test_overhead )
{
    int njobs = 100000;

    test_overhead_job(1, njobs);
    test_overhead_job(2, njobs);
    test_overhead_job(4, njobs);
    test_overhead_job(8, njobs);
    test_overhead_job(16, njobs);
}

BOOST_AUTO_TEST_CASE( test_exception )
{
    int njobs = 1000;
    set_trace_exceptions(false);

    for (unsigned i = 0;  i < 100;  ++i) {
        JML_TRACE_EXCEPTIONS(false);
        BOOST_CHECK_THROW(test_overhead_job(4, njobs, false /* verbose */,
                                            exception_job),
                          std::exception);
    }
}

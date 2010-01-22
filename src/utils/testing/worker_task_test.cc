/* worker_task_test.cc
   Jeremy Barnes, 5 May 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Test for the worker task code.
*/

/* decision_tree_xor_test.cc
   Jeremy Barnes, 25 February 2008
   Copyright (c) 2008 Jeremy Barnes.  All rights reserved.

   Test of the decision tree class.
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

#include "utils/worker_task.h"
#include "boosting/training_data.h"
#include "boosting/dense_features.h"
#include "boosting/feature_info.h"
#include "utils/smart_ptr_utils.h"
#include "utils/vector_utils.h"
#include "arch/timers.h"
#include "utils/guard.h"

using namespace ML;
using namespace std;

using boost::unit_test::test_suite;

void null_job()
{
}

void test_overhead_job(int nthreads, int ntasks, bool verbose = true)
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
            worker.add(null_job, "", group);
    }
    
    worker.run_until_finished(group);
    
    if (verbose)
        cerr << "elapsed for " << ntasks << " null tasks in " << nthreads
             << " threads:" << timer.elapsed() << endl;
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

/* backprop_cuda_test.cc
   Jeremy Barnes, 26 May 2008
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Test of the CUDA decision tree training code.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/thread.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/bind.hpp>
#include <boost/shared_array.hpp>
#include <boost/scoped_array.hpp>
#include <boost/timer.hpp>
#include <vector>
#include <stdint.h>
#include <iostream>
#include <limits>

#include "jml/boosting/training_index.h"
#include "jml/boosting/split.h"
#include "jml/boosting/feature_set.h"
#include "jml/boosting/dense_features.h"
#include "jml/boosting/feature_info.h"
#include "jml/boosting/stump_training_core.h"
#include "jml/boosting/stump_training_bin.h"
#include "jml/boosting/stump_training_cuda.h"
#include "jml/utils/smart_ptr_utils.h"
#include "jml/utils/vector_utils.h"
#include "jml/arch/timers.h"

using namespace ML;
using namespace std;

using boost::unit_test::test_suite;


typedef ML::CUDA::Test_Buckets_Binsym::Float Float;
typedef ML::CUDA::Test_Buckets_Binsym::TwoBuckets TwoBuckets;

struct Test_Context {
    const CUDA::Test_Buckets_Binsym * tester;
    boost::shared_array<TwoBuckets> accum;
    boost::shared_array<TwoBuckets> w_totals;
    std::shared_ptr<CUDA::Test_Buckets_Binsym::Plan> plan;
    std::shared_ptr<CUDA::Test_Buckets_Binsym::Context> context;

    // Host-based thread
    void operator () ()
    {
        context = tester->execute(*plan, accum.get(), w_totals[0]);
    }
};

void run_test(const uint16_t * buckets,
              const uint32_t * examples, // or 0 if example num == i
              const int32_t * labels,
              const float * divisors,
              uint32_t size,
              const float * weights,
              const float * ex_weights,
              int num_buckets,
              bool categorical,
              bool on_device,
              bool compressed,
              int num_in_parallel = 1)
{
    CUDA::Test_Buckets_Binsym tester;

    Timer t;

    std::shared_ptr<CUDA::Test_Buckets_Binsym::Plan> plan
        = tester.plan(buckets,
                      examples,
                      labels ,
                      divisors  ,
                      size,
                      weights,
                      ex_weights,
                      num_buckets,
                      on_device,
                      compressed);

    cerr << "planning took " << t.elapsed() << endl;
    t.restart();

    vector<Test_Context> contexts(num_in_parallel);

    boost::thread_group tg;

    for (unsigned i = 0;  i < num_in_parallel;  ++i) {
        Test_Context & context = contexts[i];
        context.accum.reset(new TwoBuckets[num_buckets]);
        context.w_totals.reset(new TwoBuckets[1]);
        context.plan = plan;
        context.tester = &tester;
        
        if (on_device) context();
        else tg.create_thread(boost::ref(context));
    }

    if (!on_device) tg.join_all();
    
    for (unsigned i = 0;  i < num_in_parallel;  ++i)
        tester.synchronize(*contexts[i].context);

    cerr << "execution of " << num_in_parallel
         << " parallel took " << t.elapsed() << endl;
}

static const char * xor_dataset = "\
LABEL X Y\n\
1 0 0\n\
0 1 0\n\
0 0 1\n\
1 1 1\n\
";

BOOST_AUTO_TEST_CASE( test_split_cuda )
{
    return;
    
    Dense_Feature_Space fs;

    Dense_Training_Data data;
    data.init(xor_dataset, xor_dataset + strlen(xor_dataset),
              make_unowned_sp(fs));
    guess_all_info(data, fs, true);

    float weights[4] = { 0.4, 0.3, 0.2, 0.1 };
    float ex_weights[4] = { 1.00, 1.00, 1.00, 1.00 };
    
    Joint_Index index
        = data.index()
        .joint(Feature(0, 0, 0),
               Feature(1, 0, 0),
               BY_EXAMPLE,
               IC_BUCKET | IC_LABEL | IC_EXAMPLE | IC_DIVISOR,
               2 /* num buckets */);

    BOOST_REQUIRE_EQUAL(index.size(), 4);

    run_test(index.buckets(),
             index.examples(),
             index.labels_as_int(),
             index.divisors(),
             index.size(),
             weights, ex_weights, 2,
             false, /* categorical */
             false, /* compressed */
             true /* on device */);

    run_test(index.buckets(),
             index.examples(),
             index.labels_as_int(),
             index.divisors(),
             index.size(),
             weights, ex_weights, 2,
             false, /* categorical */
             true, /* compressed */
             true /* on device */);
}

// Same test, but with a large size of the index
BOOST_AUTO_TEST_CASE( test_split_cuda2 )
{
    size_t array_size = 20000000;
    size_t num_buckets = 200;

    // Create our data
    boost::scoped_array<uint16_t> buckets   (new uint16_t[array_size]);
    boost::scoped_array<int>      labels    (new int[array_size]);
    boost::scoped_array<float>    ex_weights(new float[array_size]);
    boost::scoped_array<float>    weights   (new float[array_size]);

    for (unsigned i = 0;  i < array_size;  ++i) {
        buckets[i]    = rand() % num_buckets;
        labels[i]     = rand() % 2;
        ex_weights[i] = 1.0;
        weights[i]    = 1.0 / array_size;
    }

    boost::timer t;

    run_test(buckets.get(),
             0 /* examples */,
             labels.get(),
             0 /* divisors */,
             array_size,
             weights.get(),
             ex_weights.get(),
             num_buckets,
             false, /* categorical */
             true,  /* on device */
             false, /* compressed */
             16 /* num parallel */);

    cerr << "gpu uncompressed took " << t.elapsed() << "s" << endl;

    t.restart();

    run_test(buckets.get(),
             0 /* examples */,
             labels.get(),
             0 /* divisors */,
             array_size,
             weights.get(),
             ex_weights.get(),
             num_buckets,
             false, /* categorical */
             true, /* on device */
             true, /* compressed */
             16 /* num parallel */);
    
    cerr << "gpu compressed took " << t.elapsed() << "s" << endl;

    t.restart();

    run_test(buckets.get(),
             0 /* examples */,
             labels.get(),
             0 /* divisors */,
             array_size,
             weights.get(),
             ex_weights.get(),
             num_buckets,
             false, /* categorical */
             false, /* on device */
             false, /* compressed */
             16 /* num parallel */);
    
    cerr << "cpu uncompressed took " << t.elapsed() << "s" << endl;

    t.restart();

    run_test(buckets.get(),
             0 /* examples */,
             labels.get(),
             0 /* divisors */,
             array_size,
             weights.get(),
             ex_weights.get(),
             num_buckets,
             false, /* categorical */
             false, /* on device */
             true, /* compressed */
             16 /* num parallel */);
    
    cerr << "cpu compressed took " << t.elapsed() << "s" << endl;
}

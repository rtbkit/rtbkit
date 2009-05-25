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

#include "boosting/worker_task.h"
#include "boosting/training_data.h"
#include "boosting/dense_features.h"
#include "boosting/feature_info.h"
#include "utils/smart_ptr_utils.h"
#include "utils/vector_utils.h"

using namespace ML;
using namespace std;

using boost::unit_test::test_suite;


BOOST_AUTO_TEST_CASE( test_overhead )
{
}

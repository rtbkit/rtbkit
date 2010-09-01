/* output_encoder_test.cc
   Jeremy Barnes, 18 May 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

   Test of the output encoder.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#define JML_TESTING_PERCEPTRON

#include <boost/test/unit_test.hpp>
#include <boost/thread.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/bind.hpp>
#include <vector>
#include <stdint.h>
#include <iostream>

#include "jml/neural/output_encoder.h"
#include "jml/neural/dense_layer.h"
#include "jml/utils/smart_ptr_utils.h"
#include "jml/utils/vector_utils.h"
#include "jml/arch/exception_handler.h"

using namespace ML;
using namespace std;

using boost::unit_test::test_suite;

BOOST_AUTO_TEST_CASE( test_output_encoder )
{
    Output_Encoder encoder;
}

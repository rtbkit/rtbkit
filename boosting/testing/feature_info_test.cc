/* glz_classifier_test.cc
   Jeremy Barnes, 14 May 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

   Test of the GLZ classifier class.
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#define JML_TESTING_GLZ_CLASSIFIER

#include <boost/test/unit_test.hpp>
#include <boost/thread.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/bind.hpp>
#include <vector>
#include <stdint.h>
#include <iostream>

#include "jml/boosting/dense_features.h"
#include "jml/boosting/feature_info.h"
#include "jml/utils/smart_ptr_utils.h"
#include "jml/utils/vector_utils.h"
#include "jml/arch/exception_handler.h"

using namespace ML;
using namespace std;

using boost::unit_test::test_suite;

BOOST_AUTO_TEST_CASE( test_glz_classifier_test )
{
    string toParse = "k=CATEGORICAL/c=2,Male,Female/o=OPTIONAL";
    ML::Parse_Context context(toParse, toParse.c_str(), toParse.length());
    Mutable_Feature_Info info;
    info.parse(context);

    BOOST_CHECK_EQUAL(info.type(), CATEGORICAL);
    BOOST_CHECK_EQUAL(info.optional(), true);
    BOOST_CHECK_EQUAL(info.biased(), false);
    BOOST_REQUIRE(info.categorical());
    BOOST_CHECK_EQUAL(info.categorical()->count(), 2);
    BOOST_CHECK_EQUAL(info.categorical()->print(0), "Male");
    BOOST_CHECK_EQUAL(info.categorical()->print(1), "Female");
}

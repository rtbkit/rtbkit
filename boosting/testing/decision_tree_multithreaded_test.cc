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

#include "jml/boosting/decision_tree_generator.h"
#include "jml/boosting/training_data.h"
#include "jml/boosting/dense_features.h"
#include "jml/boosting/feature_info.h"
#include "jml/utils/smart_ptr_utils.h"
#include "jml/utils/vector_utils.h"

using namespace ML;
using namespace std;

using boost::unit_test::test_suite;

static const char * config_options = "\
trace=0\n\
";

BOOST_AUTO_TEST_CASE( test_decision_tree_multithreaded1 )
{
    /* Create the dataset */

    Dense_Feature_Space fs;
    fs.add_feature("LABEL", Feature_Info(REAL, false, true));
    fs.add_feature("feature1", REAL);
    fs.add_feature("feature2", REAL);
    fs.add_feature("feature3", REAL);

    std::shared_ptr<Dense_Feature_Space> fsp(make_unowned_sp(fs));

    Training_Data data(fsp);
    
    int nfv = 10000;

    for (unsigned i = 0;  i < nfv;  ++i) {
        distribution<float> features;
        features.push_back(i % 2);
        features.push_back(i);
        features.push_back(i);
        features.push_back(i);

        //features.push_back(random());
        //features.push_back(random());
        //features.push_back(random());

        std::shared_ptr<Feature_Set> fset
            = fs.encode(features);

        data.add_example(fset);
    }

    /* Create the decision tree generator */
    Configuration config;
    config.parse_string(config_options, "inbuilt config file");

    Decision_Tree_Generator generator;
    generator.configure(config);
    generator.init(fsp, fs.features()[0]);

    distribution<float> training_weights(nfv, 1);

    vector<Feature> features = fs.features();
    features.erase(features.begin(), features.begin() + 1);

    Thread_Context context;

    generator.generate(context, data, training_weights, features);
}

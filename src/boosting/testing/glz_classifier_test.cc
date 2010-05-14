/* glz_classifier_test.cc
   Jeremy Barnes, 14 May 2010
   Copyright (c) 2010 Jeremy Barnes.  All rights reserved.

   Test of the GLZ classifier class.
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

#include "jml/boosting/glz_classifier_generator.h"
#include "jml/boosting/training_data.h"
#include "jml/boosting/dense_features.h"
#include "jml/boosting/feature_info.h"
#include "jml/utils/smart_ptr_utils.h"
#include "jml/utils/vector_utils.h"

using namespace ML;
using namespace std;

using boost::unit_test::test_suite;

static const char * config_options = "\
verbosity=3\n\
";

BOOST_AUTO_TEST_CASE( test_glz_classifier_test )
{
    /* Create the dataset */

    Dense_Feature_Space fs;
    fs.add_feature("LABEL", Feature_Info(BOOLEAN, false, true));
    fs.add_feature("feature1", REAL);
    fs.add_feature("feature2", REAL);

    boost::shared_ptr<Dense_Feature_Space> fsp(make_unowned_sp(fs));

    Training_Data data(fsp);
    
    int nfv = 10000;

    //float NaN = std::numeric_limits<float>::quiet_NaN();

    for (unsigned i = 0;  i < nfv;  ++i) {
        distribution<float> features;

        features.push_back(i % 3  == 0);
        features.push_back(i % 3  == 0);
        features.push_back(i % 5  == 0);

        boost::shared_ptr<Feature_Set> fset
            = fs.encode(features);

        data.add_example(fset);
    }

    /* Create the decision tree generator */
    Configuration config;
    config.parse_string(config_options, "inbuilt config file");

    GLZ_Classifier_Generator generator;
    generator.configure(config);
    generator.init(fsp, fs.features()[0]);

    distribution<float> training_weights(nfv, 1);

    vector<Feature> features = fs.features();
    features.erase(features.begin(), features.begin() + 1);

    Thread_Context context;

    boost::shared_ptr<Classifier_Impl> classifier
        = generator.generate(context, data, training_weights, features);

    float accuracy JML_UNUSED = classifier->accuracy(data).first;

    cerr << "accuracy = " << accuracy << endl;

    BOOST_CHECK_EQUAL(accuracy, 1);
}

#if 0

BOOST_AUTO_TEST_CASE( test_glz_classifier_missing_features )
{
    /* Create the dataset */

    Dense_Feature_Space fs;
    fs.add_feature("LABEL", Feature_Info(BOOLEAN, false, true));
    fs.add_feature("feature1", REAL);
    fs.add_feature("feature2", REAL);

    boost::shared_ptr<Dense_Feature_Space> fsp(make_unowned_sp(fs));

    Training_Data data(fsp);
    
    int nfv = 10000;

    float NaN = std::numeric_limits<float>::quiet_NaN();

    for (unsigned i = 0;  i < nfv;  ++i) {
        distribution<float> features;
        features.push_back(i % 15 == 0);
        features.push_back(i % 3  == 0);
        features.push_back(i % 5  == 0);

        boost::shared_ptr<Feature_Set> fset
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

#endif

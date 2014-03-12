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


/* Test that we are capable of learning the XOR function in a decision tree.
   It's tricky as the first split point looks like it hasn't learnt anything,
   but is necessary in order to continue.
*/

static const char * xor_dataset = "\
LABEL X Y\n\
1 0 0\n\
0 1 0\n\
0 0 1\n\
1 1 1\n\
";

static const char * config_options = "\
trace=1\n\
";

BOOST_AUTO_TEST_CASE( test_xor_function )
{
    /* Create the dataset */

    Dense_Feature_Space fs;

    Dense_Training_Data data;
    data.init(xor_dataset, xor_dataset + strlen(xor_dataset),
              make_unowned_sp(fs));
    guess_all_info(data, fs, true);

    cerr << "dataset has " << fs.features().size() << " features"
         << endl;

    cerr << fs.features() << endl;

    /* Create the decision tree generator */
    Configuration config;
    config.parse_string(config_options, "inbuilt config file");

    Decision_Tree_Generator generator;
    generator.configure(config);
    generator.init(data.feature_space(),
                   fs.features()[0]);

    boost::multi_array<float, 2> weights(boost::extents[data.example_count()][1]);
    std::fill(weights.data(), weights.data() + data.example_count(), 1.0 / data.example_count() / 2.0);
    
    Thread_Context context;

    Decision_Tree tree
        = generator.train_weighted(context, data, weights, data.all_features(), 3);

    cerr << tree.print();

    // Get the accuracy
    float accuracy JML_UNUSED = tree.accuracy(data).first;

    // Should be 100% accurate if we were able to learn properly
    //BOOST_CHECK_EQUAL(accuracy, 1.0);
}

/* We need to test the following situations:
   - Positive and negative infinity values in the dataset
   - Explicit NaN values versus simply missing features
   - Numbers so close together that there is no number between them
   - That the predict function has the same idea about the split as the split
     function
*/


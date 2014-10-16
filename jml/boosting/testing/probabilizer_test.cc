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
#include <limits>

#include "jml/boosting/probabilizer.h"
#include "jml/utils/vector_utils.h"

using namespace ML;
using namespace std;

using boost::unit_test::test_suite;

#if 1

BOOST_AUTO_TEST_CASE( test_probabilizer_single_dof_sparse )
{
    GLZ_Probabilizer prob;

    vector<distribution<double> > data;
    
    for (unsigned i = 0;  i < 100;  ++i) {
        bool label = (i % 5 == 5);
        float output1 = 0.001 * i;
        if (label) output1 += 0.01;
        distribution<float> output;
        output.push_back(output1);
        output.push_back(1.0 - output1);

        GLZ_Probabilizer::add_data_sparse(data, output, label);

    }

    cerr << "data = " << data << endl;

    distribution<double> params = GLZ_Probabilizer::train_sparse(data);

    cerr << "params = " << params << endl;

}

#endif

BOOST_AUTO_TEST_CASE( test_probabilizer_single_dof_mode2 )
{
    // We had a problem with mode 2 when, for a binary classifier, the
    // probability was less than 50% on the entire training set.  This
    // makes the max column equal to the column for label 1, which
    // meant that the columns for all of the labels were removed.  This
    // tests sets up that situation and tests that the probabilizer is
    // properly learned.

    boost::multi_array<double, 2> outputs(boost::extents[4][100]);
    vector<distribution<double> > correct(2, distribution<double>(100));
    distribution<int> num_correct(2);
    distribution<float> weights(100, 1);

    for (unsigned i = 0;  i < 100;  ++i) {
        bool label = (i % 5 == 4);
        float output1 = 0.001 * i;
        if (label) output1 += 0.1;

        outputs[0][i] = output1;
        outputs[1][i] = 1.0 - output1;
        outputs[2][i] = outputs[1][i];
        outputs[3][i] = 1.0;

        correct[0][i] = !label;
        correct[1][i] = label;

        num_correct[label] += 1;
    }

    GLZ_Probabilizer prob;
    prob.link = LOGIT;
    prob.train_mode2(outputs, correct, num_correct, weights, true /* debug */);

    cerr << "prob.params = " << prob.params << endl;

    distribution<double> bad(4, 0);

    BOOST_CHECK_NE(prob.params[0].total(), 0);
    BOOST_CHECK_NE(prob.params[1].total(), 0);

    BOOST_CHECK_NE(prob.params[0], bad);
    BOOST_CHECK_NE(prob.params[1], bad);

    distribution<float> true_probs, false_probs;

    for (unsigned i = 0;  i < 100;  ++i) {
        bool label = (i % 5 == 4);
        float output1 = 0.001 * i;
        if (label) output1 += 0.1;
        distribution<float> output;
        output.push_back(output1);
        output.push_back(1.0 - output1);

        distribution<float> probs = prob.apply(output);

        cerr << "label " << label << " input " << output << " probs " << probs << endl;

        if (label) true_probs.push_back(probs[1]);
        else false_probs.push_back(probs[1]);
    }

    cerr << "mean_true = " << true_probs.mean() << endl;
    cerr << "mean_false = " << false_probs.mean() << endl;

    BOOST_CHECK_GT(true_probs.mean(), false_probs.mean());
    BOOST_CHECK_GT(true_probs.mean(), 0.20);
    BOOST_CHECK_LT(false_probs.mean(), 0.20);
}

/* twoway_layer_test.cc
   Jeremy Barnes, 9 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Unit tests for the twoway layer class.
*/


#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#undef NDEBUG

#include <boost/test/unit_test.hpp>
#include <boost/multi_array.hpp>
#include "jml/neural/twoway_layer.h"
#include "jml/utils/testing/serialize_reconstitute_include.h"
#include <boost/assign/list_of.hpp>
#include <limits>
#include "bprop_test.h"
#include "jml/arch/exception_handler.h"


using namespace ML;
using namespace ML::DB;
using namespace std;

using boost::unit_test::test_suite;

#if 1

BOOST_AUTO_TEST_CASE( test_serialize_reconstitute_twoway_layer )
{
    Thread_Context context;
    int ni = 2, no = 4;
    Twoway_Layer layer("test", ni, no, TF_TANH, MV_ZERO, context);

    // Test equality operator
    BOOST_CHECK_EQUAL(layer, layer);

    Twoway_Layer layer2 = layer;
    BOOST_CHECK_EQUAL(layer, layer2);
    
    layer2.forward.weights[0][0] -= 1.0;
    BOOST_CHECK(layer != layer2);

    BOOST_CHECK_EQUAL(layer.forward.weights.shape()[0], ni);
    BOOST_CHECK_EQUAL(layer.forward.weights.shape()[1], no);
    BOOST_CHECK_EQUAL(layer.forward.bias.size(), no);
    BOOST_CHECK_EQUAL(layer.forward.missing_replacements.size(), 0);
    BOOST_CHECK_EQUAL(layer.forward.missing_activations.num_elements(), 0);
    BOOST_CHECK_EQUAL(layer.ibias.size(), ni);
    BOOST_CHECK_EQUAL(layer.iscales.size(), ni);
    BOOST_CHECK_EQUAL(layer.oscales.size(), no);
    
    test_serialize_reconstitute(layer);
    test_poly_serialize_reconstitute<Layer>(layer);
}

BOOST_AUTO_TEST_CASE( test_dense_layer_none )
{
    Twoway_Layer layer("test", 2, 1, TF_IDENTITY, MV_NONE);
    layer.forward.weights[0][0] = 0.5;
    layer.forward.weights[1][0] = 2.0;
    layer.forward.bias[0] = 0.0;
    layer.iscales[0] = 1.0;
    layer.iscales[1] = 1.0;
    layer.oscales[0] = 1.0;
    layer.ibias[0] = 0.0;
    layer.ibias[1] = 0.0;

    distribution<float> input
        = boost::assign::list_of<float>(1.0)(-1.0);

    // Check that the basic functions work
    BOOST_REQUIRE_EQUAL(layer.apply(input).size(), 1);
    BOOST_CHECK_EQUAL(layer.apply(input)[0], -1.5);

    // Check the missing values throw an exception
    input[0] = numeric_limits<float>::quiet_NaN();
    {
        JML_TRACE_EXCEPTIONS(false);
        BOOST_CHECK_THROW(layer.apply(input), ML::Exception);
    }

    // Check that the wrong size throws an exception
    input.push_back(2.0);
    input[0] = 1.0;
    {
        JML_TRACE_EXCEPTIONS(false);
        BOOST_CHECK_THROW(layer.apply(input), ML::Exception);
    }

    input.pop_back();

    // Check that the bias works
    layer.forward.bias[0] = 1.0;
    BOOST_CHECK_EQUAL(layer.apply(input)[0], -0.5);

    // Check that there are parameters
    BOOST_CHECK_EQUAL(layer.parameters().parameter_count(), 8);

    // Check the info
    BOOST_CHECK_EQUAL(layer.inputs(), 2);
    BOOST_CHECK_EQUAL(layer.outputs(), 1);
    BOOST_CHECK_EQUAL(layer.name(), "test");

    // Check the copy constructor
    Twoway_Layer layer2 = layer;
    BOOST_CHECK_EQUAL(layer2, layer);
    BOOST_CHECK_EQUAL(layer2.parameters().parameter_count(), 8);

    // Check the assignment operator
    Twoway_Layer layer3;
    BOOST_CHECK(layer3 != layer);
    layer3 = layer;
    BOOST_CHECK_EQUAL(layer3, layer);
    BOOST_CHECK_EQUAL(layer3.parameters().parameter_count(), 8);

    // Make sure that the assignment operator didn't keep a reference
    layer3.forward.weights[0][0] = 5.0;
    BOOST_CHECK_EQUAL(layer.forward.weights[0][0], 0.5);
    BOOST_CHECK_EQUAL(layer3.forward.weights[0][0], 5.0);
    layer3.forward.weights[0][0] = 0.5;
    BOOST_CHECK_EQUAL(layer, layer3);

    // Check fprop (that it gives the same result as apply)
    distribution<float> applied
        = layer.apply(input);

    size_t temp_space_size = layer.fprop_temporary_space_required();

    float temp_space[temp_space_size];

    distribution<float> fproped(layer.outputs());
    layer.fprop(&input[0], temp_space, temp_space_size, &fproped[0]);

    BOOST_CHECK_EQUAL_COLLECTIONS(applied.begin(), applied.end(),
                                  fproped.begin(), fproped.end());

    // Check parameters
    Parameters_Copy<float> params(layer.parameters());
    distribution<float> & param_dist = params.values;

    BOOST_CHECK_EQUAL(param_dist.size(), 8);
    BOOST_CHECK_EQUAL(param_dist.at(0), 0.5);  // weight 0
    BOOST_CHECK_EQUAL(param_dist.at(1), 2.0);  // weight 1
    BOOST_CHECK_EQUAL(param_dist.at(2), 1.0);  // bias

    Thread_Context context;
    layer3.random_fill(-1.0, context);

    BOOST_CHECK(layer != layer3);

    layer3.parameters().set(params);
    
    BOOST_CHECK_EQUAL(layer, layer3);

    // Check backprop
    distribution<float> output_errors(1, 1.0);
    distribution<float> input_errors(layer.inputs());
    Parameters_Copy<float> gradient(layer.parameters());
    gradient.fill(0.0);

    layer.bprop(&input[0], &fproped[0], temp_space, temp_space_size,
                &output_errors[0], &input_errors[0], gradient, 1.0);

    BOOST_CHECK_EQUAL(input_errors.size(), layer.inputs());

    // Check the values of input errors.  It's easy since there's only one
    // weight that contributes to each input (since there's only one output).
    BOOST_CHECK_EQUAL(input_errors[0], layer.forward.weights[0][0]);
    BOOST_CHECK_EQUAL(input_errors[1], layer.forward.weights[1][0]);

    //cerr << "input_errors = " << input_errors << endl;

    // Check that example_weight scales the gradient
    Parameters_Copy<float> gradient2(layer.parameters());
    gradient2.fill(0.0);
    layer.bprop(&input[0], &fproped[0], temp_space, temp_space_size,
                &output_errors[0], &input_errors[0], gradient2, 2.0);

    //cerr << "gradient.values = " << gradient.values << endl;
    //cerr << "gradient2.values = " << gradient2.values << endl;
    
    distribution<float> gradient_times_2 = gradient.values * 2.0;
    
    BOOST_CHECK_EQUAL_COLLECTIONS(gradient2.values.begin(),
                                  gradient2.values.end(),
                                  gradient_times_2.begin(),
                                  gradient_times_2.end());

    // Check that subtracting the parameters from each other returns a zero
    // parameter vector
    layer3.parameters().update(layer3.parameters(), -1.0);

    Parameters_Copy<float> layer3_params(layer3);
    BOOST_CHECK_EQUAL(layer3_params.values.total(), 0.0);

    BOOST_CHECK_EQUAL(layer3.forward.weights[0][0], 0.0);
    BOOST_CHECK_EQUAL(layer3.forward.weights[1][0], 0.0);
    BOOST_CHECK_EQUAL(layer3.forward.bias[0], 0.0);

    layer3.parameters().set(layer.parameters());
    BOOST_CHECK_EQUAL(layer, layer3);

    layer3.parameters().fill(0.0);
    layer3.parameters().update(layer.parameters(), 1.0);
    BOOST_CHECK_EQUAL(layer, layer3);

    layer3.zero_fill();
    layer3.parameters().set(params);
    BOOST_CHECK_EQUAL(layer, layer3);

    Parameters_Copy<double> layer_params2(layer);
    BOOST_CHECK((params.values == layer_params2.values).all());

    // Check that on serialize and reconstitute, the params get properly
    // updated.
    {
        ostringstream stream_out;
        {
            DB::Store_Writer writer(stream_out);
            writer << layer;
        }
        
        istringstream stream_in(stream_out.str());
        DB::Store_Reader reader(stream_in);
        Twoway_Layer layer4;
        reader >> layer4;

        Parameters_Copy<float> params4(layer4.parameters());
        distribution<float> & param_dist4 = params4.values;

        BOOST_REQUIRE_EQUAL(param_dist4.size(), 8);
        BOOST_CHECK_EQUAL(param_dist4.at(0), 0.5);  // weight 0
        BOOST_CHECK_EQUAL(param_dist4.at(1), 2.0);  // weight 1
        BOOST_CHECK_EQUAL(param_dist4.at(2), 1.0);  // bias

        layer4.forward.weights[0][0] = 5.0;
        params4 = layer4.parameters();
        BOOST_CHECK_EQUAL(param_dist4[0], 5.0);
    }
}

BOOST_AUTO_TEST_CASE( test_bprop_identity_double_none )
{
    Thread_Context context;
    Twoway_Layer layer("test", 20, 40, TF_IDENTITY, MV_NONE, context);

    bprop_test<double>(layer, context);
    bprop_test_backward<double>(layer, context, 0.1);

    // We have to leave a big error margin due to numerical issues in the
    // (long) calculation
    // A better test would allow higher error magnitudes when the gradients
    // were small.  We should also, at some stage, do a sensitivity
    // analysis to figure out how to improve the numerical stability of the
    // algorithms.
    bprop_test_reconstruct<double>(layer, context, 5.0);
}

BOOST_AUTO_TEST_CASE( test_bprop_tanh_double_none )
{
    Thread_Context context;
    Twoway_Layer layer("test", 20, 40, TF_TANH, MV_NONE, context);

    bprop_test<double>(layer, context);
}

BOOST_AUTO_TEST_CASE( test_bprop_identity_float_none )
{
    Thread_Context context;
    Twoway_Layer layer("test", 20, 40, TF_IDENTITY, MV_NONE, context);

    bprop_test<double>(layer, context);
}

BOOST_AUTO_TEST_CASE( test_bprop_identity_float_zero )
{
    Thread_Context context;
    Twoway_Layer layer("test", 20, 40, TF_IDENTITY, MV_ZERO, context);

    bprop_test<double>(layer, context);
}

BOOST_AUTO_TEST_CASE( test_bprop_identity_double_zero )
{
    Thread_Context context;
    Twoway_Layer layer("test", 20, 40, TF_IDENTITY, MV_ZERO, context);

    bprop_test<double>(layer, context);
}

BOOST_AUTO_TEST_CASE( test_bprop_identity_float_input )
{
    Thread_Context context;
    Twoway_Layer layer("test", 20, 40, TF_IDENTITY, MV_INPUT, context);

    bprop_test<double>(layer, context);
}

BOOST_AUTO_TEST_CASE( test_bprop_identity_float_dense )
{
    Thread_Context context;
    Twoway_Layer layer("test", 20, 40, TF_IDENTITY, MV_DENSE, context);

    bprop_test<double>(layer, context);
}

BOOST_AUTO_TEST_CASE( test_bprop_identity_double_dense )
{
    Thread_Context context;
    Twoway_Layer layer("test", 20, 40, TF_IDENTITY, MV_DENSE, context);

    bprop_test<double>(layer, context);
}

BOOST_AUTO_TEST_CASE( test_bbprop_identity_double_none1 )
{
    Thread_Context context;
    context.seed(123);
    Twoway_Layer layer("test", 1, 1, TF_IDENTITY, MV_NONE, context);

    bbprop_test<double>(layer, context);
}

BOOST_AUTO_TEST_CASE( test_bbprop_identity_double_none2 )
{
    Thread_Context context;
    context.seed(123);
    Twoway_Layer layer("test", 5, 5, TF_IDENTITY, MV_NONE, context);

    bbprop_test<double>(layer, context);
}

BOOST_AUTO_TEST_CASE( test_bbprop_tanh_double_none1 )
{
    Thread_Context context;
    context.seed(123);
    Twoway_Layer layer("test", 1, 1, TF_TANH, MV_NONE, context);

    bbprop_test<double>(layer, context);
}

BOOST_AUTO_TEST_CASE( test_bbprop_tanh_double_none2 )
{
    Thread_Context context;
    context.seed(123);
    Twoway_Layer layer("test", 5, 5, TF_TANH, MV_NONE, context);

    bbprop_test<double>(layer, context);
}

BOOST_AUTO_TEST_CASE( test_bbprop_identity_double_none )
{
    Thread_Context context;
    Twoway_Layer layer("test", 5, 5, TF_IDENTITY, MV_NONE, context);

    cerr << "layer = " << layer << endl;

    cerr << endl << endl << "*** TESTING BBPROP FORWARD" << endl;
    bbprop_test<double>(layer, context);

    cerr << endl << endl << "*** TESTING BBPROP BACKWARD" << endl;
    bbprop_test_backward<double>(layer, context);

    cerr << endl << endl << "*** TESTING BBPROP RECONSTRUCTION" << endl;
    // We have to leave a big error margin due to numerical issues in the
    // (long) calculation
    bbprop_test_reconstruct<double>(layer, context, 3.0);
}
#endif

#if 1
BOOST_AUTO_TEST_CASE( test_bbprop_tanh_double_none )
{
    Thread_Context context;
    Twoway_Layer layer("test", 1, 1, TF_IDENTITY, MV_NONE, context);

    layer.forward.weights[0][0] = 1.0;
    layer.forward.bias[0] = 0.0;
    layer.ibias[0] = 0.0;

    cerr << "layer = " << layer << endl;

    cerr << endl << endl << "*** TESTING BBPROP FORWARD" << endl;
    bbprop_test<double>(layer, context);

    cerr << endl << endl << "*** TESTING BBPROP BACKWARD" << endl;
    bbprop_test_backward<double>(layer, context);

    cerr << endl << endl << "*** TESTING BBPROP RECONSTRUCTION" << endl;
    // We have to leave a big error margin due to numerical issues in the
    // (long) calculation
    bbprop_test_reconstruct<double>(layer, context, 3.0);
}

BOOST_AUTO_TEST_CASE( test_bbprop_tanh_double_none3 )
{
    Thread_Context context;
    Twoway_Layer layer("test", 1, 1, TF_TANH, MV_NONE, context);

    cerr << "layer = " << layer << endl;

    cerr << endl << endl << "*** TESTING BBPROP FORWARD" << endl;
    bbprop_test<double>(layer, context);

    cerr << endl << endl << "*** TESTING BBPROP BACKWARD" << endl;
    bbprop_test_backward<double>(layer, context);

    cerr << endl << endl << "*** TESTING BBPROP RECONSTRUCTION" << endl;
    // We have to leave a big error margin due to numerical issues in the
    // (long) calculation
    bbprop_test_reconstruct<double>(layer, context, 3.0);
}
#endif

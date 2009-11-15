/* dense_layer_test.cc
   Jeremy Barnes, 28 October 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Unit tests for the dense layer class.
*/


#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#undef NDEBUG

#include <boost/test/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>
#include <boost/multi_array.hpp>
#include "neural/dense_layer.h"
#include "utils/testing/serialize_reconstitute_include.h"
#include "utils/check_not_nan.h"
#include <boost/assign/list_of.hpp>
#include <limits>
#include "bprop_test.h"

using namespace ML;
using namespace ML::DB;
using namespace std;

using boost::unit_test::test_suite;

BOOST_AUTO_TEST_CASE( test_serialize_reconstitute_dense_layer0a )
{
    Thread_Context context;
    int ni = 2, no = 4;
    Dense_Layer<float> layer("test", ni, no, TF_TANH, MV_ZERO, context);

    // Test equality operator
    BOOST_CHECK_EQUAL(layer, layer);

    Dense_Layer<float> layer2 = layer;
    BOOST_CHECK_EQUAL(layer, layer2);
    
    layer2.weights[0][0] -= 1.0;
    BOOST_CHECK(layer != layer2);

    BOOST_CHECK_EQUAL(layer.weights.shape()[0], ni);
    BOOST_CHECK_EQUAL(layer.weights.shape()[1], no);
    BOOST_CHECK_EQUAL(layer.bias.size(), no);
    BOOST_CHECK_EQUAL(layer.missing_replacements.size(), 0);
    BOOST_CHECK_EQUAL(layer.missing_activations.num_elements(), 0);
    
    test_serialize_reconstitute(layer);
    test_poly_serialize_reconstitute<Layer>(layer);
}

BOOST_AUTO_TEST_CASE( test_serialize_reconstitute_dense_layer0b )
{
    Thread_Context context;
    Dense_Layer<float> layer("test", 20, 40, TF_TANH, MV_INPUT, context);
    BOOST_CHECK_EQUAL(layer.weights.shape()[0], 20);
    BOOST_CHECK_EQUAL(layer.weights.shape()[1], 40);
    BOOST_CHECK_EQUAL(layer.bias.size(), 40);
    BOOST_CHECK_EQUAL(layer.missing_replacements.size(), 20);
    BOOST_CHECK_EQUAL(layer.missing_activations.num_elements(), 0);
    test_serialize_reconstitute(layer);
    test_poly_serialize_reconstitute<Layer>(layer);
}

BOOST_AUTO_TEST_CASE( test_serialize_reconstitute_dense_layer0c )
{
    Thread_Context context;
    Dense_Layer<float> layer("test", 20, 40, TF_IDENTITY, MV_DENSE, context);
    BOOST_CHECK_EQUAL(layer.weights.shape()[0], 20);
    BOOST_CHECK_EQUAL(layer.weights.shape()[1], 40);
    BOOST_CHECK_EQUAL(layer.bias.size(), 40);
    BOOST_CHECK_EQUAL(layer.missing_replacements.size(), 0);
    BOOST_CHECK_EQUAL(layer.missing_activations.num_elements(), 800);
    test_serialize_reconstitute(layer);
    test_poly_serialize_reconstitute<Layer>(layer);
}

BOOST_AUTO_TEST_CASE( test_serialize_reconstitute_dense_layer0d )
{
    Thread_Context context;
    Dense_Layer<float> layer("test", 20, 40, TF_LOGSIG, MV_NONE, context);
    BOOST_CHECK_EQUAL(layer.weights.shape()[0], 20);
    BOOST_CHECK_EQUAL(layer.weights.shape()[1], 40);
    BOOST_CHECK_EQUAL(layer.bias.size(), 40);
    BOOST_CHECK_EQUAL(layer.missing_replacements.size(), 0);
    BOOST_CHECK_EQUAL(layer.missing_activations.num_elements(), 0);
    test_serialize_reconstitute(layer);
    test_poly_serialize_reconstitute<Layer>(layer);
}

BOOST_AUTO_TEST_CASE( test_dense_layer_none )
{
    Dense_Layer<float> layer("test", 2, 1, TF_IDENTITY, MV_NONE);
    layer.weights[0][0] = 0.5;
    layer.weights[1][0] = 2.0;
    layer.bias[0] = 0.0;

    distribution<float> input
        = boost::assign::list_of<float>(1.0)(-1.0);

    // Check that the basic functions work
    BOOST_REQUIRE_EQUAL(layer.apply(input).size(), 1);
    BOOST_CHECK_EQUAL(layer.apply(input)[0], -1.5);

    // Check the missing values throw an exception
    input[0] = numeric_limits<float>::quiet_NaN();
    BOOST_CHECK_THROW(layer.apply(input), ML::Exception);

    // Check that the wrong size throws an exception
    input.push_back(2.0);
    input[0] = 1.0;
    BOOST_CHECK_THROW(layer.apply(input), ML::Exception);

    input.pop_back();

    // Check that the bias works
    layer.bias[0] = 1.0;
    BOOST_CHECK_EQUAL(layer.apply(input)[0], -0.5);

    // Check that there are parameters
    BOOST_CHECK_EQUAL(layer.parameters().parameter_count(), 3);

    // Check the info
    BOOST_CHECK_EQUAL(layer.inputs(), 2);
    BOOST_CHECK_EQUAL(layer.outputs(), 1);
    BOOST_CHECK_EQUAL(layer.name(), "test");

    // Check the copy constructor
    Dense_Layer<float> layer2 = layer;
    BOOST_CHECK_EQUAL(layer2, layer);
    BOOST_CHECK_EQUAL(layer2.parameters().parameter_count(), 3);

    // Check the assignment operator
    Dense_Layer<float> layer3;
    BOOST_CHECK(layer3 != layer);
    layer3 = layer;
    BOOST_CHECK_EQUAL(layer3, layer);
    BOOST_CHECK_EQUAL(layer3.parameters().parameter_count(), 3);

    // Make sure that the assignment operator didn't keep a reference
    layer3.weights[0][0] = 5.0;
    BOOST_CHECK_EQUAL(layer.weights[0][0], 0.5);
    BOOST_CHECK_EQUAL(layer3.weights[0][0], 5.0);
    layer3.weights[0][0] = 0.5;
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

    BOOST_CHECK_EQUAL(param_dist.size(), 3);
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
    BOOST_CHECK_EQUAL(input_errors[0], layer.weights[0][0]);
    BOOST_CHECK_EQUAL(input_errors[1], layer.weights[1][0]);

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

    // Check that the result of numerical differentiation is the same as the
    // output of the bprop routine.  It should be, since we have a linear
    // loss function.

#if 0
    // First, check the input gradients
    for (unsigned i = 0;  i < 2;  ++i) {
        distribution<float> inputs2 = inputs;
        inputs2[i] += 1.0;

        distribution<float> outputs2 = layer.apply(inputs2);

        BOOST_CHECK_EQUAL(outputs2[0], output[0]...);
    }
#endif


    // Check that subtracting the parameters from each other returns a zero
    // parameter vector
    layer3.parameters().update(layer3.parameters(), -1.0);

    Parameters_Copy<float> layer3_params(layer3);
    BOOST_CHECK_EQUAL(layer3_params.values.total(), 0.0);

    BOOST_CHECK_EQUAL(layer3.weights[0][0], 0.0);
    BOOST_CHECK_EQUAL(layer3.weights[1][0], 0.0);
    BOOST_CHECK_EQUAL(layer3.bias[0], 0.0);

    layer3.parameters().set(layer.parameters());
    BOOST_CHECK_EQUAL(layer, layer3);

    layer3.zero_fill();
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
        Dense_Layer<float> layer4;
        reader >> layer4;

        Parameters_Copy<float> params4(layer4.parameters());
        distribution<float> & param_dist4 = params4.values;

        BOOST_REQUIRE_EQUAL(param_dist4.size(), 3);
        BOOST_CHECK_EQUAL(param_dist4.at(0), 0.5);  // weight 0
        BOOST_CHECK_EQUAL(param_dist4.at(1), 2.0);  // weight 1
        BOOST_CHECK_EQUAL(param_dist4.at(2), 1.0);  // bias

        layer4.weights[0][0] = 5.0;
        params4 = layer4.parameters();
        BOOST_CHECK_EQUAL(param_dist4[0], 5.0);
    }
}

BOOST_AUTO_TEST_CASE( test_serialize_reconstitute_dense_layer1 )
{
    Thread_Context context;
    Dense_Layer<float> layer("test", 200, 400, TF_TANH, MV_ZERO, context);
    test_serialize_reconstitute(layer);
    test_poly_serialize_reconstitute<Layer>(layer);
}

BOOST_AUTO_TEST_CASE( test_serialize_reconstitute_dense_layer2 )
{
    Thread_Context context;
    Dense_Layer<float> layer("test", 200, 400, TF_TANH, MV_INPUT, context);
    test_serialize_reconstitute(layer);
    test_poly_serialize_reconstitute<Layer>(layer);
}

BOOST_AUTO_TEST_CASE( test_serialize_reconstitute_dense_layer3 )
{
    Thread_Context context;
    Dense_Layer<float> layer("test", 200, 400, TF_TANH, MV_DENSE, context);
    test_serialize_reconstitute(layer);
    test_poly_serialize_reconstitute<Layer>(layer);
}

BOOST_AUTO_TEST_CASE( test_serialize_reconstitute_dense_layer_double )
{
    Thread_Context context;
    Dense_Layer<double> layer("test", 200, 400, TF_TANH, MV_DENSE, context);
    test_serialize_reconstitute(layer);
    test_poly_serialize_reconstitute<Layer>(layer);
}

BOOST_AUTO_TEST_CASE( test_single_precision_accuracy )
{
    Thread_Context context;
    Dense_Layer<double> layerd("test", 20, 40, TF_IDENTITY, MV_NONE, context);
    Dense_Layer<float> layerf("test", 20, 40, TF_IDENTITY, MV_NONE);

    // Get the parameters as floats, then copy them into the two so that
    // they are equivalent.
    Parameters_Copy<float> params(layerd);
    layerd.parameters().set(params);
    layerf.parameters().set(params);

    int ni = layerd.inputs(), no = layerd.outputs();

    BOOST_REQUIRE(ni > 0);
    BOOST_REQUIRE(no > 0);

    distribution<float> inputf(ni);
    for (unsigned i = 0;  i < ni;  ++i)
        inputf[i] = 0.5 - context.random01();

    distribution<double> inputd(inputf);

    // Check the result of apply
    // Do in both single and double precision.  The result should be
    // identical, since we accumulate in double precision and all of the
    // stored values are equivalent between float and double
    distribution<float> resultdd = layerd.apply(inputd).cast<float>();
    distribution<float> resultdf = layerd.apply(inputf);
    distribution<float> resultfd = layerf.apply(inputd).cast<float>();
    distribution<float> resultff = layerf.apply(inputf);

    BOOST_CHECK_EQUAL_COLLECTIONS(resultdd.begin(), resultdd.end(),
                                  resultdf.begin(), resultdf.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(resultdd.begin(), resultdd.end(),
                                  resultfd.begin(), resultfd.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(resultdd.begin(), resultdd.end(),
                                  resultff.begin(), resultff.end());


    // Check the results of the fprop in a similar way
    size_t temp_space_size = layerd.fprop_temporary_space_required();
    double ddtemp_space[temp_space_size];
    float  dftemp_space[temp_space_size];
    double fdtemp_space[temp_space_size];
    float  fftemp_space[temp_space_size];

    resultdd = layerd.fprop(inputd, ddtemp_space, temp_space_size)
                   .cast<float>();
    resultdf = layerd.fprop(inputf, dftemp_space, temp_space_size);
    resultfd = layerf.fprop(inputd, fdtemp_space, temp_space_size)
                   .cast<float>();
    resultff = layerf.fprop(inputf, fftemp_space, temp_space_size);


    BOOST_CHECK_EQUAL_COLLECTIONS(resultdd.begin(), resultdd.end(),
                                  resultdf.begin(), resultdf.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(resultdd.begin(), resultdd.end(),
                                  resultfd.begin(), resultfd.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(resultdd.begin(), resultdd.end(),
                                  resultff.begin(), resultff.end());


    // Now for the bprop
    distribution<double> doutput_errors(no, 1.0);
    distribution<float> foutput_errors(no, 1.0);

    Parameters_Copy<float> ddgradient(params, 0.0);
    Parameters_Copy<float> dfgradient(params, 0.0);
    Parameters_Copy<float> fdgradient(params, 0.0);
    Parameters_Copy<float> ffgradient(params, 0.0);
    
    distribution<float> ddinput_errors
        = layerd.bprop(inputd, resultdd.cast<double>(),
                       ddtemp_space, temp_space_size,
                       doutput_errors, ddgradient, 1.0).cast<float>();
    
    distribution<float> dfinput_errors
        = layerd.bprop(inputf, resultdf,
                       dftemp_space, temp_space_size,
                       foutput_errors, dfgradient, 1.0);
    
    BOOST_CHECK_EQUAL_COLLECTIONS(ddgradient.values.begin(),
                                  ddgradient.values.end(),
                                  dfgradient.values.begin(),
                                  dfgradient.values.end());

    BOOST_CHECK_EQUAL_COLLECTIONS(ddinput_errors.begin(),
                                  ddinput_errors.end(),
                                  dfinput_errors.begin(),
                                  dfinput_errors.end());

    distribution<float> fdinput_errors
        = layerf.bprop(inputd, resultdd.cast<double>(),
                       fdtemp_space, temp_space_size,
                       doutput_errors, fdgradient, 1.0).cast<float>();
    

    BOOST_CHECK_EQUAL_COLLECTIONS(ddgradient.values.begin(),
                                  ddgradient.values.end(),
                                  fdgradient.values.begin(),
                                  fdgradient.values.end());

    BOOST_CHECK_EQUAL_COLLECTIONS(ddinput_errors.begin(),
                                  ddinput_errors.end(),
                                  fdinput_errors.begin(),
                                  fdinput_errors.end());
    
    distribution<float> ffinput_errors
        = layerf.bprop(inputf, resultdf,
                       dftemp_space, temp_space_size,
                       foutput_errors, ffgradient, 1.0);

    BOOST_CHECK_EQUAL_COLLECTIONS(ddgradient.values.begin(),
                                  ddgradient.values.end(),
                                  ffgradient.values.begin(),
                                  ffgradient.values.end());

    BOOST_CHECK_EQUAL_COLLECTIONS(ddinput_errors.begin(),
                                  ddinput_errors.end(),
                                  ffinput_errors.begin(),
                                  ffinput_errors.end());
}

BOOST_AUTO_TEST_CASE( test_bprop_identity_double_none )
{
    Thread_Context context;
    Dense_Layer<double> layer("test", 20, 40, TF_IDENTITY, MV_NONE, context);

    bprop_test<double>(layer, context);
}

BOOST_AUTO_TEST_CASE( test_bprop_tanh_double_none )
{
    Thread_Context context;
    Dense_Layer<double> layer("test", 20, 40, TF_TANH, MV_NONE, context);

    bprop_test<double>(layer, context);
}

BOOST_AUTO_TEST_CASE( test_bprop_identity_float_none )
{
    Thread_Context context;
    Dense_Layer<float> layer("test", 20, 40, TF_IDENTITY, MV_NONE, context);

    bprop_test<double>(layer, context);
}

BOOST_AUTO_TEST_CASE( test_bprop_identity_float_zero )
{
    Thread_Context context;
    Dense_Layer<float> layer("test", 20, 40, TF_IDENTITY, MV_ZERO, context);

    bprop_test<double>(layer, context);
}

BOOST_AUTO_TEST_CASE( test_bprop_identity_double_zero )
{
    Thread_Context context;
    Dense_Layer<double> layer("test", 20, 40, TF_IDENTITY, MV_ZERO, context);

    bprop_test<double>(layer, context);
}

BOOST_AUTO_TEST_CASE( test_bprop_identity_float_input )
{
    Thread_Context context;
    Dense_Layer<float> layer("test", 20, 40, TF_IDENTITY, MV_INPUT, context);

    bprop_test<double>(layer, context);
}

BOOST_AUTO_TEST_CASE( test_bprop_identity_float_dense )
{
    Thread_Context context;
    Dense_Layer<float> layer("test", 20, 40, TF_IDENTITY, MV_DENSE, context);

    bprop_test<double>(layer, context);
}

BOOST_AUTO_TEST_CASE( test_bprop_identity_double_dense )
{
    Thread_Context context;
    Dense_Layer<double> layer("test", 20, 40, TF_IDENTITY, MV_DENSE, context);

    bprop_test<double>(layer, context);
}

template<typename Float, class Layer>
void bbprop_test(Layer & layer, Thread_Context & context)
{
    double tolerance = get_tolerance(Float());
    
    int ni = layer.inputs(), no = layer.outputs();

    BOOST_REQUIRE(ni > 0);
    BOOST_REQUIRE(no > 0);

    // Create a random input vector
    distribution<Float> input(ni);
    for (unsigned i = 0;  i < ni;  ++i)
        input[i] = 0.5 - context.random01();

    if (layer.supports_missing_inputs()) {
        for (unsigned i = 0;  i < ni;  i += 2)
            input[i] = numeric_limits<float>::quiet_NaN();
    }

    if (tolerance == -1.0)
        tolerance = get_tolerance(Float());

    // If we set the derror/doutput to 1 for one output and to zero for the
    // rest, then the error calculated should equal the change in that output
    // per unit change in the parameter.  We can use this to test all of the
    // parameters.

    BOOST_REQUIRE_EQUAL(input.size(), ni);
    BOOST_REQUIRE(ni > 0);
    BOOST_REQUIRE(no > 0);

    // Get the original parameters
    const Parameters_Copy<Float> params(layer);

    // Set them from the copied values
    layer.parameters().set(params);

    size_t temp_space_size = layer.fprop_temporary_space_required();
    Float temp_space[temp_space_size];

    // Perform the original fprop and bprop to get the baseline
    distribution<Float> output
        = layer.fprop(input, temp_space, temp_space_size);

    // Let's say that we want the output to be zero, and we use the
    // squared error, so error = sum(output^2).  In that case, the
    // dError/doutput = 2 * output, and the d2error/doutput2 is 2.

    distribution<Float> derrors = 2.0 * output;
    distribution<Float> d2errors(no, 2.0);

    // Hessian computation
    // We only need the diagonal entries, which are what we are trying to
    // compute here.  We use the finite difference method.

    Parameters_Copy<Float> gradient0(layer, 0.0);
    distribution<Float> input_errors0(ni);

    // Original gradient (with no delta)
    layer.bprop(&input[0], &output[0], temp_space,
                temp_space_size,
                &derrors[0],
                &input_errors0[0],
                gradient0,
                1.0);

    // For each parameter, calculate numerically the gradient and test
    // against the analytical version
    int np = params.values.size();

    Parameters_Copy<Float> dgradienth(layer, 0.0);

    double delta = 1e-5;

    for (unsigned i = 0;  i < np;  ++i) {

        // Get a value of delta that registers
        double real_delta = 0.0;
        for (double delta_to_try = delta;
             real_delta == 0.0;
             delta_to_try *= 10.0) {

            Parameters_Copy<Float> new_params = params;

            double old_value = new_params.values[i];

            double new_value1 = new_params.values[i] * 1.001;
            double new_value2
                = sign(new_params.values[i])
                * (abs(new_params.values[i]) + delta_to_try);

            new_params.values[i] = (abs(new_value1) > abs(new_value2)
                                    ? new_value1 : new_value2);

            layer.parameters().set(new_params);
            
            Parameters_Copy<Float> set_params(layer);

            real_delta = set_params.values[i] - old_value;
        }

        if (real_delta == 0.0)
            throw Exception("real_delta is zero");

        // Perform a new fprop
        distribution<Float> new_output
            = layer.fprop(input, temp_space, temp_space_size);

        // New error
        distribution<Float> new_derrors
            = 2.0 * new_output;
        
        Parameters_Copy<Float> gradient_eps(layer, 0.0);

        // Perform a new bprop
        layer.bprop(&input[0], &new_output[0], temp_space,
                    temp_space_size,
                    &new_derrors[0],
                    0, gradient_eps,
                    1.0);

        // The value of the hessian is (gradient_eps - gradient0)/delta
        double epsilon = (gradient_eps.values[i] - gradient0.values[i])
            / real_delta;

        cerr << "i = " << i << " val = " << params.values[i]
             << " delta = " << real_delta << " g1 = "
             << gradient_eps.values[i] << " g0 = "
             << gradient0.values[i]
             << " eps = " << epsilon << endl;
        
        dgradienth.values[i] = epsilon;
    }

    // Put the parameters back to their old state
    layer.parameters().set(params);

    distribution<Float> input_derrorsh(ni);

    // Calculate numerically the rate of change of dE/di with respect to
    // di.
    for (unsigned i = 0;  i < ni;  ++i) {

        distribution<Float> input2 = input;

        // Get a value of delta that registers
        double real_delta = 0.0;
        for (double delta_to_try = delta;
             real_delta == 0.0;
             delta_to_try *= 10.0) {

            Parameters_Copy<Float> new_params = params;

            double new_value1 = input[i] * 1.001;
            double new_value2
                = sign(input[i]) * (abs(input[i]) + delta_to_try);

            input2[i] = (abs(new_value1) > abs(new_value2)
                         ? new_value1 : new_value2);

            real_delta = input2[i] - input[i];
        }

        if (real_delta == 0.0)
            throw Exception("real_delta is zero");

        // Perform a new fprop
        distribution<Float> new_output
            = layer.fprop(input2, temp_space, temp_space_size);
        
        // New error
        distribution<Float> new_derrors
            = 2.0 * new_output;
        
        Parameters_Copy<Float> gradient_eps(layer, 0.0);

        Float input_errors[ni];

        // Perform a new bprop
        layer.bprop(&input2[0], &new_output[0], temp_space,
                    temp_space_size,
                    &new_derrors[0],
                    input_errors, gradient_eps,
                    1.0);
        
        // The value of the hessian is (gradient_eps - gradient0)/delta
        double epsilon = (input_errors[i] - input_errors0[i])
            / real_delta;

        input_derrorsh[i] = epsilon;
    }

    // Perform the original fprop again to set temp_space back to what it
    // should be.
    output = layer.fprop(input, temp_space, temp_space_size);

    Parameters_Copy<Float> gradient1(layer, 0.0);
    Parameters_Copy<Float> dgradient1(layer, 0.0);
    distribution<Float> input_errors1(ni);
    distribution<Float> input_derrors1(ni);

    layer.bbprop_jacobian(&input[0], &output[0], temp_space,
                          temp_space_size,
                          &derrors[0],
                          &d2errors[0],
                          &input_errors1[0],
                          &input_derrors1[0],
                          gradient1, &dgradient1,
                          1.0);
    
    Parameters_Copy<Float> gradient2(layer, 0.0);
    Parameters_Copy<Float> dgradient2(layer, 0.0);
    distribution<Float> input_errors2(ni);
    distribution<Float> input_derrors2(ni);

    layer.bbprop(&input[0], &output[0], temp_space,
                 temp_space_size,
                 &derrors[0],
                 &d2errors[0],
                 &input_errors2[0],
                 &input_derrors2[0],
                 gradient2, &dgradient2,
                 1.0);

    cerr << "gradient1 = " << gradient1.values << endl;
    cerr << "gradient2 = " << gradient2.values << endl;
    cerr << endl;

    cerr << "dgradienth = " << dgradienth.values << endl;
    cerr << "dgradient1 = " << dgradient1.values << endl;
    cerr << "dgradient2 = " << dgradient2.values << endl;
    cerr << endl;

    cerr << "input_derrorsh = " << input_derrorsh << endl;
    cerr << "input_derrors1 = " << input_derrors1 << endl;
    cerr << "input_derrors2 = " << input_derrors2 << endl;
}

BOOST_AUTO_TEST_CASE( test_bbprop_identity_double_none1 )
{
    Thread_Context context;
    context.seed(123);
    Dense_Layer<double> layer("test", 1, 1, TF_IDENTITY, MV_NONE, context);

    bbprop_test<double>(layer, context);
}

BOOST_AUTO_TEST_CASE( test_bbprop_identity_double_none2 )
{
    Thread_Context context;
    context.seed(123);
    Dense_Layer<double> layer("test", 5, 5, TF_IDENTITY, MV_NONE, context);

    bbprop_test<double>(layer, context);
}

BOOST_AUTO_TEST_CASE( test_bbprop_tanh_double_none1 )
{
    Thread_Context context;
    context.seed(123);
    Dense_Layer<double> layer("test", 1, 1, TF_TANH, MV_NONE, context);

    bbprop_test<double>(layer, context);
}

BOOST_AUTO_TEST_CASE( test_bbprop_tanh_double_none2 )
{
    Thread_Context context;
    context.seed(123);
    Dense_Layer<double> layer("test", 5, 5, TF_TANH, MV_NONE, context);

    bbprop_test<double>(layer, context);
}


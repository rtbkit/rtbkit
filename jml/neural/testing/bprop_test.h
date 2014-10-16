/* bprop_test.h                                                    -*- C++ -*-
   Jeremy Barnes, 12 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Include file that allows backpropagation to be tested for any layer
   derivative.
*/

#ifndef __jml__neural__testing__bprop_test_h__
#define __jml__neural__testing__bprop_test_h__

#include <boost/test/floating_point_comparison.hpp>
#include "jml/neural/reverse_layer_adaptor.h"
#include "jml/neural/reconstruct_layer_adaptor.h"

namespace {

template<typename F>
F sign(F val)
{
    return (val >= 0 ? 1.0 : -1.0);
}

} // file scope

template<class Float, class Layer, class Float2>
void bprop_test(Layer & layer, const distribution<Float2> & input_,
                double epsilon, double tolerance)
{
    // If we set the derror/doutput to 1 for one output and to zero for the
    // rest, then the error calculated should equal the change in that output
    // per unit change in the parameter.  We can use this to test all of the
    // parameters.

    int ni = layer.inputs(), no = layer.outputs();

    BOOST_REQUIRE_EQUAL(input_.size(), ni);
    BOOST_REQUIRE(ni > 0);
    BOOST_REQUIRE(no > 0);

    // Create a random input vector

    distribution<Float> input(input_);

    size_t temp_space_size = layer.fprop_temporary_space_required();
    Float temp_space_storage[temp_space_size + 2];
    temp_space_storage[0] = 0.123456;
    temp_space_storage[temp_space_size + 1] = 0.8765432;

    Float * temp_space = temp_space_storage + 1;

    // Put values here to make sure that the space is used
    std::fill(temp_space, temp_space + temp_space_size,
              numeric_limits<float>::quiet_NaN());


    // Get the original parameters
    const Parameters_Copy<Float> params(layer);

    // Set them from the copied values
    layer.parameters().set(params);

    // Perform the original fprop and bprop to get the baseline
    distribution<Float> baseline_output
        = layer.fprop(input, temp_space, temp_space_size);
    
    // Make sure that the temp space was used but not the guard values at
    // either end
    BOOST_CHECK_EQUAL(temp_space_storage[0],
                      (Float)0.123456);
    BOOST_CHECK_EQUAL(temp_space_storage[temp_space_size + 1],
                      (Float)0.8765432);
    for (unsigned i = 0;  i < temp_space_size;  ++i)
        BOOST_CHECK(!std::isnan(temp_space[i]));
    

    // Check that there is zero gradient if the error gradients are all zero
    distribution<Float> output_errors(no);

    Parameters_Copy<Float> gradient(layer, 0.0);
    BOOST_CHECK((gradient.values == 0.0).all());

    distribution<Float> input_errors
        = layer.bprop(input, baseline_output,
                      temp_space, temp_space_size,
                      output_errors, gradient, 1.0);

    BOOST_CHECK((input_errors == 0.0).all());
    BOOST_CHECK((gradient.values == 0.0).all());

    // We take our output error to be the sum of the outputs, which makes the
    // derror/doutput 1 for all of the outputs
    output_errors.fill(1.0);
    
    gradient.values.fill(0.0);

    input_errors
        = layer.bprop(input, baseline_output,
                      temp_space, temp_space_size,
                      output_errors, gradient, 1.0);

    // For each parameter, calculate numerically the gradient and test
    // against the analytical version
    int np = params.values.size();

    for (unsigned i = 0;  i < np;  ++i) {

        // Get a value of epsilon that registers
        double real_epsilon = 0.0;
        for (double epsilon_to_try = epsilon;
             real_epsilon == 0.0;
             epsilon_to_try *= 10.0) {

            Parameters_Copy<Float> new_params = params;

            double old_value = new_params.values[i];

            double new_value1 = new_params.values[i] * 1.001;
            double new_value2
                = sign(new_params.values[i])
                * (abs(new_params.values[i]) + epsilon_to_try);

            new_params.values[i] = (abs(new_value1) > abs(new_value2)
                                    ? new_value1 : new_value2);

            layer.parameters().set(new_params);
            
            Parameters_Copy<Float> set_params(layer);

            real_epsilon = set_params.values[i] - old_value;
        }

        if (real_epsilon == 0.0)
            throw Exception("real_epsilon is zero");

        // Put values here to make sure that the space is used
        std::fill(temp_space, temp_space + temp_space_size,
                  numeric_limits<float>::quiet_NaN());

        // Perform a new fprop to see the change in the error
        distribution<Float> new_output
            = layer.fprop(input, temp_space, temp_space_size);
        
        // Make sure that the temp space was used but not the guard values at
        // either end
        BOOST_CHECK_EQUAL(temp_space_storage[0],
                          (Float)0.123456);
        BOOST_CHECK_EQUAL(temp_space_storage[temp_space_size + 1],
                          (Float)0.8765432);
        for (unsigned j = 0;  j < temp_space_size;  ++j)
            BOOST_CHECK(!std::isnan(temp_space[j]));

        double delta = (new_output - baseline_output).total();

        double this_gradient = delta / real_epsilon;
        double calc_gradient = gradient.values[i];

#if 0
        cerr << "differences = " << (new_output - baseline_output)
             << " delta = " << delta << " epsilon = " << epsilon
             << " real_epsilon = " << real_epsilon << " this_gradient "
             << this_gradient << " calc_gradient " << calc_gradient
             << endl;
#endif

        if (abs(this_gradient - calc_gradient)
            / std::max(abs(this_gradient), abs(calc_gradient))
            > tolerance / 100.0)
            cerr << "error is on parameter " << i << " "
                 << params.parameter_info(i) << endl;

        BOOST_CHECK_CLOSE(this_gradient, calc_gradient, tolerance);
    }

    // Do the same for the input parameters, in order to check the propagation
    // of the gradients

    layer.parameters().set(params);

    for (unsigned i = 0;  i < ni;  ++i) {
        distribution<Float> input2 = input;

        if (std::isnan(input2[i])) continue;  // can't take deriv

        double old_value = input2[i];
        input2[i] += epsilon;
        double real_epsilon = input2[i] - old_value;

        if (real_epsilon == 0.0)
            throw Exception("epsilon was too low");


        // Put values here to make sure that the space is used
        std::fill(temp_space, temp_space + temp_space_size,
                  numeric_limits<float>::quiet_NaN());

        // Perform a new fprop to see the change in the error
        distribution<Float> new_output
            = layer.fprop(input2, temp_space, temp_space_size);
        
        // Make sure that the temp space was used but not the guard values at
        // either end
        BOOST_CHECK_EQUAL(temp_space_storage[0],
                          (Float)0.123456);
        BOOST_CHECK_EQUAL(temp_space_storage[temp_space_size + 1],
                          (Float)0.8765432);
        for (unsigned j = 0;  j < temp_space_size;  ++j)
            BOOST_CHECK(!std::isnan(temp_space[j]));

        double delta = (new_output - baseline_output).total();

        double this_gradient = delta / real_epsilon;
        double calc_gradient = input_errors[i];
        
        if (abs(this_gradient - calc_gradient)
            / std::max(abs(this_gradient), abs(calc_gradient))
            > tolerance / 100.0)
            cerr << "error is input error " << i << endl;

        BOOST_CHECK_CLOSE(this_gradient, calc_gradient, tolerance);
    }
}

double get_tolerance(float)
{
    return 0.1;
}

double get_tolerance(double)
{
    return 0.01;
}

double get_epsilon(float)
{
    return 1e-5;
}

double get_epsilon(double)
{
    return 1e-9;
}

template<class Float, class Layer>
void bprop_test(Layer & layer, Thread_Context & context, double tolerance = -1.0)
{
    int ni = layer.inputs(), no = layer.outputs();

    BOOST_REQUIRE(ni > 0);
    BOOST_REQUIRE(no > 0);

    distribution<Float> input(ni);
    for (unsigned i = 0;  i < ni;  ++i)
        input[i] = 0.5 - context.random01();

    if (layer.supports_missing_inputs()) {
        for (unsigned i = 0;  i < ni;  i += 2)
            input[i] = numeric_limits<float>::quiet_NaN();
    }

    if (tolerance == -1.0)
        tolerance = get_tolerance(Float());

    bprop_test<Float>(layer, input, get_epsilon(Float()), tolerance);
}

template<class Float, class Layer>
void bprop_test_backward(Layer & layerfwd, Thread_Context & context, double tolerance = -1.0)
{
    Reverse_Layer_Adaptor layer(make_unowned_sp(layerfwd));

    int ni = layer.inputs(), no = layer.outputs();

    BOOST_REQUIRE(ni > 0);
    BOOST_REQUIRE(no > 0);

    distribution<Float> input(ni);
    for (unsigned i = 0;  i < ni;  ++i)
        input[i] = 0.5 - context.random01();

    if (layer.supports_missing_inputs()) {
        for (unsigned i = 0;  i < ni;  i += 2)
            input[i] = numeric_limits<float>::quiet_NaN();
    }

    if (tolerance == -1.0)
        tolerance = get_tolerance(Float());

    bprop_test<Float>(layer, input, get_epsilon(Float()), tolerance);
}

template<class Float, class Layer>
void bprop_test_reconstruct(Layer & layerfwd, Thread_Context & context, double tolerance = -1.0)
{
    Reconstruct_Layer_Adaptor layer(make_unowned_sp(layerfwd));

    int ni = layer.inputs(), no = layer.outputs();

    BOOST_REQUIRE(ni > 0);
    BOOST_REQUIRE(no > 0);

    distribution<Float> input(ni);
    for (unsigned i = 0;  i < ni;  ++i)
        input[i] = 0.5 - context.random01();

    if (layer.supports_missing_inputs()) {
        for (unsigned i = 0;  i < ni;  i += 2)
            input[i] = numeric_limits<float>::quiet_NaN();
    }

    if (tolerance == -1.0)
        tolerance = get_tolerance(Float());

    bprop_test<Float>(layer, input, get_epsilon(Float()), tolerance);
}

template<typename Float, class Layer>
void bbprop_test(Layer & layer, Thread_Context & context,
                 double tolerance = -1.0)
{
    if (tolerance == -1.0)
        tolerance = get_tolerance(Float());
    
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

    double delta = 1;//1e-5;

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

        if (i == 0)
            cerr << "layer = " << layer << endl;

        // Perform a new fprop
        distribution<Float> new_output
            = layer.fprop(input, temp_space, temp_space_size);

        if (i == 0) {
            cerr << "old output = " << output << endl;
            cerr << "new_output = " << new_output << endl;
        }

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

        if (i == 0)
            cerr << "dgradient = "
                 << (gradient_eps.values - gradient0.values) / real_delta
                 << endl;

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

    // The values calculated by bprop and bbprop should be identical, or
    // there is a problem.  Gradient1 is calculated direcly by bprop.
    BOOST_CHECK_EQUAL_COLLECTIONS(gradient1.values.begin(),
                                  gradient1.values.end(),
                                  gradient2.values.begin(),
                                  gradient2.values.end());

    // So should the input errors
    BOOST_CHECK_EQUAL_COLLECTIONS(input_errors1.begin(),
                                  input_errors1.end(),
                                  input_errors2.begin(),
                                  input_errors2.end());

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

template<class Float, class Layer>
void bbprop_test_backward(Layer & layerfwd, Thread_Context & context, double tolerance = -1.0)
{
    Reverse_Layer_Adaptor layer(make_unowned_sp(layerfwd));

    if (tolerance == -1.0)
        tolerance = get_tolerance(Float());

    bbprop_test<Float>(layer, context, tolerance);
}

template<class Float, class Layer>
void bbprop_test_reconstruct(Layer & layerfwd, Thread_Context & context, double tolerance = -1.0)
{
    Reconstruct_Layer_Adaptor layer(make_unowned_sp(layerfwd));

    if (tolerance == -1.0)
        tolerance = get_tolerance(Float());

    bbprop_test<Float>(layer, context, tolerance);
}

#endif /* __jml__neural__testing__bprop_test_h__ */


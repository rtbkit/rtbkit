/* layer_stack_test.cc
   Jeremy Barnes, 9 November2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Unit tests for the layer stack class.
*/


#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#undef NDEBUG

#include <boost/test/unit_test.hpp>
#include <boost/multi_array.hpp>
#include "jml/neural/dense_layer.h"
#include "jml/neural/layer_stack.h"
#include "jml/utils/testing/serialize_reconstitute_include.h"
#include <boost/assign/list_of.hpp>
#include <limits>
#include "bprop_test.h"
#include "jml/arch/exception_handler.h"

using namespace ML;
using namespace ML::DB;
using namespace std;

using boost::unit_test::test_suite;


BOOST_AUTO_TEST_CASE( test_serialize_reconstitute_layer_stack )
{
    Thread_Context context;
    int ni = 2, no = 4;

    Dense_Layer<float> layer("test", ni, no, TF_TANH, MV_ZERO, context);
 
    Layer_Stack<Dense_Layer<float> > layers("test_layer");
    layers.add(make_unowned_sp(layer));

    BOOST_CHECK_EQUAL(layer.inputs(), layers.inputs());
    BOOST_CHECK_EQUAL(layer.outputs(), layers.outputs());

    // Test equality operator
    BOOST_CHECK_EQUAL(layers, layers);
    BOOST_CHECK(!layers.equal(layer));

    BOOST_CHECK_NO_THROW(layers.validate());

    // Check we can't add a null layer
    {
        JML_TRACE_EXCEPTIONS(false);
        BOOST_CHECK_THROW(layers.add(0), Exception);
    }

    BOOST_CHECK_NO_THROW(layers.validate());

    // Check conversion
    //These don't work as only Layer_Stack<Layer> works properly
    //test_serialize_reconstitute(layers);
    //test_poly_serialize_reconstitute<Layer>(layers);

    Layer_Stack<Layer> layers2 = layers;
    BOOST_CHECK_EQUAL(layers.size(), layers2.size());
    BOOST_CHECK_EQUAL(layers.name(), layers2.name());
    
    BOOST_CHECK_NO_THROW(layers2.validate());

    test_serialize_reconstitute(layers2);
    test_poly_serialize_reconstitute<Layer>(layers2);
    
    Layer_Stack<Layer> layers3;
    BOOST_CHECK_EQUAL(layers3.size(), 0);
    BOOST_CHECK_EQUAL(layers3.inputs(), 0);
    BOOST_CHECK_EQUAL(layers3.outputs(), 0);

    // Convert back
    Layer_Stack<Dense_Layer<float> > layers4;
    layers4 = layers2;

    BOOST_CHECK_EQUAL(layers4, layers);
}

BOOST_AUTO_TEST_CASE( test_one_dense_layer_stack )
{
    Dense_Layer<float> layer("test", 2, 1, TF_IDENTITY, MV_NONE);
    layer.weights[0][0] = 0.5;
    layer.weights[1][0] = 2.0;
    layer.bias[0] = 0.0;

    Dense_Layer<float> identity("test2", 1, 1, TF_IDENTITY, MV_NONE);
    identity.weights[0][0] = 1.0;
    identity.bias[0] = 0.0;

    Layer_Stack<Dense_Layer<float> > layers("test_layer");
    BOOST_CHECK_EQUAL(layers.max_width(), 0);
    BOOST_CHECK_EQUAL(layers.max_internal_width(), 0);
    layers.add(make_unowned_sp(layer));
    BOOST_CHECK_EQUAL(layers.max_width(), 2);
    BOOST_CHECK_EQUAL(layers.max_internal_width(), 0);

    Layer_Stack<Dense_Layer<float> > layersb(layers, Deep_Copy_Tag());
    BOOST_CHECK_EQUAL(layersb.max_width(), 2);
    BOOST_CHECK_EQUAL(layersb.max_internal_width(), 0);
    layersb.add(make_unowned_sp(identity));
    BOOST_CHECK_EQUAL(layersb.max_width(), 2);
    BOOST_CHECK_EQUAL(layersb.max_internal_width(), 1);

    Layer_Stack<Dense_Layer<float> > layersc;
    layersc = layersb;
    BOOST_CHECK_EQUAL(layersc.max_width(), 2);
    BOOST_CHECK_EQUAL(layersc.max_internal_width(), 1);
    BOOST_CHECK_EQUAL(layersb, layersc);

    distribution<float> input
        = boost::assign::list_of<float>(1.0)(-1.0);

    // Check that the basic functions work
    BOOST_REQUIRE_EQUAL(layers.apply(input).size(), 1);
    BOOST_CHECK_EQUAL(layers.apply(input)[0], -1.5);
    BOOST_REQUIRE_EQUAL(layersb.apply(input).size(), 1);
    BOOST_CHECK_EQUAL(layersb.apply(input)[0], -1.5);

    // Check the missing values throw an exception
    input[0] = numeric_limits<float>::quiet_NaN();
    {
        JML_TRACE_EXCEPTIONS(false);
        BOOST_CHECK_THROW(layers.apply(input), ML::Exception);
        BOOST_CHECK_THROW(layersb.apply(input), ML::Exception);
    }
        
    // Check that the wrong size throws an exception
    input.push_back(2.0);
    input[0] = 1.0;

    {
        JML_TRACE_EXCEPTIONS(false);
        BOOST_CHECK_THROW(layers.apply(input), ML::Exception);
        BOOST_CHECK_THROW(layersb.apply(input), ML::Exception);
    }

    input.pop_back();

    // Check that the bias works
    layer.bias[0] = 1.0;
    BOOST_CHECK_EQUAL(layers.apply(input)[0], -0.5);

    // Check that the parameter wasn't kept
    BOOST_CHECK_EQUAL(layersb.apply(input)[0], -1.5);

    layer.bias[0] = 0.0;

    // Check that there are parameters
    BOOST_CHECK_EQUAL(layers.parameters().parameter_count(), 3);

    // Check the info
    BOOST_CHECK_EQUAL(layers.inputs(), 2);
    BOOST_CHECK_EQUAL(layers.outputs(), 1);
    BOOST_CHECK_EQUAL(layers.name(), "test_layer");

    BOOST_CHECK_EQUAL(layersb.inputs(), 2);
    BOOST_CHECK_EQUAL(layersb.outputs(), 1);
    BOOST_CHECK_EQUAL(layersb.name(), "test_layer");

    // Check the copy constructor
    Layer_Stack<Dense_Layer<float> > layers2 = layers;
    BOOST_CHECK_EQUAL(layers2, layers);
    BOOST_CHECK_EQUAL(layers2.parameters().parameter_count(), 3);
    BOOST_CHECK_EQUAL(layersb.parameters().parameter_count(), 5);

    // Check the assignment operator
    Layer_Stack<Dense_Layer<float> > layers3;
    BOOST_CHECK(layers3 != layers);
    layers3 = layers;
    BOOST_CHECK_EQUAL(layers3, layers);
    BOOST_CHECK_EQUAL(layers3.parameters().parameter_count(), 3);

    // Make sure that the assignment kept a reference
    {
        Dense_Layer<float> & layer3
            = dynamic_cast<Dense_Layer<float> &>(layers3[0]);
        
        layer3.weights[0][0] = 5.0;
        BOOST_CHECK_EQUAL(layer.weights[0][0], 5.0);
        BOOST_CHECK_EQUAL(layer3.weights[0][0], 5.0);
        layer3.weights[0][0] = 0.5;
        BOOST_CHECK_EQUAL(layer, layer3);
    }

    // Make sure that the deep copy didn't keep a reference
    layers3 = Layer_Stack<Dense_Layer<float> >(layers, Deep_Copy_Tag());
    Dense_Layer<float> & layer3
        = dynamic_cast<Dense_Layer<float> &>(layers3[0]);
    layer3.weights[0][0] = 5.0;
    BOOST_CHECK_EQUAL(layer.weights[0][0], 0.5);
    BOOST_CHECK_EQUAL(layer3.weights[0][0], 5.0);
    layer3.weights[0][0] = 0.5;
    BOOST_CHECK_EQUAL(layer, layer3);

    // Check fprop (that it gives the same result as apply)
    distribution<float> applied = layers.apply(input);

    size_t temp_space_size = layers.fprop_temporary_space_required();

    float temp_space[temp_space_size + 2];
    temp_space[0] = 123456.789f;
    temp_space[temp_space_size + 1] = 9876.54321f;

    distribution<float> fproped(layer.outputs());
    layers.fprop(&input[0], temp_space + 1, temp_space_size, &fproped[0]);

    // Check for overwrite of temp space
    BOOST_CHECK_EQUAL(temp_space[0], 123456.789f);
    BOOST_CHECK_EQUAL(temp_space[temp_space_size + 1],  9876.54321f);
    
    BOOST_CHECK_EQUAL_COLLECTIONS(applied.begin(), applied.end(),
                                  fproped.begin(), fproped.end());

    distribution<float> appliedb = layersb.apply(input);
    BOOST_CHECK_EQUAL_COLLECTIONS(applied.begin(), applied.end(),
                                  appliedb.begin(), appliedb.end());
    

    size_t temp_space_sizeb = layersb.fprop_temporary_space_required();

    float temp_spaceb[temp_space_sizeb];

    distribution<float> fpropedb(layersb.outputs());
    layersb.fprop(&input[0], temp_spaceb, temp_space_sizeb, &fpropedb[0]);

    BOOST_CHECK_EQUAL_COLLECTIONS(appliedb.begin(), appliedb.end(),
                                  fpropedb.begin(), fpropedb.end());

    // Check parameters
    BOOST_CHECK_EQUAL(layers.parameter_count(), 3);

    Parameters_Copy<float> params(layers.parameters());
    distribution<float> & param_dist = params.values;

    BOOST_CHECK_EQUAL(param_dist.size(), 3);
    BOOST_CHECK_EQUAL(param_dist.at(0), 0.5);  // weight 0
    BOOST_CHECK_EQUAL(param_dist.at(1), 2.0);  // weight 1
    BOOST_CHECK_EQUAL(param_dist.at(2), 0.0);  // bias

    Thread_Context context;
    layers3.random_fill(-1.0, context);

    BOOST_CHECK(layers != layers3);

    layers3.parameters().set(params);
    
    BOOST_CHECK_EQUAL(layers, layers3);

    // Check backprop
    distribution<float> output_errors(1, 1.0);
    distribution<float> input_errors(input.size());
    Parameters_Copy<float> gradient(layers.parameters());
    gradient.fill(0.0);
    layers.bprop(&input[0], &fproped[0], temp_space + 1, temp_space_size,
                 &output_errors[0], &input_errors[0], gradient, 1.0);

    BOOST_CHECK_EQUAL(input_errors.size(), layers.inputs());
    
    // Check the values of input errors.  It's easy since there's only one
    // weight that contributes to each input (since there's only one output).
    BOOST_CHECK_EQUAL(input_errors[0], layer.weights[0][0]);
    BOOST_CHECK_EQUAL(input_errors[1], layer.weights[1][0]);

    //cerr << "input_errors = " << input_errors << endl;

    // Check that example_weight scales the gradient
    Parameters_Copy<float> gradient2(layers.parameters());
    gradient2.fill(0.0);
    layers.bprop(&input[0], &fproped[0], temp_space + 1, temp_space_size,
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
    layers3.parameters().update(layers3.parameters(), -1.0);

    Parameters_Copy<float> layers3_params(layers3);
    BOOST_CHECK_EQUAL(layers3_params.values.total(), 0.0);

    BOOST_CHECK_EQUAL(layer3.weights[0][0], 0.0);
    BOOST_CHECK_EQUAL(layer3.weights[1][0], 0.0);
    BOOST_CHECK_EQUAL(layer3.bias[0], 0.0);

    layers3.parameters().set(layers.parameters());
    BOOST_CHECK_EQUAL(layers, layers3);

    layers3.zero_fill();
    layers3.parameters().update(layers.parameters(), 1.0);
    BOOST_CHECK_EQUAL(layers, layers3);

    layers3.zero_fill();
    layers3.parameters().set(params);
    BOOST_CHECK_EQUAL(layers, layers3);

    Parameters_Copy<double> layer_params2(layers);
    BOOST_CHECK((params.values == layer_params2.values).all());

    // Check that on serialize and reconstitute, the params get properly
    // updated.
    {
        ostringstream stream_out;
        {
            DB::Store_Writer writer(stream_out);
            writer << layers;
        }
        
        istringstream stream_in(stream_out.str());
        DB::Store_Reader reader(stream_in);
        Layer_Stack<Dense_Layer<float> > layers4;

        BOOST_CHECK_NO_THROW(reader >> layers4);

        Parameters_Copy<float> params4(layers4.parameters());
        distribution<float> & param_dist4 = params4.values;

        BOOST_REQUIRE_EQUAL(param_dist4.size(), 3);
        BOOST_CHECK_EQUAL(param_dist4.at(0), 0.5);  // weight 0
        BOOST_CHECK_EQUAL(param_dist4.at(1), 2.0);  // weight 1
        BOOST_CHECK_EQUAL(param_dist4.at(2), 0.0);  // bias

        Dense_Layer<float> & layer4
            = dynamic_cast<Dense_Layer<float> &>(layers4[0]);

        layer4.weights[0][0] = 5.0;
        params4 = layers4.parameters();
        BOOST_CHECK_EQUAL(param_dist4[0], 5.0);

        BOOST_CHECK_EQUAL(layers.max_internal_width(), layers4.max_internal_width());
        BOOST_CHECK_EQUAL(layers.max_width(), layers4.max_width());
    }
}

BOOST_AUTO_TEST_CASE( test_bprop_one_layer )
{
    Thread_Context context;
    Dense_Layer<double> layer("test", 5, 10, TF_IDENTITY, MV_NONE, context);

    Layer_Stack<Dense_Layer<double> > layers("test_layers");
    layers.add(make_unowned_sp(layer));

    bprop_test<double>(layers, context);
}

BOOST_AUTO_TEST_CASE( test_bprop_two_layers )
{
    Thread_Context context;
    Dense_Layer<double> layer1("test1", 5, 10, TF_IDENTITY, MV_NONE, context);
    Dense_Layer<double> layer2("test2", 10, 20, TF_IDENTITY, MV_NONE, context);

    Layer_Stack<Dense_Layer<double> > layers("test_layers");
    layers.add(make_unowned_sp(layer1));
    layers.add(make_unowned_sp(layer2));

    bprop_test<double>(layers, context);
}

BOOST_AUTO_TEST_CASE( test_bprop_two_nonlinear_layers )
{
    Thread_Context context;
    Dense_Layer<double> layer1("test1", 5, 10, TF_TANH, MV_NONE, context);
    Dense_Layer<double> layer2("test2", 10, 20, TF_TANH, MV_NONE, context);

    Layer_Stack<Dense_Layer<double> > layers("test_layers");
    layers.add(make_unowned_sp(layer1));
    layers.add(make_unowned_sp(layer2));

    bprop_test<double>(layers, context, 0.05);
}

BOOST_AUTO_TEST_CASE( test_bprop_three_nonlinear_layers )
{
    Thread_Context context;
    Dense_Layer<double> layer1("test1", 5, 10, TF_TANH, MV_DENSE, context);
    Dense_Layer<double> layer2("test2", 10, 20, TF_TANH, MV_NONE, context);
    Dense_Layer<double> layer3("test3", 20, 5, TF_TANH, MV_NONE,  context);

    Layer_Stack<Dense_Layer<double> > layers("test_layers");
    layers.add(make_unowned_sp(layer1));
    layers.add(make_unowned_sp(layer2));
    layers.add(make_unowned_sp(layer3));

    bprop_test<double>(layers, context, 0.1);
}

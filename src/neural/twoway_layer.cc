/* twoway_layer.cc
   Jeremy Barnes, 4 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Two-way neural network layer.
*/

#include "twoway_layer.h"
#include "layer_stack_impl.h"
#include "utils/check_not_nan.h"
#include "boosting/registry.h"
#include "algebra/matrix_ops.h"
#include "stats/distribution_ops.h"


using namespace std;


namespace ML {


/*****************************************************************************/
/* TWOWAY_LAYER                                                              */
/*****************************************************************************/

Twoway_Layer::
Twoway_Layer()
{
}

Twoway_Layer::
Twoway_Layer(const std::string & name,
             size_t inputs, size_t outputs,
             Transfer_Function_Type transfer,
             Missing_Values missing_values,
             Thread_Context & context,
             float limit)
    : Auto_Encoder(name, inputs, outputs),
      forward(name, inputs, outputs, transfer, missing_values),
      ibias(inputs), iscales(inputs), oscales(outputs)
{
    if (limit == -1.0)
        limit = 1.0 / sqrt(inputs);
    random_fill(limit, context);
    update_parameters();
}

Twoway_Layer::
Twoway_Layer(const std::string & name,
             size_t inputs, size_t outputs,
             Transfer_Function_Type transfer,
             Missing_Values missing_values)
    : Auto_Encoder(name, inputs, outputs),
      forward(name, inputs, outputs, transfer, missing_values),
      ibias(inputs), iscales(inputs), oscales(outputs)
{
    update_parameters();
}

std::pair<float, float>
Twoway_Layer::
targets(float maximum) const
{
    return forward.targets(maximum);
}

void
Twoway_Layer::
apply(const float * input, float * output) const
{
    forward.apply(input, output);
}

void
Twoway_Layer::
apply(const double * input, double * output) const
{
    forward.apply(input, output);
}

size_t
Twoway_Layer::
fprop_temporary_space_required() const
{
    return forward.fprop_temporary_space_required();
}

void
Twoway_Layer::
fprop(const float * inputs,
      float * temp_space, size_t temp_space_size,
      float * outputs) const
{
    forward.fprop(inputs, temp_space, temp_space_size, outputs);
}

void
Twoway_Layer::
fprop(const double * inputs,
      double * temp_space, size_t temp_space_size,
      double * outputs) const
{
    forward.fprop(inputs, temp_space, temp_space_size, outputs);
}

void
Twoway_Layer::
bprop(const float * inputs,
      const float * outputs,
      const float * temp_space, size_t temp_space_size,
      const float * output_errors,
      float * input_errors,
      Parameters & gradient,
      double example_weight) const
{
    forward.bprop(inputs, outputs, temp_space, temp_space_size,
                  output_errors, input_errors, gradient, example_weight);
}

void
Twoway_Layer::
bprop(const double * inputs,
      const double * outputs,
      const double * temp_space, size_t temp_space_size,
      const double * output_errors,
      double * input_errors,
      Parameters & gradient,
      double example_weight) const
{
    forward.bprop(inputs, outputs, temp_space, temp_space_size,
                  output_errors, input_errors, gradient, example_weight);
}

template<typename F>
void
Twoway_Layer::
iapply(const F * outputs, F * inputs) const
{
    int no = this->outputs(), ni = this->inputs();

    F scaled_outputs[no];
    for (unsigned o = 0;  o < no;  ++o)
        scaled_outputs[o] = oscales[o] * outputs[o];

    F activations[ni];
    for (unsigned i = 0;  i < ni;  ++i)
        activations[i]
            = ibias[i]
            + iscales[i]
            * SIMD::vec_dotprod_dp(&forward.weights[i][0], scaled_outputs, no);

    forward.transfer_function->transfer(activations, inputs, ni);
}

void
Twoway_Layer::
iapply(const float * outputs, float * inputs) const
{
    iapply<float>(outputs, inputs);
}

void
Twoway_Layer::
iapply(const double * outputs, double * inputs) const
{
    iapply<double>(outputs, inputs);
}

size_t
Twoway_Layer::
ifprop_temporary_space_required() const
{
    return 0;
}

template<typename F>
void
Twoway_Layer::
ifprop(const F * outputs,
       F * temp_space, size_t temp_space_size,
       F * inputs) const
{
    if (temp_space_size != 0)
        throw Exception("temp_space_size is zero");

    iapply(outputs, inputs);
}

void
Twoway_Layer::
ifprop(const float * outputs,
       float * temp_space, size_t temp_space_size,
       float * inputs) const
{
    ifprop<float>(outputs, temp_space, temp_space_size, inputs);
}

void
Twoway_Layer::
ifprop(const double * outputs,
       double * temp_space, size_t temp_space_size,
       double * inputs) const
{
    ifprop<double>(outputs, temp_space, temp_space_size, inputs);
}

template<typename F>
void
Twoway_Layer::
ibprop(const F * outputs,
       const F * inputs,
       const F * temp_space, size_t temp_space_size,
       const F * input_errors,
       F * output_errors,
       Parameters & gradient,
       double example_weight) const
{
    int ni = this->inputs(), no = this->outputs();

    if (temp_space_size != 0)
        throw Exception("Dense_Layer::bprop(): wrong temp size");
    
    // Differentiate the output function
    F derivs[ni];
    forward.transfer_function->derivative(inputs, derivs, ni);

    // Bias updates are simply derivs in multiplied by transfer deriv
    F dbias[ni];
    SIMD::vec_prod(derivs, input_errors, dbias, ni);
    
    gradient.vector(4, "ibias").update(dbias, example_weight);

    F outputs_scaled[no];
    SIMD::vec_prod(outputs, &oscales[0], outputs_scaled, no);

    // Update weights
    for (unsigned i = 0;  i < ni;  ++i)
        gradient.matrix(0, "weights")
            .update_row(i, outputs_scaled,
                        dbias[i] * iscales[i] * example_weight);

    // Update iscales and oscales
    F iscales_updates[ni];
    double oscales_updates[no];
    for (unsigned o = 0;  o < no;  ++o)
        oscales_updates[o] = 0.0;

    for (unsigned i = 0;  i < ni;  ++i) {
        iscales_updates[i]
            = dbias[i]
            * SIMD::vec_dotprod_dp(&forward.weights[i][0],
                                   outputs_scaled, no);
        SIMD::vec_add(oscales_updates, dbias[i] * iscales[i],
                      &forward.weights[i][0], oscales_updates, no);
#if 0
        double total = 0.0;
        for (unsigned o = 0;  o < no;  ++o) {
            total += forward.weights[i][o] * outputs_scaled[o];
            oscales_updates[o] += forward.weights[i][o] * dbias[i] * iscales[i];
        }
        iscales_updates[i] = total * dbias[i];
#endif
    }

    gradient.vector(5, "iscales").update(iscales_updates, example_weight);
    gradient.vector(6, "oscales").update(oscales_updates, example_weight);
}

void
Twoway_Layer::
ibprop(const float * outputs,
       const float * inputs,
       const float * temp_space, size_t temp_space_size,
       const float * input_errors,
       float * output_errors,
       Parameters & gradient,
       double example_weight) const
{
    ibprop<float>(outputs, inputs, temp_space, temp_space_size,
                  input_errors, output_errors, gradient, example_weight);
}

void
Twoway_Layer::
ibprop(const double * outputs,
       const double * inputs,
       const double * temp_space, size_t temp_space_size,
       const double * input_errors,
       double * output_errors,
       Parameters & gradient,
       double example_weight) const
{
    ibprop<double>(outputs, inputs, temp_space, temp_space_size,
                  input_errors, output_errors, gradient, example_weight);
}

std::string
Twoway_Layer::
print() const
{
    string result = forward.print();

    size_t ni = inputs(), no = outputs();

    result += "  ibias: \n    [ ";
    for (unsigned j = 0;  j < ni;  ++j)
        result += format("%8.4f", ibias[j]);
    result += " ]\n";

    result += "  iscales: \n    [ ";
    for (unsigned j = 0;  j < ni;  ++j)
        result += format("%8.4f", iscales[j]);
    result += " ]\n";

    result += "  oscales: \n    [ ";
    for (unsigned j = 0;  j < no;  ++j)
        result += format("%8.4f", oscales[j]);
    result += " ]\n";

    return result;
}

std::string
Twoway_Layer::
class_id() const
{
    return "Twoway_Layer";
}

void
Twoway_Layer::
validate() const
{
    Base::validate();
    forward.validate();
    if (isnan(ibias).any())
        throw Exception("nan in ibias");
    if (isnan(iscales).any())
        throw Exception("nan in iscales");
    if (isnan(oscales).any())
        throw Exception("nan in oscales");
}

bool
Twoway_Layer::
equal_impl(const Layer & other) const
{
    const Twoway_Layer & cast
        = dynamic_cast<const Twoway_Layer &>(other);
    return operator == (cast);
}

void
Twoway_Layer::
add_parameters(Parameters & params)
{
    forward.add_parameters(params);
    params
        .add(4, "ibias", ibias)
        .add(5, "iscales", iscales)
        .add(6, "oscales", oscales);
}

size_t
Twoway_Layer::
parameter_count() const
{
    return forward.parameter_count() + 2 * inputs() + outputs();
}

void
Twoway_Layer::
serialize(DB::Store_Writer & store) const
{
    store << (char)1;  // version
    forward.serialize(store);
    store << ibias << iscales << oscales;
}

void
Twoway_Layer::
reconstitute(DB::Store_Reader & store)
{
    char version;
    store >> version;

    if (version != 1)
        throw Exception("Twoway_Layer::reconstitute(): invalid version");
    forward.reconstitute(store);
    store >> ibias >> iscales >> oscales;

    validate();
    update_parameters();
}

void
Twoway_Layer::
random_fill(float limit, Thread_Context & context)
{

    forward.random_fill(limit, context);
    for (unsigned i = 0;  i < ibias.size();  ++i) {
        ibias[i] = limit * (context.random01() * 2.0f - 1.0f);
        iscales[i] = context.random01();
    }
    for (unsigned i = 0;  i < outputs();  ++i)
        oscales[i] = context.random01();
    iscales.fill(1.0);
    oscales.fill(1.0);
}

void
Twoway_Layer::
zero_fill()
{
    forward.zero_fill();
    ibias.fill(0.0);
    iscales.fill(0.0);
    oscales.fill(0.0);
}

bool
Twoway_Layer::
operator == (const Twoway_Layer & other) const
{
    if (forward != other.forward) return false;
    return equivalent(ibias, other.ibias)
        && equivalent(iscales, other.iscales)
        && equivalent(oscales, other.oscales);
}

namespace {

Register_Factory<Layer, Twoway_Layer>
TWOWAY_REGISTER("Twoway_Layer");

} // file scope

template class Layer_Stack<Twoway_Layer>;


#if 0

// Float type to use for calculations
typedef double CFloat;

void
Twoway_Layer::
ibackprop_example(const distribution<double> & outputs,
                  const distribution<double> & output_deltas,
                  const distribution<double> & inputs,
                  distribution<double> & input_deltas,
                  Twoway_Layer_Updates & updates) const
{
#if 0
    distribution<double> dibias = iderivative(outputs) * output_deltas;

    updates.ibias += dibias;

    // The inputs can't be missing in this direction

    int ni = this->inputs(), no = this->outputs();

    for (unsigned i = 0;  i < ni;  ++i)
        SIMD::vec_add(&updates.weights[i][0], inputs[i], &dbias[0],
                      &updates.weights[i][0], no);

    input_deltas = weights * dbias;
#endif
}

#if 0
pair<double, double>
train_example2(const Twoway_Layer & layer,
               const vector<distribution<float> > & data,
               int example_num,
               float max_prob_cleared,
               Thread_Context & thread_context,
               Twoway_Layer_Updates & updates,
               Lock & update_lock,
               bool need_lock,
               int verbosity)
{
    int ni JML_UNUSED = layer.inputs();
    int no JML_UNUSED = layer.outputs();

    // Present this input
    distribution<CFloat> model_input(data.at(example_num));

    CHECK_NOT_NAN(model_input);
    
    if (model_input.size() != ni) {
        cerr << "model_input.size() = " << model_input.size() << endl;
        cerr << "ni = " << ni << endl;
        throw Exception("wrong sizes");
    }

    float prob_cleared = max_prob_cleared;

    distribution<CFloat> noisy_input;

    // Every second example we add with zero noise so that we don't end up
    // losing the ability to reconstruct the non-noisy input
    if (thread_context.random01() < 0.5)
        noisy_input = add_noise(model_input, thread_context, prob_cleared);
    else noisy_input = model_input;

    // Apply the layer
    distribution<CFloat> hidden_rep
        = layer.apply(noisy_input);

    CHECK_NOT_NAN(hidden_rep);
            
    // Reconstruct the input
    distribution<CFloat> denoised_input
        = layer.iapply(hidden_rep);

    CHECK_NOT_NAN(denoised_input);
            
    // Error signal
    distribution<CFloat> diff
        = model_input - denoised_input;
    
    // Overall error
    double error = pow(diff.two_norm(), 2);
    

    double error_exact = pow((model_input - layer.iapply(layer.apply(model_input))).two_norm(), 2);

    distribution<CFloat> denoised_input_deltas = -2 * diff;

    distribution<CFloat> hidden_rep_deltas;

    // Now we backprop in the two directions
    distribution<CFloat> hidden_rep_deltas;
    ibackprop_examples(denoised_input,
                       denoised_input_deltas,
                       hidden_rep,
                       hidden_rep_deltas,
                       updates);

    CHECK_NOT_NAN(hidden_rep_deltas);

    // And the other one
    distribution<CFloat> noisy_input_deltas;
    backprop_example(hidden_rep,
                     hidden_rep_deltas,
                     noisy_input,
                     noisy_input_deltas,
                     updates);

    return make_pair(error_exact, error);
}
#endif

#if 0

pair<double, double>
train_example(const Twoway_Layer & layer,
              const vector<distribution<float> > & data,
              int example_num,
              float max_prob_cleared,
              Thread_Context & thread_context,
              Twoway_Layer_Updates & updates,
              Lock & update_lock,
              bool need_lock,
              int verbosity)
{
    //cerr << "training example " << example_num << endl;

    int ni JML_UNUSED = layer.inputs();
    int no JML_UNUSED = layer.outputs();

    // Present this input
    distribution<CFloat> model_input(data.at(example_num));

    CHECK_NOT_NAN(model_input);
    
    if (model_input.size() != ni) {
        cerr << "model_input.size() = " << model_input.size() << endl;
        cerr << "ni = " << ni << endl;
        throw Exception("wrong sizes");
    }

    // Add noise up to the threshold
    // We don't add a uniform amount as this causes a bias in things like the
    // total.
    //float prob_cleared = thread_context.random01() * max_prob_cleared;
    //float prob_cleared = thread_context.random01() < 0.5 ? max_prob_cleared : 0.0;
    float prob_cleared = max_prob_cleared;

    distribution<CFloat> noisy_input;

    if (thread_context.random01() < 0.5)
        noisy_input = add_noise(model_input, thread_context, prob_cleared);
    else noisy_input = model_input;

    distribution<CFloat> noisy_pre
        = layer.preprocess(noisy_input);

    distribution<CFloat> hidden_act
        = layer.activation(noisy_pre);

    //cerr << "noisy_pre = " << noisy_pre << " hidden_act = "
    //     << hidden_act << endl;

    CHECK_NOT_NAN(hidden_act);
            
    // Apply the layer
    distribution<CFloat> hidden_rep
        = layer.transfer(hidden_act);

    CHECK_NOT_NAN(hidden_rep);
            
    // Reconstruct the input
    distribution<CFloat> denoised_input
        = layer.iapply(hidden_rep);

    CHECK_NOT_NAN(denoised_input);
            
    // Error signal
    distribution<CFloat> diff
        = model_input - denoised_input;
    
    // Overall error
    double error = pow(diff.two_norm(), 2);
    

    double error_exact = pow((model_input - layer.iapply(layer.apply(model_input))).two_norm(), 2);

    if (example_num < 10 && false) {
        cerr << " ex " << example_num << endl;
        cerr << "  input: " << distribution<float>(model_input.begin(),
                                                   model_input.begin() + 10)
             << endl;
        cerr << "  noisy input: " << distribution<float>(noisy_input.begin(),
                                                         noisy_input.begin() + 10)
             << endl;
        cerr << "  output: " << distribution<float>(denoised_input.begin(),
                                                    denoised_input.begin() + 10)
             << endl;
        cerr << "  diff: " << distribution<float>(diff.begin(),
                                                  diff.begin() + 10)
             << endl;
        cerr << "error: " << error << endl;
        cerr << endl;
        
    }
        
    // NOTE: OUT OF DATE, NEEDS CORRECTIONS
    // Now we solve for the gradient direction for the two biases as
    // well as for the weights matrix
    //
    // If f() is the activation function for the forward direction and
    // g() is the activation function for the reverse direction, we can
    // write
    //
    // h = f(Wi1 + b)
    //
    // where i1 is the (noisy) inputs, h is the hidden unit outputs, W
    // is the weight matrix and b is the forward bias vector.  Going
    // back again, we then take
    // 
    // i2 = g(W*h + c) = g(W*f(Wi + b) + c)
    //
    // where i2 is the denoised approximation of the true input weights
    // (i) and W* is W transposed.
    //
    // Using the MSE, we get
    //
    // e = sqr(||i2 - i||) = sum(sqr(i2 - i))
    //
    // where e is the MSE.
    //
    // Differentiating with respect to i2, we get
    //
    // de/di2 = 2(i2 - i)
    //
    // Finally, we want to know the gradient direction for each of the
    // parameters W, b and c.  Taking c first, we get
    //
    // de/dc = de/di2 di2/dc
    //       = 2 (i2 - i) g'(i2)
    //
    // As for b, we get
    //
    // de/db = de/di2 di2/db
    //       = 2 (i2 - i) g'(i2) W* f'(Wi + b)
    //
    // And for W:
    //
    // de/dW = de/di2 di2/dW
    //       = 2 (i2 - i) g'(i2) [ h + W* f'(Wi + b) i ]
    //
    // Since we want to minimise the reconstruction error, we use the
    // negative of the gradient.
    // END OUT OF DATE

    // NOTE: here, the activation function for the input and the output
    // are the same.
        
    const boost::multi_array<LFloat, 2> & W
        = layer.weights;

    const distribution<LFloat> & b JML_UNUSED = layer.bias;
    const distribution<LFloat> & c JML_UNUSED = layer.ibias;
    const distribution<LFloat> & d JML_UNUSED = layer.iscales;
    const distribution<LFloat> & e JML_UNUSED = layer.oscales;

    distribution<CFloat> c_updates
        = -2 * diff * layer.iderivative(denoised_input);

    if (!need_lock)
        updates.ibias += c_updates;

    CHECK_NOT_NAN(c_updates);

#if 0
    Twoway_Layer layer2 = layer;

    // Calculate numerically the c updates
    for (unsigned i = 0;  i < ni;  ++i) {

        float epsilon = 1e-8;
        double old = layer2.ibias[i];
        layer2.ibias[i] += epsilon;

        // Apply the layer
        distribution<CFloat> hidden_rep2
            = layer2.apply(noisy_input);
        
        distribution<CFloat> denoised_input2
            = layer2.iapply(hidden_rep2);

        // Error signal
        distribution<CFloat> diff2
            = model_input - denoised_input2;
            
        // Overall error
        double error2 = pow(diff2.two_norm(), 2);

        double delta = error2 - error;

        double deriv  = c_updates[i];
        double deriv2 = xdiv(delta, epsilon);

        cerr << format("%3d %7.4f %9.5f %9.5f %9.5f %8.5f\n",
                       i,
                       100.0 * xdiv(abs(deriv - deriv2),
                                    max(abs(deriv), abs(deriv2))),
                       abs(deriv - deriv2),
                       deriv, deriv2, noisy_input[i]);

        layer2.ibias[i] = old;
    }
#endif

    distribution<CFloat> hidden_rep_e
        = hidden_rep * e;

    distribution<CFloat> d_updates(ni);
    d_updates = multiply_r<CFloat>(W, hidden_rep_e) * c_updates;

    CHECK_NOT_NAN(d_updates);

    if (!need_lock)
        updates.iscales += d_updates;

#if 0
    Twoway_Layer layer2 = layer;

    // Calculate numerically the c updates
    for (unsigned i = 0;  i < ni;  ++i) {

        float epsilon = 1e-8;
        double old = layer2.iscales[i];
        layer2.iscales[i] += epsilon;

        // Apply the layer
        distribution<CFloat> hidden_rep2
            = layer2.apply(noisy_input);
        
        distribution<CFloat> denoised_input2
            = layer2.iapply(hidden_rep2);

        // Error signal
        distribution<CFloat> diff2
            = model_input - denoised_input2;
            
        // Overall error
        double error2 = pow(diff2.two_norm(), 2);

        double delta = error2 - error;

        double deriv  = d_updates[i];
        double deriv2 = xdiv(delta, epsilon);

        cerr << format("%3d %7.4f %9.5f %9.5f %9.5f %8.5f\n",
                       i,
                       100.0 * xdiv(abs(deriv - deriv2),
                                    max(abs(deriv), abs(deriv2))),
                       abs(deriv - deriv2),
                       deriv, deriv2, noisy_input[i]);

        layer2.iscales[i] = old;
    }
#endif

    distribution<CFloat> cupdates_d_W
        = multiply_r<CFloat>((c_updates * d), W);
    
    distribution<CFloat> e_updates = cupdates_d_W * hidden_rep;

    if (!need_lock)
        updates.oscales += e_updates;

    CHECK_NOT_NAN(e_updates);

#if 0
    Twoway_Layer layer2 = layer;

    // Calculate numerically the c updates
    for (unsigned i = 0;  i < no;  ++i) {

        float epsilon = 1e-8;
        double old = layer2.oscales[i];
        layer2.oscales[i] += epsilon;

        // Apply the layer
        distribution<CFloat> hidden_rep2
            = layer2.apply(noisy_input);
        
        distribution<CFloat> denoised_input2
            = layer2.iapply(hidden_rep2);

        // Error signal
        distribution<CFloat> diff2
            = model_input - denoised_input2;
            
        // Overall error
        double error2 = pow(diff2.two_norm(), 2);

        double delta = error2 - error;

        double deriv  = e_updates[i];
        double deriv2 = xdiv(delta, epsilon);

        cerr << format("%3d %7.4f %9.5f %9.5f %9.5f %8.5f\n",
                       i,
                       100.0 * xdiv(abs(deriv - deriv2),
                                    max(abs(deriv), abs(deriv2))),
                       abs(deriv - deriv2),
                       deriv, deriv2, noisy_input[i]);

        layer2.oscales[i] = old;
    }
#endif

    distribution<CFloat> hidden_deriv
        = layer.derivative(hidden_rep);

    // Check hidden_deriv numerically
#if 0
    distribution<CFloat> hidden_act2 = hidden_act;

    //cerr << "hidden_act = " << hidden_act << endl;
    //cerr << "hidden_deriv = " << hidden_deriv << endl;

    // Calculate numerically the c updates
    for (unsigned i = 0;  i < no;  ++i) {

        float epsilon = 1e-8;
        double old = hidden_act2[i];
        hidden_act2[i] += epsilon;

        // Apply the layer
        distribution<CFloat> hidden_rep2
            = layer.transfer(hidden_act2);
        
        // Error signal

        // Overall error
        double delta = hidden_rep2[i] - hidden_rep[i];

        double deriv  = hidden_deriv[i];
        double deriv2 = xdiv(delta, epsilon);

        cerr << format("%3d %7.4f %9.5f %9.5f %9.5f %8.5f\n",
                       i,
                       100.0 * xdiv(abs(deriv - deriv2),
                                    max(abs(deriv), abs(deriv2))),
                       abs(deriv - deriv2),
                       deriv, deriv2, hidden_act[i]);

        hidden_act2[i] = old;
    }
#endif
    

    CHECK_NOT_NAN(hidden_deriv);

    distribution<CFloat> b_updates = cupdates_d_W * hidden_deriv * e;

    CHECK_NOT_NAN(b_updates);

    if (!need_lock)
        updates.bias += b_updates;

#if 0
    Twoway_Layer layer2 = layer;

    // Calculate numerically the c updates
    for (unsigned i = 0;  i < no;  ++i) {

        float epsilon = 1e-8;
        double old = layer2.bias[i];
        layer2.bias[i] += epsilon;

        // Apply the layer
        distribution<CFloat> hidden_rep2
            = layer2.apply(noisy_input);
        
        distribution<CFloat> denoised_input2
            = layer2.iapply(hidden_rep2);

        // Error signal
        distribution<CFloat> diff2
            = model_input - denoised_input2;
            
        // Overall error
        double error2 = pow(diff2.two_norm(), 2);

        double delta = error2 - error;

        double deriv  = b_updates[i];
        double deriv2 = xdiv(delta, epsilon);

        cerr << format("%3d %7.4f %9.5f %9.5f %9.5f %8.5f\n",
                       i,
                       100.0 * xdiv(abs(deriv - deriv2),
                                    max(abs(deriv), abs(deriv2))),
                       abs(deriv - deriv2),
                       deriv, deriv2, noisy_input[i]);

        layer2.bias[i] = old;
    }
#endif

    distribution<double> factor_totals_accum(no);

    for (unsigned i = 0;  i < ni;  ++i)
        SIMD::vec_add(&factor_totals_accum[0], c_updates[i] * d[i], &W[i][0],
                      &factor_totals_accum[0], no);

    distribution<CFloat> factor_totals
        = factor_totals_accum.cast<CFloat>() * e;

    boost::multi_array<CFloat, 2> W_updates;
    vector<distribution<double> > missing_act_updates;

    distribution<double> hidden_rep_ed(hidden_rep_e);

    if (need_lock) {
        W_updates.resize(boost::extents[ni][no]);
        missing_act_updates.resize(ni, distribution<double>(no));
    }

    for (unsigned i = 0;  i < ni;  ++i) {

        if (!layer.use_dense_missing
            || !isnan(noisy_input[i])) {
            
            CFloat W_updates_row[no];
            
            // We use the W value for both the input and the output, so we
            // need to accumulate it's total effect on the derivative
            calc_W_updates(c_updates[i] * d[i],
                           &hidden_rep_e[0],
                           
                           model_input[i],
                           &factor_totals[0],
                           &hidden_deriv[0],
                           (need_lock ? &W_updates[i][0] : W_updates_row),
                           no);
            
            if (!need_lock) {
                CHECK_NOT_NAN_RANGE(&W_updates_row[0], &W_updates_row[0] + no);
                SIMD::vec_add(&updates.weights[i][0], W_updates_row,
                              &updates.weights[i][0], no);
            }
        }
        else {
            // The weight updates are simpler, but we also have to calculate
            // the missing activation updates

            // W value only used on the way out; simpler calculation

            if (need_lock)
                SIMD::vec_add(&W_updates[i][0], c_updates[i] * d[i],
                              &hidden_rep_e[0], &W_updates[i][0], no);
            else
                SIMD::vec_add(&updates.weights[i][0], c_updates[i] * d[i],
                              &hidden_rep_e[0], &updates.weights[i][0], no);

            distribution<CFloat> mau_updates
                = factor_totals * hidden_deriv;

            if (need_lock) {
                // Missing values were used on the way in
                missing_act_updates[i] = mau_updates;
            }
            else {
                SIMD::vec_add(&updates.missing_activations[i][0],
                              &mau_updates[0],
                              &updates.missing_activations[i][0],
                              no);
            }
        }
    }
       
    
#if 0  // test numerically
    Twoway_Layer layer2 = layer;

    for (unsigned i = 0;  i < ni;  ++i) {

        for (unsigned j = 0;  j < no;  ++j) {
            double epsilon = 1e-8;

            double old_W = layer2.weights[i][j];
            layer2.weights[i][j] += epsilon;

            // Apply the layer
            distribution<CFloat> hidden_rep2
                = layer2.apply(noisy_input);

            distribution<CFloat> denoised_input2
                = layer2.iapply(hidden_rep2);
            
            // Error signal
            distribution<CFloat> diff2
                = model_input - denoised_input2;
                    
            //cerr << "diff = " << diff << endl;
            //cerr << "diff2 = " << diff2 << endl;
                    
            // Overall error
            double error2 = pow(diff2.two_norm(), 2);
                    
            double delta = error2 - error;

            double deriv2 = xdiv(delta, epsilon);

            double deriv = W_updates[i][j];

            cerr << format("%3d %3d %7.4f %9.5f %9.5f %9.5f %8.5f\n",
                           i, j,
                           100.0 * xdiv(abs(deriv - deriv2),
                                        max(abs(deriv), abs(deriv2))),
                           abs(deriv - deriv2),
                           deriv, deriv2, noisy_input[i]);

            //cerr << "error = " << error << " error2 = " << error2
            //     << " delta = " << delta
            //    << " deriv " << W_updates[i][j]
            //     << " deriv2 " << deriv2 << endl;


            layer2.weights[i][j] = old_W;
        }
    }
#endif  // if one/zero

#if 0  // test numerically the missing activations
    Twoway_Layer layer2 = layer;

    for (unsigned i = 0;  i < ni;  ++i) {
        if (!isnan(noisy_input[i])) continue;  // will be zero

        for (unsigned j = 0;  j < no;  ++j) {
            double epsilon = 1e-8;

            double old_W = layer2.missing_activations[i][j];
            layer2.missing_activations[i][j] += epsilon;

            // Apply the layer
            distribution<CFloat> hidden_rep2
                = layer2.apply(noisy_input);

            distribution<CFloat> denoised_input2
                = layer2.iapply(hidden_rep2);
            
            // Error signal
            distribution<CFloat> diff2
                = model_input - denoised_input2;
                    
            //cerr << "diff = " << diff << endl;
            //cerr << "diff2 = " << diff2 << endl;
                    
            // Overall error
            double error2 = pow(diff2.two_norm(), 2);
                    
            double delta = error2 - error;

            double deriv2 = xdiv(delta, epsilon);

            double deriv = missing_act_updates[i][j];

            cerr << format("%3d %3d %7.4f %9.5f %9.5f %9.5f %8.5f\n",
                           i, j,
                           100.0 * xdiv(abs(deriv - deriv2),
                                        max(abs(deriv), abs(deriv2))),
                           abs(deriv - deriv2),
                           deriv, deriv2, noisy_input[i]);

            layer2.missing_activations[i][j] = old_W;
        }
    }
#endif  // if one/zero

    distribution<double> cleared_value_updates(ni);
    
    if (!layer.use_dense_missing) {
        cleared_value_updates = isnan(noisy_input) * W * b_updates;

        if (!need_lock)
            updates.missing_replacements += cleared_value_updates;
    }

#if 0  // test numerically
    for (unsigned i = 0;  i < ni;  ++i) {
        double epsilon = 1e-6;

        distribution<CFloat> noisy_pre2 = noisy_pre;
        noisy_pre2[i] += epsilon;

        // Apply the layer
        distribution<CFloat> hidden_act2
            = layer.activation(noisy_pre2);

        distribution<CFloat> hidden_rep2
            = layer.transfer(hidden_act2);
                    
        distribution<CFloat> denoised_input2
            = layer.iapply(hidden_rep2);
                    
        // Error signal
        distribution<CFloat> diff2
            = model_input - denoised_input2;
                    
        // Overall error
        double error2 = pow(diff2.two_norm(), 2);
                    
        double delta = error2 - error;

        double deriv2 = xdiv(delta, epsilon);

        cerr << "error = " << error << " error2 = " << error2
             << " delta = " << delta
             << " deriv " << cleared_value_updates[i]
             << " deriv2 " << deriv2 << endl;
    }
#endif // test numerically

#if 0
    cerr << "cleared_value_updates.size() = " << cleared_value_updates.size()
         << endl;
    cerr << "ni = " << ni << endl;
    cerr << "b_updates.size() = " << b_updates.size() << endl;
#endif

    if (need_lock && true) {  // faster, despite the lock
        Guard guard(update_lock);
        
        for (unsigned i = 0;  i < ni;  ++i)
            if (isnan(noisy_input[i]))
                updates.missing_replacements[i] += cleared_value_updates[i];

#if 0
        cerr << "b_updates = " << b_updates << endl;
        cerr << "c_updates = " << c_updates << endl;
        cerr << "d_updates = " << d_updates << endl;
        cerr << "e_updates = " << e_updates << endl;
        cerr << "W_updates = " << W_updates << endl;
#endif
        
        updates.bias += b_updates;
        updates.ibias += c_updates;
        updates.iscales += d_updates;
        updates.oscales += e_updates;
        
        for (unsigned i = 0;  i < ni;  ++i) {
            SIMD::vec_add(&updates.weights[i][0],
                          &W_updates[i][0],
                          &updates.weights[i][0], no);
            
            if (layer.use_dense_missing && isnan(noisy_input[i]))
                updates.missing_activations[i] += missing_act_updates[i];
        }
    }
    else if (need_lock) {
        for (unsigned i = 0;  i < ni;  ++i)
            if (isnan(noisy_input[i]))
                atomic_accumulate(updates.missing_replacements[i],
                                  cleared_value_updates[i]);

        atomic_accumulate(&updates.bias[0], &b_updates[0], no);
        atomic_accumulate(&updates.ibias[0], &c_updates[0], ni);
        atomic_accumulate(&updates.iscales[0], &d_updates[0], ni);
        atomic_accumulate(&updates.oscales[0], &e_updates[0], no);
        
        for (unsigned i = 0;  i < ni;  ++i) {
            atomic_accumulate(&updates.weights[i][0], &W_updates[i][0], no);

            if (layer.use_dense_missing && isnan(noisy_input[i]))
                atomic_accumulate(&updates.missing_activations[i][0],
                                  &missing_act_updates[i][0],
                                  no);
        }
    }

    return make_pair(error_exact, error);
}

#endif

#endif

} // namespace ML

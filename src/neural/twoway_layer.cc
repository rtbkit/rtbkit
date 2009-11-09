/* twoway_layer.cc
   Jeremy Barnes, 4 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Two-way neural network layer.
*/

#include "twoway_layer.h"
#include "layer_stack_impl.h"

namespace ML {


/*****************************************************************************/
/* TWOWAY_LAYER_UPDATES                                                      */
/*****************************************************************************/

Twoway_Layer_Updates::
Twoway_Layer_Updates()
{
    init(false, false, 0, 0);
}

Twoway_Layer_Updates::
Twoway_Layer_Updates(bool train_generative,
                     const Twoway_Layer & layer)
{
    init(train_generative, layer);
}

Twoway_Layer_Updates::
Twoway_Layer_Updates(bool use_dense_missing,
                     bool train_generative,
                     int inputs, int outputs)
{
    init(use_dense_missing, train_generative, inputs, outputs);
}

void
Twoway_Layer_Updates::
zero_fill()
{
    std::fill(weights.data(), weights.data() + weights.num_elements(),
              0.0);
    bias.fill(0.0);
    missing_replacements.fill(0.0);
    if (use_dense_missing)
        for (unsigned i = 0;  i < missing_activations.size();  ++i)
            missing_activations[i].fill(0.0);
    ibias.fill(0.0);
    iscales.fill(0.0);
    hscales.fill(0.0);
}

void
Twoway_Layer_Updates::
init(bool train_generative,
     const Twoway_Layer & layer)
{
    init(layer.use_dense_missing, train_generative,
         layer.inputs(), layer.outputs());
}

void
Twoway_Layer_Updates::
init(bool use_dense_missing, bool train_generative,
     int inputs, int outputs)
{
    this->use_dense_missing = use_dense_missing;
    this->train_generative = train_generative;
    
    weights.resize(boost::extents[inputs][outputs]);
    bias.resize(outputs);
    missing_replacements.resize(inputs);
    
    if (use_dense_missing)
        missing_activations.resize(inputs, distribution<double>(outputs));
    
    ibias.resize(inputs);
    iscales.resize(inputs);
    hscales.resize(outputs);
    
    zero_fill();
}

Twoway_Layer_Updates &
Twoway_Layer_Updates::
operator += (const Twoway_Layer_Updates & other)
{
    int ni = inputs();
    int no = outputs();
    
    //cerr << "ni = " << ni << " no = " << no << endl;
    
    if (ni != other.inputs() || no != other.outputs()
        || use_dense_missing != other.use_dense_missing)
        throw Exception("incompatible update objects");
    
    ibias += other.ibias;
    bias += other.bias;
    iscales += other.iscales;
    hscales += other.hscales;
    
    CHECK_NO_NAN(ibias);
    CHECK_NO_NAN(other.ibias);
    CHECK_NO_NAN(bias);
    CHECK_NO_NAN(other.bias);
    CHECK_NO_NAN(iscales);
    CHECK_NO_NAN(other.iscales);
    CHECK_NO_NAN(hscales);
    CHECK_NO_NAN(other.hscales);
    
    for (unsigned i = 0;  i < ni;  ++i) {
        SIMD::vec_add(&weights[i][0],
                      &other.weights[i][0],
                      &weights[i][0], no);
        CHECK_NO_NAN_RANGE(&other.weights[i][0], &other.weights[i][0] + no);
        CHECK_NO_NAN_RANGE(&weights[i][0], &weights[i][0] + no);
    }
    
    if (use_dense_missing) {
        if (missing_activations.size() != ni
            || other.missing_activations.size() != ni) {
            throw Exception("wrong missing activation size");
        }
        
        for (unsigned i = 0;  i < ni;  ++i) {
            if (missing_activations[i].size() != no
                || other.missing_activations[i].size() != no)
                throw Exception("wrong inner missing activation size");
            
            const distribution<double> & me JML_UNUSED
                = missing_activations[i];
            CHECK_NO_NAN(me);
            const distribution<double> & them JML_UNUSED
                = missing_activations[i];
            CHECK_NO_NAN(them);
            
            SIMD::vec_add(&missing_activations[i][0],
                          &other.missing_activations[i][0],
                          &missing_activations[i][0], no);
        }
    }
    else missing_replacements += other.missing_replacements;
    
    return *this;
}


/*****************************************************************************/
/* TWOWAY_LAYER                                                              */
/*****************************************************************************/

Twoway_Layer::
Twoway_Layer()
{
}

Twoway_Layer::
Twoway_Layer(bool use_dense_missing,
             size_t inputs, size_t outputs,
             Transfer_Function_Type transfer,
             Thread_Context & context,
             float limit)
    : Base(use_dense_missing, inputs, outputs, transfer), ibias(inputs),
      iscales(inputs), hscales(outputs)
{
    if (limit == -1.0)
        limit = 1.0 / sqrt(inputs);
    random_fill(limit, context);
}

Twoway_Layer::
Twoway_Layer(bool use_dense_missing,
             size_t inputs, size_t outputs,
             Transfer_Function_Type transfer)
    : Base(use_dense_missing, inputs, outputs, transfer), ibias(inputs),
      iscales(inputs), hscales(outputs)
{
}

distribution<double>
Twoway_Layer::
iapply(const distribution<double> & output) const
{
    CHECK_NO_NAN(output);
    distribution<double> activation
        = multiply_r<double>(weights, (hscales * output)) * iscales;
    activation += ibias;
    transfer(&activation[0], &activation[0], inputs(), transfer_function);
    return activation;
}

distribution<float>
Twoway_Layer::
iapply(const distribution<float> & output) const
{
    CHECK_NO_NAN(output);
    distribution<float> activation
        = multiply_r<float>(weights, (hscales * output)) * iscales;
    activation += ibias;
    transfer(&activation[0], &activation[0], inputs(), transfer_function);
    return activation;
}

distribution<double>
Twoway_Layer::
ipreprocess(const distribution<double> & input) const
{
    return input;
}

distribution<float>
Twoway_Layer::
ipreprocess(const distribution<float> & input) const
{
    return input;
}

distribution<double>
Twoway_Layer::
iactivation(const distribution<double> & output) const
{
    CHECK_NO_NAN(output);
    distribution<double> activation
        = multiply_r<double>(weights, (hscales * output)) * iscales;
    activation += ibias;
    return activation;
}

distribution<float>
Twoway_Layer::
iactivation(const distribution<float> & output) const
{
    CHECK_NO_NAN(output);
    distribution<float> activation
        = multiply_r<float>(weights, (hscales * output)) * iscales;
    activation += ibias;
    return activation;
}

distribution<double>
Twoway_Layer::
itransfer(const distribution<double> & activation) const
{
    CHECK_NO_NAN(activation);
    int ni = inputs();
    if (activation.size() != ni)
        throw Exception("invalid sizes in itransfer");
    distribution<double> result(ni);
    transfer(&activation[0], &result[0], ni, transfer_function);
    return activation;
}

distribution<float>
Twoway_Layer::
itransfer(const distribution<float> & activation) const
{
    CHECK_NO_NAN(activation);
    int ni = inputs();
    if (activation.size() != ni)
        throw Exception("invalid sizes in itransfer");
    distribution<float> result(ni);
    transfer(&activation[0], &result[0], ni, transfer_function);
    return activation;
}

distribution<double>
Twoway_Layer::
iderivative(const distribution<double> & input) const
{
    CHECK_NO_NAN(input);
    if (input.size() != this->inputs())
        throw Exception("iderivative(): wrong size");
    int ni = this->inputs();
    distribution<double> result(ni);
    derivative(&input[0], &result[0], ni, transfer_function);
    return result;
}

distribution<float>
Twoway_Layer::
iderivative(const distribution<float> & input) const
{
    CHECK_NO_NAN(input);
    if (input.size() != this->inputs())
        throw Exception("iderivative(): wrong size");
    int ni = this->inputs();
    distribution<float> result(ni);
    derivative(&input[0], &result[0], ni, transfer_function);
    return result;
}

void
Twoway_Layer::
update(const Twoway_Layer_Updates & updates, double learning_rate)
{
    //cerr << "------------ BEFORE -------------" << endl;
    //cerr << print();

    int ni = inputs();
    int no = outputs();
    
    //cerr << "updates.ibias = " << updates.ibias << endl;

    if (use_dense_missing != updates.use_dense_missing)
        throw Exception("use_dense_missing mismatch");

    ibias -= learning_rate * updates.ibias;
    bias -= learning_rate * updates.bias;
    iscales -= learning_rate * updates.iscales;
    hscales -= learning_rate * updates.hscales;

    for (unsigned i = 0;  i < ni;  ++i)
        SIMD::vec_add(&weights[i][0], -learning_rate,
                      &updates.weights[i][0],
                      &weights[i][0], no);

    if (use_dense_missing) {
        for (unsigned i = 0;  i < ni;  ++i)
            SIMD::vec_add(&missing_activations[i][0],
                          -learning_rate,
                          &updates.missing_activations[i][0],
                          &missing_activations[i][0], no);
    }
    else 
        missing_replacements
            -= learning_rate * updates.missing_replacements;

    //cerr << "------------ AFTER -------------" << endl;
    //cerr << print();
}

void
Twoway_Layer::
random_fill(float limit, Thread_Context & context)
{
    Base::random_fill(limit, context);
    for (unsigned i = 0;  i < ibias.size();  ++i) {
        ibias[i] = limit * (context.random01() * 2.0f - 1.0f);
        iscales[i] = context.random01();
    }
    for (unsigned i = 0;  i < outputs();  ++i)
        hscales[i] = context.random01();
    iscales.fill(1.0);
    hscales.fill(1.0);
}

void
Twoway_Layer::
zero_fill()
{
    Base::zero_fill();
    ibias.fill(0.0);
    iscales.fill(0.0);
    hscales.fill(0.0);
}

void
Twoway_Layer::
serialize(DB::Store_Writer & store) const
{
    Base::serialize(store);
    store << ibias << iscales << hscales;
}

void
Twoway_Layer::
reconstitute(DB::Store_Reader & store)
{
    Base::reconstitute(store);
    store >> ibias >> iscales >> hscales;
}

std::string
Twoway_Layer::
print() const
{
    string result = Base::print();

    size_t ni = inputs(), no = outputs();

    result += "  ibias: \n    [ ";
    for (unsigned j = 0;  j < ni;  ++j)
        result += format("%8.4f", ibias[j]);
    result += " ]\n";

    result += "  iscales: \n    [ ";
    for (unsigned j = 0;  j < ni;  ++j)
        result += format("%8.4f", iscales[j]);
    result += " ]\n";

    result += "  hscales: \n    [ ";
    for (unsigned j = 0;  j < no;  ++j)
        result += format("%8.4f", hscales[j]);
    result += " ]\n";

    return result;
}

bool
Twoway_Layer::
operator == (const Twoway_Layer & other) const
{
    if (!Base::operator == (other)) return false;
    return equivalent(ibias, other.ibias)
        && equivalent(iscales, other.iscales)
        && equivalent(hscales, other.hscales);
}


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

void
Twoway_Layer::
backprop_example(const distribution<double> & outputs,
                 const distribution<double> & output_deltas,
                 const distribution<double> & inputs,
                 distribution<double> & input_deltas,
                 Twoway_Layer_Updates & updates) const
{
}

template<typename Float>
distribution<Float>
add_noise(const distribution<Float> & inputs,
          Thread_Context & context,
          float prob_cleared)
{
    distribution<Float> result = inputs;

    for (unsigned i = 0;  i < inputs.size();  ++i)
        if (context.random01() < prob_cleared)
            result[i] = NaN;
    
    return result;
}

template class Layer_Stack<Twoway_Layer>;

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

    CHECK_NO_NAN(model_input);
    
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

    CHECK_NO_NAN(hidden_rep);
            
    // Reconstruct the input
    distribution<CFloat> denoised_input
        = layer.iapply(hidden_rep);

    CHECK_NO_NAN(denoised_input);
            
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

    CHECK_NO_NAN(hidden_rep_deltas);

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

    CHECK_NO_NAN(model_input);
    
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

    CHECK_NO_NAN(hidden_act);
            
    // Apply the layer
    distribution<CFloat> hidden_rep
        = layer.transfer(hidden_act);

    CHECK_NO_NAN(hidden_rep);
            
    // Reconstruct the input
    distribution<CFloat> denoised_input
        = layer.iapply(hidden_rep);

    CHECK_NO_NAN(denoised_input);
            
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
    const distribution<LFloat> & e JML_UNUSED = layer.hscales;

    distribution<CFloat> c_updates
        = -2 * diff * layer.iderivative(denoised_input);

    if (!need_lock)
        updates.ibias += c_updates;

    CHECK_NO_NAN(c_updates);

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

    CHECK_NO_NAN(d_updates);

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
        updates.hscales += e_updates;

    CHECK_NO_NAN(e_updates);

#if 0
    Twoway_Layer layer2 = layer;

    // Calculate numerically the c updates
    for (unsigned i = 0;  i < no;  ++i) {

        float epsilon = 1e-8;
        double old = layer2.hscales[i];
        layer2.hscales[i] += epsilon;

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

        layer2.hscales[i] = old;
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
    

    CHECK_NO_NAN(hidden_deriv);

    distribution<CFloat> b_updates = cupdates_d_W * hidden_deriv * e;

    CHECK_NO_NAN(b_updates);

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
                CHECK_NO_NAN_RANGE(&W_updates_row[0], &W_updates_row[0] + no);
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
        updates.hscales += e_updates;
        
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
        atomic_accumulate(&updates.hscales[0], &e_updates[0], no);
        
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

struct Train_Examples_Job {

    const Twoway_Layer & layer;
    const vector<distribution<float> > & data;
    int first;
    int last;
    const vector<int> & examples;
    float prob_cleared;
    const Thread_Context & context;
    int random_seed;
    Twoway_Layer_Updates & updates;
    Lock & update_lock;
    double & error_exact;
    double & error_noisy;
    boost::progress_display * progress;
    int verbosity;

    Train_Examples_Job(const Twoway_Layer & layer,
                       const vector<distribution<float> > & data,
                       int first, int last,
                       const vector<int> & examples,
                       float prob_cleared,
                       const Thread_Context & context,
                       int random_seed,
                       Twoway_Layer_Updates & updates,
                       Lock & update_lock,
                       double & error_exact,
                       double & error_noisy,
                       boost::progress_display * progress,
                       int verbosity)
        : layer(layer), data(data), first(first), last(last),
          examples(examples), prob_cleared(prob_cleared),
          context(context), random_seed(random_seed), updates(updates),
          update_lock(update_lock),
          error_exact(error_exact), error_noisy(error_noisy),
          progress(progress), verbosity(verbosity)
    {
    }

    void operator () ()
    {
        Thread_Context thread_context(context);
        thread_context.seed(random_seed);
        
        double total_error_exact = 0.0, total_error_noisy = 0.0;

        Twoway_Layer_Updates local_updates(true /* train_generative */, layer);

        for (unsigned x = first;  x < last;  ++x) {

            double eex, eno;
            boost::tie(eex, eno)
                = train_example(layer, data, x,
                                prob_cleared, thread_context,
                                local_updates, update_lock,
                                false /* need_lock */,
                                verbosity);

            total_error_exact += eex;
            total_error_noisy += eno;
        }

        Guard guard(update_lock);

        //cerr << "applying local updates" << endl;
        updates += local_updates;

        error_exact += total_error_exact;
        error_noisy += total_error_noisy;
        

        if (progress && verbosity >= 3)
            (*progress) += (last - first);
    }
};

std::pair<double, double>
Twoway_Layer::
train_iter(const vector<distribution<float> > & data,
           float prob_cleared,
           Thread_Context & thread_context,
           int minibatch_size, float learning_rate,
           int verbosity,
           float sample_proportion,
           bool randomize_order)
{
    Worker_Task & worker = thread_context.worker();

    int nx = data.size();
    int ni JML_UNUSED = inputs();
    int no JML_UNUSED = outputs();

    int microbatch_size = minibatch_size / (num_cpus() * 4);
            
    Lock update_lock;

    double total_mse_exact = 0.0, total_mse_noisy = 0.0;
    
    vector<int> examples;
    for (unsigned x = 0;  x < nx;  ++x) {
        // Randomly exclude some samples
        if (thread_context.random01() >= sample_proportion)
            continue;
        examples.push_back(x);
    }
    
    if (randomize_order) {
        Thread_Context::RNG_Type rng = thread_context.rng();
        std::random_shuffle(examples.begin(), examples.end(), rng);
    }
    
    int nx2 = examples.size();

    std::auto_ptr<boost::progress_display> progress;
    if (verbosity >= 3) progress.reset(new boost::progress_display(nx2, cerr));

    for (unsigned x = 0;  x < nx2;  x += minibatch_size) {
                
        Twoway_Layer_Updates updates(true /* train_generative */, *this);
                
        // Now, submit it as jobs to the worker task to be done
        // multithreaded
        int group;
        {
            int parent = -1;  // no parent group
            group = worker.get_group(NO_JOB, "dump user results task",
                                     parent);
                    
            // Make sure the group gets unlocked once we've populated
            // everything
            Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                         boost::ref(worker),
                                         group));
                    
                    
            for (unsigned x2 = x;  x2 < nx2 && x2 < x + minibatch_size;
                 x2 += microbatch_size) {
                        
                Train_Examples_Job job(*this,
                                       data,
                                       x2,
                                       min<int>(nx2,
                                                min(x + minibatch_size,
                                                    x2 + microbatch_size)),
                                       examples,
                                       prob_cleared,
                                       thread_context,
                                       thread_context.random(),
                                       updates,
                                       update_lock,
                                       total_mse_exact,
                                       total_mse_noisy,
                                       progress.get(),
                                       verbosity);
                // Send it to a thread to be processed
                worker.add(job, "blend job", group);
            }
        }
                
        worker.run_until_finished(group);

        //cerr << "applying minibatch updates" << endl;
        
        update(updates, learning_rate);
    }

    return make_pair(sqrt(total_mse_exact / nx2), sqrt(total_mse_noisy / nx2));
}

struct Test_Examples_Job {

    const Twoway_Layer & layer;
    const vector<distribution<float> > & data_in;
    vector<distribution<float> > & data_out;
    int first;
    int last;
    float prob_cleared;
    const Thread_Context & context;
    int random_seed;
    Lock & update_lock;
    double & error_exact;
    double & error_noisy;
    boost::progress_display * progress;
    int verbosity;

    Test_Examples_Job(const Twoway_Layer & layer,
                      const vector<distribution<float> > & data_in,
                      vector<distribution<float> > & data_out,
                      int first, int last,
                      float prob_cleared,
                      const Thread_Context & context,
                      int random_seed,
                      Lock & update_lock,
                      double & error_exact,
                      double & error_noisy,
                      boost::progress_display * progress,
                      int verbosity)
        : layer(layer), data_in(data_in), data_out(data_out),
          first(first), last(last),
          prob_cleared(prob_cleared),
          context(context), random_seed(random_seed),
          update_lock(update_lock),
          error_exact(error_exact), error_noisy(error_noisy),
          progress(progress), verbosity(verbosity)
    {
    }

    void operator () ()
    {
        Thread_Context thread_context(context);
        thread_context.seed(random_seed);

        double test_error_exact = 0.0, test_error_noisy = 0.0;

        for (unsigned x = first;  x < last;  ++x) {
            int ni JML_UNUSED = layer.inputs();
            int no JML_UNUSED = layer.outputs();

            // Present this input
            distribution<CFloat> model_input(data_in[x]);
            
            distribution<bool> was_cleared;

            // Add noise
            distribution<CFloat> noisy_input
                = add_noise(model_input, thread_context, prob_cleared);
            
            // Apply the layer
            distribution<CFloat> hidden_rep
                = layer.apply(noisy_input);
            
            // Reconstruct the input
            distribution<CFloat> denoised_input
                = layer.iapply(hidden_rep);
            
            // Error signal
            distribution<CFloat> diff
                = model_input - denoised_input;
    
            // Overall error
            double error = pow(diff.two_norm(), 2);

            test_error_noisy += error;


            // Apply the layer
            distribution<CFloat> hidden_rep2
                = layer.apply(model_input);

            if (!data_out.empty())
                data_out.at(x) = hidden_rep2.cast<float>();
            
            // Reconstruct the input
            distribution<CFloat> reconstructed_input
                = layer.iapply(hidden_rep2);
            
            // Error signal
            distribution<CFloat> diff2
                = model_input - reconstructed_input;
    
            // Overall error
            double error2 = pow(diff2.two_norm(), 2);
    
            test_error_exact += error2;

            if (x < 5 && false) {
                Guard guard(update_lock);
                cerr << "ex " << x << " error " << error2 << endl;
                cerr << "    input " << model_input << endl;
                //cerr << "    act   " << layer.activation(model_input) << endl;
                cerr << "    rep   " << hidden_rep2 << endl;
                //cerr << "    act2  " << layer.iactivation(hidden_rep2) << endl;
                cerr << "    ibias " << layer.ibias << endl;
                cerr << "    out   " << reconstructed_input << endl;
                cerr << "    diff  " << diff2 << endl;
                cerr << endl;
            }
        }

        Guard guard(update_lock);
        error_exact += test_error_exact;
        error_noisy += test_error_noisy;
        if (progress && verbosity >= 3)
            (*progress) += (last - first);
    }
};

pair<double, double>
Twoway_Layer::
test_and_update(const vector<distribution<float> > & data_in,
                vector<distribution<float> > & data_out,
                float prob_cleared,
                Thread_Context & thread_context,
                int verbosity) const
{
    Lock update_lock;
    double error_exact = 0.0;
    double error_noisy = 0.0;

    int nx = data_in.size();

    std::auto_ptr<boost::progress_display> progress;
    if (verbosity >= 3) progress.reset(new boost::progress_display(nx, cerr));

    Worker_Task & worker = thread_context.worker();
            
    // Now, submit it as jobs to the worker task to be done
    // multithreaded
    int group;
    {
        int parent = -1;  // no parent group
        group = worker.get_group(NO_JOB, "dump user results task",
                                 parent);
        
        // Make sure the group gets unlocked once we've populated
        // everything
        Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                     boost::ref(worker),
                                     group));
        
        // 20 jobs per CPU
        int batch_size = nx / (num_cpus() * 20);
        
        for (unsigned x = 0; x < nx;  x += batch_size) {
            
            Test_Examples_Job job(*this, data_in, data_out,
                                  x, min<int>(x + batch_size, nx),
                                  prob_cleared,
                                  thread_context,
                                  thread_context.random(),
                                  update_lock,
                                  error_exact, error_noisy,
                                  progress.get(),
                                  verbosity);
            
            // Send it to a thread to be processed
            worker.add(job, "blend job", group);
        }
    }

    worker.run_until_finished(group);

    return make_pair(sqrt(error_exact / nx),
                     sqrt(error_noisy / nx));
}


} // namespace ML

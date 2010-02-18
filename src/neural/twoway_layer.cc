/* twoway_layer.cc
   Jeremy Barnes, 4 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Two-way neural network layer.
*/

#undef NDEBUG

#include "twoway_layer.h"
#include "layer_stack_impl.h"
#include "jml/utils/check_not_nan.h"
#include "jml/boosting/registry.h"
#include "jml/algebra/matrix_ops.h"
#include "jml/stats/distribution_ops.h"

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

Twoway_Layer::
Twoway_Layer(const Twoway_Layer & other)
    : Auto_Encoder(other), forward(other.forward), ibias(other.ibias),
      iscales(other.iscales), oscales(other.oscales)
{
    update_parameters();
}

Twoway_Layer &
Twoway_Layer::
operator = (const Twoway_Layer & other)
{
    Twoway_Layer new_me(other);
    swap(new_me);
    return *this;
}

void
Twoway_Layer::
swap(Twoway_Layer & other)
{
    Auto_Encoder::swap(other);
    forward.swap(other.forward);
    ibias.swap(other.ibias);
    iscales.swap(other.iscales);
    oscales.swap(other.oscales);
}

std::pair<float, float>
Twoway_Layer::
targets(float maximum) const
{
    return forward.targets(maximum);
}

bool
Twoway_Layer::
supports_missing_inputs() const
{
    return forward.supports_missing_inputs();
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

void
Twoway_Layer::
bbprop(const float * inputs,
       const float * outputs,
       const float * temp_space, size_t temp_space_size,
       const float * output_errors,
       const float * d2output_errors,
       float * input_errors,
       float * d2input_errors,
       Parameters & gradient,
       Parameters * dgradient,
       double example_weight) const
{
    forward.bbprop(inputs, outputs, temp_space, temp_space_size,
                   output_errors, d2output_errors, input_errors,
                   d2input_errors, gradient, dgradient,
                   example_weight);
}
 
void
Twoway_Layer::
bbprop(const double * inputs,
       const double * outputs,
       const double * temp_space, size_t temp_space_size,
       const double * output_errors,
       const double * d2output_errors,
       double * input_errors,
       double * d2input_errors,
       Parameters & gradient,
       Parameters * dgradient,
       double example_weight) const
{
    return forward.bbprop(inputs, outputs, temp_space, temp_space_size,
                          output_errors, d2output_errors, input_errors,
                          d2input_errors, gradient, dgradient,
                          example_weight);
}

std::pair<float, float>
Twoway_Layer::
itargets(float maximum) const
{
    return forward.transfer_function->targets(maximum);
}

bool
Twoway_Layer::
supports_missing_outputs() const
{
    return false;
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

    CHECK_NOT_NAN_N(outputs, no);
    CHECK_NOT_NAN_N(inputs, ni);
    CHECK_NOT_NAN_N(input_errors, ni);
    
    // Differentiate the output function
    F derivs[ni];
    forward.transfer_function->derivative(inputs, derivs, ni);

    CHECK_NOT_NAN_N((F *)derivs, ni);

    // Bias updates are simply derivs in multiplied by transfer deriv
    F dbias[ni];
    SIMD::vec_prod(derivs, input_errors, dbias, ni);

    CHECK_NOT_NAN_N((F *)dbias, ni);
    
    gradient.vector(4, "ibias").update(dbias, example_weight);

    F outputs_scaled[no];
    SIMD::vec_prod(outputs, &oscales[0], outputs_scaled, no);

    CHECK_NOT_NAN_N((F *)outputs_scaled, no);

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

    if (output_errors)
        SIMD::vec_prod(&oscales[0], &oscales_updates[0], output_errors, no);

    for (unsigned o = 0;  o < no;  ++o)
        oscales_updates[o] *= outputs[o];

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

void
Twoway_Layer::
ibbprop(const float * outputs,
        const float * inputs,
        const float * temp_space, size_t temp_space_size,
        const float * input_errors,
        const float * d2input_errors,
        float * output_errors,
        float * d2output_errors,
        Parameters & gradient,
        Parameters * dgradient,
        double example_weight) const
{
    ibbprop<float>(outputs, inputs, temp_space, temp_space_size,
                   input_errors, d2input_errors, output_errors,
                   d2output_errors, gradient, dgradient, example_weight);
}
 
void
Twoway_Layer::
ibbprop(const double * outputs,
        const double * inputs,
        const double * temp_space, size_t temp_space_size,
        const double * input_errors,
        const double * d2input_errors,
        double * output_errors,
        double * d2output_errors,
        Parameters & gradient,
        Parameters * dgradient,
        double example_weight) const
{
    ibbprop<double>(outputs, inputs, temp_space, temp_space_size,
                    input_errors, d2input_errors, output_errors,
                    d2output_errors, gradient, dgradient, example_weight);
}

#if 0 // for reference; to be deleted
template<typename Float>
template<typename F>
void
Dense_Layer<Float>::
bbprop(const F * inputs,
       const F * outputs,
       const F * temp_space, size_t temp_space_size,
       const F * output_errors,
       const F * d2output_errors,
       F * input_errors,
       F * d2input_errors,
       Parameters & gradient,
       Parameters * dgradient,
       double example_weight) const
{
    int ni = this->inputs(), no = this->outputs();

    if (temp_space_size != 0)
        throw Exception("Dense_Layer::bprop(): wrong temp size");
    
    // Differentiate the output function
    F derivs[no];
    transfer_function->derivative(outputs, derivs, no);

    F ddbias[no];
    if (dgradient || d2input_errors) {
        // Approximation to the second derivative of the output errors
        for (unsigned o = 0;  o < no;  ++o)
            ddbias[o] = d2output_errors[o] * sqr(derivs[o]);

#if 1 // improve the approximation using the second derivative
        F ddtransfer[no];
        // Second derivative of the output errors
        transfer_function->second_derivative(outputs, ddtransfer, no);
        for (unsigned o = 0;  o < no;  ++o)
            ddbias[o] += outputs[o] * ddtransfer[o] * d2output_errors[o];
#endif // improve the approximation

        // These are the bias errors...
        dgradient->vector(1, "bias").update(ddbias, example_weight);
    }

    // Bias updates are simply derivs in multiplied by transfer deriv
    F dbias[no];
    SIMD::vec_prod(derivs, &output_errors[0], dbias, no);
    gradient.vector(1, "bias").update(dbias, example_weight);

    for (unsigned i = 0;  i < ni;  ++i) {
        bool was_missing = isnan(inputs[i]);
        if (input_errors) input_errors[i] = 0.0;

        if (!was_missing) {
            if (inputs[i] == 0.0) continue;

            gradient.matrix(0, "weights")
                .update_row(i, dbias, inputs[i] * example_weight);

            if (input_errors)
                input_errors[i]
                    = SIMD::vec_dotprod_dp(&weights[i][0],
                                           &dbias[0], no);
            
            if (dgradient)
                dgradient->matrix(0,"weights")
                    .update_row(i, ddbias,
                                inputs[i] * inputs[i] * example_weight);

            if (d2input_errors)
                d2input_errors[i]
                    = SIMD::vec_accum_prod3(&weights[i][0],
                                            &weights[i][0],
                                            ddbias,
                                            no);
        }
        else if (missing_values == MV_NONE)
            throw Exception("MV_NONE but missing value");
        else if (missing_values == MV_ZERO) {
            // No update as everything is multiplied by zero
        }
        else if (missing_values == MV_DENSE) {
            gradient.matrix(3, "missing_activations")
                .update_row(i, dbias, example_weight);
            if (dgradient)
                dgradient->matrix(3, "missing_activations")
                    .update_row(i, ddbias, example_weight);
        }
        else if (missing_values == MV_INPUT) {
            // Missing

            // Update the weights
            gradient.matrix(0, "weights")
                .update_row(i, dbias,
                            missing_replacements[i] * example_weight);
            
            gradient.vector(2, "missing_replacements")
                .update_element(i,
                                (example_weight
                                 * SIMD::vec_dotprod_dp(&weights[i][0], dbias,
                                                        no)));

            if (dgradient) {
                dgradient->matrix(0, "weights")
                    .update_row(i, ddbias,
                                sqr(missing_replacements[i]) * example_weight);

                dgradient->vector(2, "missing_replacements")
                    .update_element(i,
                                    (example_weight
                                     * SIMD::vec_accum_prod3(&weights[i][0],
                                                             &weights[i][0],
                                                             ddbias,
                                                             no)));
            }
        }
    }
}
#endif // for reference

namespace {

template<typename F>
F sqr(F val)
{
    return val * val;
}

} // file scope

template<typename F>
void
Twoway_Layer::
ibbprop(const F * outputs,
        const F * inputs,
        const F * temp_space, size_t temp_space_size,
        const F * input_errors,
        const F * d2input_errors,
        F * output_errors,
        F * d2output_errors,
        Parameters & gradient,
        Parameters * dgradient,
        double example_weight) const
{
    int ni = this->inputs(), no = this->outputs();

    if (temp_space_size != 0)
        throw Exception("Dense_Layer::bprop(): wrong temp size");

    CHECK_NOT_NAN_N(outputs, no);
    CHECK_NOT_NAN_N(inputs, ni);
    CHECK_NOT_NAN_N(input_errors, ni);
    
    // Differentiate the output function
    F derivs[ni];
    forward.transfer_function->derivative(inputs, derivs, ni);

    CHECK_NOT_NAN_N((F *)derivs, ni);

    // Bias updates are simply derivs in multiplied by transfer deriv
    F dbias[ni];
    SIMD::vec_prod(derivs, input_errors, dbias, ni);

    CHECK_NOT_NAN_N((F *)dbias, ni);
    
    gradient.vector(4, "ibias").update(dbias, example_weight);

    F ddbias[ni];
    if (dgradient || d2output_errors) {
        // Approximation to the second derivative of the input errors
        for (unsigned i = 0;  i < ni;  ++i)
            ddbias[i] = d2input_errors[i] * sqr(derivs[i]);

#if 0 // improve the approximation using the second derivative
        F ddtransfer[ni];
        // Second derivative of the output errors
        forward.transfer_function->second_derivative(inputs, ddtransfer, ni);
        for (unsigned i = 0;  i < ni;  ++i)
            ddbias[i] += inputs[i] * ddtransfer[i] * d2input_errors[i];
#endif // improve the approximation

        // These are the bias errors...
        dgradient->vector(4, "ibias").update(ddbias, example_weight);
    }

    F outputs_scaled[no];
    SIMD::vec_prod(outputs, &oscales[0], outputs_scaled, no);

    CHECK_NOT_NAN_N((F *)outputs_scaled, no);

    F iscales_updates[ni];
    for (unsigned i = 0;  i < ni;  ++i) {
        iscales_updates[i]
            = dbias[i]
            * SIMD::vec_dotprod_dp(&forward.weights[i][0],
                                   outputs_scaled, no);
    }

    F iscales_dupdates[ni];
    for (unsigned i = 0;  i < ni;  ++i)
        iscales_dupdates[i] = sqr(iscales_updates[i] / dbias[i]) * ddbias[i];
    
    if (dgradient)
        dgradient->vector(5, "iscales")
            .update(iscales_dupdates, example_weight);

    // Update weights
    for (unsigned i = 0;  i < ni;  ++i) {
        gradient.matrix(0, "weights")
            .update_row(i, outputs_scaled,
                        dbias[i] * iscales[i] * example_weight);

        if (dgradient)
            dgradient->matrix(0, "weights")
                .update_row_sqr(i, outputs_scaled,
                                iscales[i] * iscales[i] * ddbias[i]);
    }

    // Update iscales and oscales
    double oscales_updates[no];
    for (unsigned o = 0;  o < no;  ++o)
        oscales_updates[o] = 0.0;

    for (unsigned i = 0;  i < ni;  ++i) {
        SIMD::vec_add(oscales_updates, dbias[i] * iscales[i],
                      &forward.weights[i][0], oscales_updates, no);
    }

    double oscales_dupdates[no];
    for (unsigned o = 0;  o < no;  ++o)
        oscales_dupdates[o] = 0.0;
    for (unsigned i = 0;  i < ni;  ++i) {
        SIMD::vec_add_sqr(oscales_dupdates, ddbias[i] * iscales[i] * iscales[i],
                          &forward.weights[i][0], oscales_dupdates, no);
    }

    if (d2output_errors) {
        for (unsigned o = 0;  o < no;  ++o)
            d2output_errors[o] = oscales_dupdates[o] * oscales[o] * oscales[o];
    }

    if (dgradient) {
        for (unsigned o = 0;  o < no;  ++o)
            oscales_dupdates[o] *= outputs[o] * outputs[o];
        dgradient->vector(6, "oscales")
            .update(oscales_dupdates, example_weight);
    }

    if (output_errors)
        SIMD::vec_prod(&oscales[0], &oscales_updates[0], output_errors, no);

    for (unsigned o = 0;  o < no;  ++o)
        oscales_updates[o] *= outputs[o];

    gradient.vector(5, "iscales").update(iscales_updates, example_weight);
    gradient.vector(6, "oscales").update(oscales_updates, example_weight);
}

namespace {

void calc_W_updates(double k1, const double * x, double k2, const double * y,
                    const double * z, double * r, size_t n)
{
    return SIMD::vec_k1_x_plus_k2_y_z(k1, x, k2, y, z, r, n);
}
 
void calc_W_updates(float k1, const float * x, float k2, const float * y,
                    const float * z, float * r, size_t n)
{
    return SIMD::vec_k1_x_plus_k2_y_z(k1, x, k2, y, z, r, n);
}

} // file scope

template<typename F>
void
Twoway_Layer::
rbprop(const F * inputs,
       const F * reconstruction,
       const F * temp_space,
       size_t temp_space_size,
       const F * reconstruction_errors,
       F * input_errors_out,
       Parameters & gradient,
       double example_weight) const
{
    // Temporary space:
    // 
    // +-----------+-------------+---------------+
    // |  fprop    | outputs     |   ifprop      |
    // +-----------+-------------+---------------+
    // |<- fspace->|<-   no    ->|<-  ifspace  ->|

    size_t fspace = fprop_temporary_space_required();
    size_t ifspace = ifprop_temporary_space_required();

    if (temp_space_size != this->outputs() + fspace + ifspace)
        throw Exception("wrong temporary space size");

    const F * outputs = temp_space + fspace;

    typedef Twoway_Layer::Float LFloat;

    int ni JML_UNUSED = this->inputs();
    int no JML_UNUSED = this->outputs();

    distribution<F> noisy_input(inputs, inputs + ni);

    // Apply the layer
    distribution<F> hidden_rep(outputs, outputs + no);

    CHECK_NOT_NAN(hidden_rep);
            
    // Reconstruct the input
    distribution<F> denoised_input(reconstruction, reconstruction + ni);

    CHECK_NOT_NAN(denoised_input);
            
    // Error signal
    distribution<F> diff(reconstruction_errors, reconstruction_errors + ni);
    
    const boost::multi_array<LFloat, 2> & W = forward.weights;

    const distribution<LFloat> & b JML_UNUSED = forward.bias;
    const distribution<LFloat> & c JML_UNUSED = ibias;
    const distribution<LFloat> & d JML_UNUSED = iscales;
    const distribution<LFloat> & e JML_UNUSED = oscales;

    distribution<F> c_updates
        = diff * forward.transfer_function->derivative(denoised_input);

    CHECK_NOT_NAN(c_updates);

    gradient.vector(4, "ibias").update(c_updates, example_weight);

    distribution<F> hidden_rep_e
        = hidden_rep * e;

    distribution<F> d_updates(ni);
    d_updates = multiply_r<F>(W, hidden_rep_e) * c_updates;

    CHECK_NOT_NAN(d_updates);

    gradient.vector(5, "iscales").update(d_updates, example_weight);

    distribution<F> cupdates_d_W
        = multiply_r<F>((c_updates * d), W);
    
    distribution<F> e_updates = cupdates_d_W * hidden_rep;

    gradient.vector(6, "oscales").update(e_updates, example_weight);


    CHECK_NOT_NAN(e_updates);

    distribution<F> hidden_deriv
        = forward.transfer_function->derivative(hidden_rep);

    CHECK_NOT_NAN(hidden_deriv);

    distribution<F> b_updates = cupdates_d_W * hidden_deriv * e;

    CHECK_NOT_NAN(b_updates);

    gradient.vector(1, "bias").update(b_updates, example_weight);

    distribution<double> factor_totals_accum(no);

    for (unsigned i = 0;  i < ni;  ++i)
        SIMD::vec_add(&factor_totals_accum[0], c_updates[i] * d[i], &W[i][0],
                      &factor_totals_accum[0], no);

    distribution<F> factor_totals
        = factor_totals_accum.cast<F>() * e;

    boost::multi_array<F, 2> W_updates;
    vector<distribution<double> > missing_act_updates;

    distribution<double> hidden_rep_ed(hidden_rep_e);

    for (unsigned i = 0;  i < ni;  ++i) {

        if (!isnan(noisy_input[i])) {
            
            F W_updates_row[no];
            
            // We use the W value for both the input and the output, so we
            // need to accumulate it's total effect on the derivative
            calc_W_updates(c_updates[i] * d[i],
                           &hidden_rep_e[0],
                           
                           noisy_input[i],
                           &factor_totals[0],
                           &hidden_deriv[0],
                           W_updates_row,
                           no);
            
            CHECK_NOT_NAN_RANGE(&W_updates_row[0], &W_updates_row[0] + no);

            gradient.matrix(0, "weights")
                .update_row(i, W_updates_row, example_weight);
        }
        else if (forward.missing_values == MV_NONE)
            throw Exception("MV_NONE but missing value");
        else if (forward.missing_values == MV_ZERO) {
            gradient.matrix(0, "weights")
                .update_row(i, hidden_rep_e,
                            c_updates[i] * d[i] * example_weight);
        }
        else if (forward.missing_values == MV_DENSE) {
            // The weight updates are simpler, but we also have to calculate
            // the missing activation updates
            
            // W value only used on the way out; simpler calculation
            
            gradient.matrix(0, "weights")
                .update_row(i, hidden_rep_e,
                            c_updates[i] * d[i] * example_weight);

            distribution<F> mau_updates
                = factor_totals * hidden_deriv;

            gradient.matrix(3, "missing_activations")
                .update_row(i, mau_updates, example_weight);
        }
        else if (forward.missing_values == MV_INPUT) {

            F W_updates_row[no];
            
            // We use the W value for both the input and the output, so we
            // need to accumulate it's total effect on the derivative
            calc_W_updates(c_updates[i] * d[i],
                           &hidden_rep_e[0],
                           
                           forward.missing_replacements[i],
                           &factor_totals[0],
                           &hidden_deriv[0],
                           W_updates_row,
                           no);
            
            CHECK_NOT_NAN_RANGE(&W_updates_row[0], &W_updates_row[0] + no);

            gradient.matrix(0, "weights")
                .update_row(i, W_updates_row, example_weight);
        }
        else throw Exception("unknown updates");
    }

    distribution<F> cleared_value_updates(ni);
    cleared_value_updates = isnan(noisy_input) * (W * b_updates);
        
    if (forward.missing_values == MV_INPUT) {
        gradient.vector(2, "missing_replacements")
            .update(cleared_value_updates, example_weight);
    }


    if (input_errors_out) {
        distribution<F> input_updates(W * b_updates);
        std::copy(input_updates.begin(), input_updates.end(),
                  input_errors_out);
    }
}
    
void
Twoway_Layer::
rbprop(const float * inputs,
       const float * reconstruction,
       const float * temp_space,
       size_t temp_space_size,
       const float * reconstruction_errors,
       float * input_errors_out,
       Parameters & gradient,
       double example_weight) const
{
    return rbprop<float>(inputs, reconstruction, temp_space, temp_space_size,
                         reconstruction_errors, input_errors_out,
                         gradient, example_weight);
}
    
void
Twoway_Layer::
rbprop(const double * inputs,
       const double * reconstruction,
       const double * temp_space,
       size_t temp_space_size,
       const double * reconstruction_errors,
       double * input_errors_out,
       Parameters & gradient,
       double example_weight) const
{
    return rbprop<double>(inputs, reconstruction, temp_space, temp_space_size,
                          reconstruction_errors, input_errors_out,
                          gradient, example_weight);
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
    store << name_;
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

    store >> name_;
    forward.reconstitute(store);
    Layer::init(name_, forward.inputs(), forward.outputs());
    store >> ibias >> iscales >> oscales;

    validate();
    update_parameters();
}

void
Twoway_Layer::
random_fill(float limit, Thread_Context & context)
{
    cerr << "Twoway_Layer::random_fill(): limit = " << limit
         << " ni = " << inputs() << " no = " << outputs()
         << endl;

    forward.random_fill(limit, context);
    for (unsigned i = 0;  i < ibias.size();  ++i)
        ibias[i] = limit * (context.random01() * 2.0f - 1.0f);

    for (unsigned i = 0;  i < outputs();  ++i)
        oscales[i] = 0.5 + context.random01();

    for (unsigned i = 0;  i < inputs();  ++i)
        iscales[i] = 0.5 + context.random01();

    iscales.fill(1.0);
    oscales.fill(1.0);

    if (forward.missing_values == MV_DENSE) {
        for (unsigned i = 0;  i < inputs();  ++i)
            for (unsigned o = 0;  o < outputs();  ++o)
                forward.missing_activations[i][o] = limit * limit * (context.random01() * 2.0f - 1.0f);
    }
}

void
Twoway_Layer::
zero_fill()
{
    forward.zero_fill();
    ibias.fill(0.0);
    iscales.fill(1.0);
    oscales.fill(1.0);
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

} // namespace ML

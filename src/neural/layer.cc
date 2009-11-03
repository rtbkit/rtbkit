/* layer.cc
   Jeremy Barnes, 20 October 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

*/

#include "layer.h"
#include "db/persistent.h"
#include "arch/demangle.h"
#include "algebra/matrix_ops.h"
#include "arch/simd_vector.h"


using namespace std;
using namespace ML::DB;

namespace ML {


/*****************************************************************************/
/* LAYER                                                                     */
/*****************************************************************************/

/** A basic layer of a neural network.  Other kinds of layers can be built on
    this base.
*/

Layer::
Layer()
{
}

distribution<float>
Layer::
apply(const distribution<float> & input) const
{
    if (input.size() != inputs())
        throw Exception("Layer::apply(): invalid number of inputs");
    distribution<float> result(outputs());
    apply(&input[0], &result[0]);
    return result;
}

distribution<double>
Layer::
apply(const distribution<double> & input) const
{
    if (input.size() != inputs())
        throw Exception("Layer::apply(): invalid number of inputs");
    distribution<double> result(outputs());
    apply(&input[0], &result[0]);
    return result;
}
        
void
Layer::
apply(const distribution<float> & input,
      distribution<float> & output) const
{
    if (input.size() != inputs())
        throw Exception("Layer::apply(): invalid number of inputs");
    output.resize(outputs());
    apply(&input[0], &output[0]);
}

void
Layer::
apply(const distribution<double> & input,
      distribution<double> & output) const
{
    if (input.size() != inputs())
        throw Exception("Layer::apply(): invalid number of inputs");
    output.resize(outputs());
    apply(&input[0], &output[0]);
}

void
Layer::
apply(const float * input,
      float * output) const
{
    int ni = inputs(), no = outputs();
    float pre[ni];
    preprocess(input, pre);
    float act[no];
    activation(pre, act);
    transfer(act, output);
}

void
Layer::
apply(const double * input,
      double * output) const
{
    int ni = inputs(), no = outputs();
    double pre[ni];
    preprocess(input, pre);
    double act[no];
    activation(pre, act);
    transfer(act, output);
}

void
Layer::
preprocess(const float * input,
           float * preprocessed) const
{
    if (input == preprocessed) return;
    int ni = inputs();
    std::copy(input, input + ni, preprocessed);
}

void
Layer::
preprocess(const double * input,
           double * preprocessed) const
{
    if (input == preprocessed) return;
    int ni = inputs();
    std::copy(input, input + ni, preprocessed);
}

distribution<float>
Layer::
preprocess(const distribution<float> & input) const
{
    int ni = inputs();
    distribution<float> pre(ni);
    preprocess(&input[0], &pre[0]);
    return pre;
}

distribution<double>
Layer::
preprocess(const distribution<double> & input) const
{
    int ni = inputs();
    distribution<double> pre(ni);
    preprocess(&input[0], &pre[0]);
    return pre;
}

void
Layer::
activation(const float * preprocessed,
           float * act) const
{
    int ni = inputs(), no = outputs();
    double preprocessedd[ni], actd[no];
    std::copy(preprocessed, preprocessed + ni, preprocessedd);
    activation(preprocessedd, actd);
    std::copy(actd, actd + no, act);
}

distribution<float>
Layer::
activation(const distribution<float> & preprocessed) const
{
    int no = outputs();
    distribution<float> output(no);
    activation(&preprocessed[0], &output[0]);
    return output;
}

distribution<double>
Layer::
activation(const distribution<double> & preprocessed) const
{
    int no = outputs();
    distribution<double> output(no);
    activation(&preprocessed[0], &output[0]);
    return output;
}

template<typename Float>
void
Layer::
transfer(const Float * activation, Float * outputs, int nvals,
         Transfer_Function_Type transfer_function)
{
    switch (transfer_function) {
    case TF_IDENTITY:
        std::copy(activation, activation + nvals, outputs);
        return;
        
    case TF_LOGSIG:
        for (unsigned i = 0;  i < nvals;  ++i) {
            // See https://bugzilla.redhat.com/show_bug.cgi?id=521190
            // for why we use double version of exp
            outputs[i] = 1.0 / (1.0 + exp((double)-activation[i]));
        }
        break;
        
    case TF_TANH:
        for (unsigned i = 0;  i < nvals;  ++i)
            outputs[i] = tanh(activation[i]);
        break;
        
    case TF_LOGSOFTMAX: {
        double total = 0.0;
        
        for (unsigned i = 0;  i < nvals;  ++i) {
            // See https://bugzilla.redhat.com/show_bug.cgi?id=521190
            // for why we use double version of exp
            total += (outputs[i] = exp((double)activation[i]));
        }

        double factor = 1.0 / total;

        for (unsigned i = 0;  i < nvals;  ++i)
            outputs[i] *= factor;

        break;
    }

    default:
        throw Exception("Layer::transfer(): invalid transfer_function");
    }
}

void
Layer::
transfer(const float * activation, float * outputs) const
{
    transfer(activation, outputs, this->outputs(), transfer_function);
}

void
Layer::
transfer(const double * activation, double * outputs) const
{
    transfer(activation, outputs, this->outputs(), transfer_function);
}

distribution<float>
Layer::
transfer(const distribution<float> & activation) const
{
    int no = outputs();
    distribution<float> output(no);
    transfer(&activation[0], &output[0]);
    return output;
}

distribution<double>
Layer::
transfer(const distribution<double> & activation) const
{
    int no = outputs();
    distribution<double> output(no);
    transfer(&activation[0], &output[0]);
    return output;
}

template<class Float>
void
Layer::
derivative(const Float * outputs, Float * deriv, int nvals,
           Transfer_Function_Type transfer_function)
{
    switch (transfer_function) {

    case TF_IDENTITY:
        std::fill(deriv, deriv + nvals, 1.0);
        break;
        
    case TF_LOGSIG:
        for (unsigned i = 0;  i < nvals;  ++i)
            deriv[i] = outputs[i] * (1.0 - outputs[i]);
        break;
        
    case TF_TANH:
        for (unsigned i = 0;  i < nvals;  ++i)
            deriv[i] = 1.0 - (outputs[i] * outputs[i]);
        break;

    case TF_LOGSOFTMAX:
        for (unsigned i = 0;  i < nvals;  ++i)
            deriv[i] = 1.0 / outputs[i];
        break;
        
    default:
        throw Exception("Layer::transfer(): invalid transfer_function");
    }
}

distribution<float>
Layer::
derivative(const distribution<float> & outputs) const
{
    if (outputs.size() != this->outputs())
        throw Exception("derivative(): wrong size");
    int no = this->outputs();
    distribution<float> result(no);
    derivative(&outputs[0], &result[0], no, transfer_function);
    return result;
}

distribution<double>
Layer::
derivative(const distribution<double> & outputs) const
{
    if (outputs.size() != this->outputs())
        throw Exception("derivative(): wrong size");
    int no = this->outputs();
    distribution<double> result(no);
    derivative(&outputs[0], &result[0], no, transfer_function);
    return result;
}

void
Layer::
derivative(const float * outputs,
           float * derivatives) const
{
    int no = this->outputs();
    derivative(outputs, derivatives, no, transfer_function);
}

void
Layer::
derivative(const double * outputs,
           double * derivatives) const
{
    int no = this->outputs();
    derivative(outputs, derivatives, no, transfer_function);
}

template<class Float>
void
Layer::
second_derivative(const Float * outputs, Float * deriv, int nvals,
                  Transfer_Function_Type transfer_function)
{
    switch (transfer_function) {

    case TF_IDENTITY:
        std::fill(deriv, deriv + nvals, 0.0);
        break;
        
    case TF_TANH:
        for (unsigned i = 0;  i < nvals;  ++i)
            deriv[i] = -2.0 * outputs[i] * (1.0 - (outputs[i] * outputs[i]));
        break;

#if 0
    case TF_LOGSIG:
        for (unsigned i = 0;  i < nvals;  ++i)
            deriv[i] = ...;
        break;
        
    case TF_LOGSOFTMAX:
        for (unsigned i = 0;  i < nvals;  ++i)
            deriv[i] = ...;
        break;
#endif
        
    default:
        throw Exception("Layer::transfer(): second derivative not implemented "
                        "for this transfer_function "
                        + ML::print(transfer_function));
    }
}

distribution<float>
Layer::
second_derivative(const distribution<float> & outputs) const
{
    if (outputs.size() != this->outputs())
        throw Exception("second_derivative(): wrong size");
    int no = this->outputs();
    distribution<float> result(no);
    second_derivative(&outputs[0], &result[0], no, transfer_function);
    return result;
}

distribution<double>
Layer::
second_derivative(const distribution<double> & outputs) const
{
    if (outputs.size() != this->outputs())
        throw Exception("second_derivative(): wrong size");
    int no = this->outputs();
    distribution<double> result(no);
    second_derivative(&outputs[0], &result[0], no, transfer_function);
    return result;
}

void
Layer::
second_derivative(const float * outputs,
                  float * second_derivatives) const
{
    int no = this->outputs();
    second_derivative(outputs, second_derivatives, no, transfer_function);
}

void
Layer::
second_derivative(const double * outputs,
                  double * second_derivatives) const
{
    int no = this->outputs();
    second_derivative(outputs, second_derivatives, no, transfer_function);
}

void
Layer::
deltas(const float * outputs, const float * errors, float * deltas) const
{
    derivative(outputs, deltas);
    int no = this->outputs();
    for (unsigned i = 0;  i < no;  ++i)
        deltas[i] *= errors[i];
}

void
Layer::
deltas(const double * outputs, const double * errors, double * deltas) const
{
    derivative(outputs, deltas);
    int no = this->outputs();
    for (unsigned i = 0;  i < no;  ++i)
        deltas[i] *= errors[i];
}

void
Layer::
validate() const
{
    // Default does none
}

std::pair<float, float>
Layer::
targets(float maximum, Transfer_Function_Type transfer_function)
{
    switch (transfer_function) {
    case TF_TANH:
    case TF_IDENTITY: return std::make_pair(-maximum, maximum);
    case TF_LOGSOFTMAX:
    case TF_LOGSIG: return std::make_pair(0.0f, maximum);
    default:
        throw Exception("Layer::targets(): invalid transfer_function");
    }
}

} // namespace ML


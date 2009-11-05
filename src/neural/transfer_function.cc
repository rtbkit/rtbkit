/* transfer_function.cc
   Jeremy Barnes, 4 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Transfer function implementation.
*/

#include "transfer_function.h"

namespace ML {

/*****************************************************************************/
/* RANGE_TYPE                                                                */
/*****************************************************************************/


/*****************************************************************************/
/* RANGE                                                                     */
/*****************************************************************************/


/*****************************************************************************/
/* TRANSFER_FUNCTION                                                         */
/*****************************************************************************/


template<typename FloatIn>
void
Tranfer_Function::
transfer(const FloatIn * activation, FloatIn * outputs, int nvals,
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
        throw Exception("Tranfer_Function::transfer(): invalid transfer_function");
    }
}

void
Tranfer_Function::
transfer(const float * activation, float * outputs) const
{
    transfer(activation, outputs, this->outputs(), transfer_function);
}

void
Tranfer_Function::
transfer(const double * activation, double * outputs) const
{
    transfer(activation, outputs, this->outputs(), transfer_function);
}

distribution<float>
Tranfer_Function::
transfer(const distribution<float> & activation) const
{
    int no = outputs();
    distribution<float> output(no);
    transfer(&activation[0], &output[0]);
    return output;
}

distribution<double>
Tranfer_Function::
transfer(const distribution<double> & activation) const
{
    int no = outputs();
    distribution<double> output(no);
    transfer(&activation[0], &output[0]);
    return output;
}

template<class Float>
template<typename FloatIn>
void
Tranfer_Function::
derivative(const FloatIn * outputs, FloatIn * deriv, int nvals,
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
        throw Exception("Tranfer_Function::transfer(): invalid transfer_function");
    }
}

distribution<float>
Tranfer_Function::
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
Tranfer_Function::
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
Tranfer_Function::
derivative(const float * outputs,
           float * derivatives) const
{
    int no = this->outputs();
    derivative(outputs, derivatives, no, transfer_function);
}

void
Tranfer_Function::
derivative(const double * outputs,
           double * derivatives) const
{
    int no = this->outputs();
    derivative(outputs, derivatives, no, transfer_function);
}

template<class Float>
template<typename FloatIn>
void
Tranfer_Function::
second_derivative(const FloatIn * outputs, FloatIn * deriv, int nvals,
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
        throw Exception("Tranfer_Function::transfer(): second derivative not implemented "
                        "for this transfer_function "
                        + ML::print(transfer_function));
    }
}

distribution<float>
Tranfer_Function::
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
Tranfer_Function::
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
Tranfer_Function::
second_derivative(const float * outputs,
                  float * second_derivatives) const
{
    int no = this->outputs();
    second_derivative(outputs, second_derivatives, no, transfer_function);
}

void
Tranfer_Function::
second_derivative(const double * outputs,
                  double * second_derivatives) const
{
    int no = this->outputs();
    second_derivative(outputs, second_derivatives, no, transfer_function);
}

void
Tranfer_Function::
deltas(const float * outputs, const float * errors, float * deltas) const
{
    derivative(outputs, deltas);
    int no = this->outputs();
    for (unsigned i = 0;  i < no;  ++i)
        deltas[i] *= errors[i];
}

void
Tranfer_Function::
deltas(const double * outputs, const double * errors, double * deltas) const
{
    derivative(outputs, deltas);
    int no = this->outputs();
    for (unsigned i = 0;  i < no;  ++i)
        deltas[i] *= errors[i];
}

std::pair<float, float>
Tranfer_Function::
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

void
Transfer_Function::
poly_serialize(ML::DB::Store_Writer & store) const
{
}

std::string
Transfer_Function::
print() const
{
}

boost::shared_ptr<Transfer_Function>
Transfer_Function::
poly_reconstitute(ML::DB::Store_Reader & store)
{
}


/*****************************************************************************/
/* FACTORY                                                                   */
/*****************************************************************************/

boost::shared_ptr<Transfer_Function>
create_transfer_function(const Transfer_Function_Type & function)
{
}

boost::shared_ptr<Transfer_Function>
create_transfer_function(const std::string & name)
{
}


} // namespace ML

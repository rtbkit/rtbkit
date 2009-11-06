/* transfer_function.cc
   Jeremy Barnes, 4 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Transfer function implementation.
*/

#include "transfer_function.h"
#include <cmath>
#include "db/persistent.h"
#include "boosting/registry.h"
#include "utils/smart_ptr_utils.h"


using namespace ML::DB;
using namespace std;

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

void
Transfer_Function::
poly_serialize(DB::Store_Writer & store) const
{
    Registry<Transfer_Function>::singleton().serialize(store, this);
}

boost::shared_ptr<Transfer_Function>
Transfer_Function::
poly_reconstitute(DB::Store_Reader & store)
{
    return Registry<Transfer_Function>::singleton().reconstitute(store);
}

distribution<float>
Transfer_Function::
transfer(const distribution<float> & activation) const
{
    int no = activation.size();
    distribution<float> output(no);
    transfer(&activation[0], &output[0], no);
    return output;
}

distribution<double>
Transfer_Function::
transfer(const distribution<double> & activation) const
{
    int no = activation.size();
    distribution<double> output(no);
    transfer(&activation[0], &output[0], no);
    return output;
}

distribution<float>
Transfer_Function::
derivative(const distribution<float> & outputs) const
{
    int no = outputs.size();
    distribution<float> result(no);
    derivative(&outputs[0], &result[0], no);
    return result;
}

distribution<double>
Transfer_Function::
derivative(const distribution<double> & outputs) const
{
    int no = outputs.size();
    distribution<double> result(no);
    derivative(&outputs[0], &result[0], no);
    return result;
}

distribution<float>
Transfer_Function::
second_derivative(const distribution<float> & outputs) const
{
    int no = outputs.size();
    distribution<float> result(no);
    second_derivative(&outputs[0], &result[0], no);
    return result;
}

distribution<double>
Transfer_Function::
second_derivative(const distribution<double> & outputs) const
{
    int no = outputs.size();
    distribution<double> result(no);
    second_derivative(&outputs[0], &result[0], no);
    return result;
}


/*****************************************************************************/
/* STANDARD_TRANSFER_FUNCTION                                                */
/*****************************************************************************/

Standard_Transfer_Function::
Standard_Transfer_Function(Transfer_Function_Type transfer_function)
    : transfer_function(transfer_function)
{
}

std::string
Standard_Transfer_Function::
print() const
{
    return ML::print(transfer_function);
}

Range
Standard_Transfer_Function::
range() const
{
    throw Exception("Standard_Transfer_Function::range(): not implemented");
}

std::pair<float, float>
Standard_Transfer_Function::
targets(float maximum) const
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
Standard_Transfer_Function::
serialize(DB::Store_Writer & store) const
{
    store << (char)0 // version
          << transfer_function;
}

void
Standard_Transfer_Function::
reconstitute(DB::Store_Reader & store)
{
    char version;
    store >> version;
    if (version != 0)
        throw Exception("Standard_Transfer_Function::reconstitute(): "
                        "unknown version");
    store >> transfer_function;
}

std::string
Standard_Transfer_Function::
class_id() const
{
    return "Standard";
}

template<typename FloatIn>
void
Standard_Transfer_Function::
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
        throw Exception("Standard_Transfer_Function::transfer(): invalid transfer_function");
    }
}

void
Standard_Transfer_Function::
transfer(const float * activation, float * outputs, size_t n) const
{
    transfer(activation, outputs, n, transfer_function);
}

void
Standard_Transfer_Function::
transfer(const double * activation, double * outputs, size_t n) const
{
    transfer(activation, outputs, n, transfer_function);
}

template<typename FloatIn>
void
Standard_Transfer_Function::
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
        throw Exception("Standard_Transfer_Function::transfer(): invalid transfer_function");
    }
}

void
Standard_Transfer_Function::
derivative(const float * outputs, float * derivatives, size_t n) const
{
    derivative(outputs, derivatives, n, transfer_function);
}

void
Standard_Transfer_Function::
derivative(const double * outputs, double * derivatives, size_t n) const
{
    derivative(outputs, derivatives, n, transfer_function);
}

template<typename FloatIn>
void
Standard_Transfer_Function::
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
        throw Exception("Standard_Transfer_Function::transfer(): second derivative not implemented "
                        "for this transfer_function "
                        + ML::print(transfer_function));
    }
}

void
Standard_Transfer_Function::
second_derivative(const float * outputs, float * second_derivatives,
                  size_t n) const
{
    second_derivative(outputs, second_derivatives, n, transfer_function);
}

void
Standard_Transfer_Function::
second_derivative(const double * outputs, double * second_derivatives,
                  size_t n) const
{
    second_derivative(outputs, second_derivatives, n, transfer_function);
}

namespace {

Register_Factory<Transfer_Function, Standard_Transfer_Function>
    STF_REGISTER("Standard");

} // file scope



/*****************************************************************************/
/* FACTORY                                                                   */
/*****************************************************************************/

boost::shared_ptr<Transfer_Function>
create_transfer_function(const Transfer_Function_Type & function)
{
    return make_sp(new Standard_Transfer_Function(function));
}

boost::shared_ptr<Transfer_Function>
create_transfer_function(const std::string & name)
{
    throw Exception("create_transfer_function(name): not implemented");
}


} // namespace ML

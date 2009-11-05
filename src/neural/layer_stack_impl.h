/* layer_stack_impl.h                                              -*- C++ -*-
   Jeremy Barnes, 4 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Tools for a stack of layers.
*/

#ifndef __jml__neural__layer_stack_impl_h__
#define __jml__neural__layer_stack_impl_h__

#include "layer_stack.h"
#include "utils/smart_ptr_utils.h"

namespace ML {

template<class LayerT>
Layer_Stack<LayerT>::
Layer_Stack(const std::string & name)
    : Layer(name, 0, 0)
{
}

template<class OtherLayer>
template<class LayerT>
Layer_Stack<LayerT>::
Layer_Stack(const Layer_Stack<OtherLayer> & other)
    : Layer(other.name(), 0, 0)
{
}

template<class LayerT>
void
Layer_Stack<LayerT>::
add(const LayerT * layer)
{
    return add(make_sp(layer));
}

template<class LayerT>
void
Layer_Stack<LayerT>::
add(const boost::shared_ptr<LayerT> & layer)
{
}

template<class LayerT>
std::string
Layer_Stack<LayerT>::
print() const
{
}
    
template<class LayerT>
std::string
Layer_Stack<LayerT>::
type() const
{
}

template<class LayerT>
void
Layer_Stack<LayerT>::
serialize(DB::Store_Writer & store) const
{
}

template<class LayerT>
void
Layer_Stack<LayerT>::
reconstitute(DB::Store_Reader & store)
{
}

template<class LayerT>
boost::shared_ptr<Parameters>
Layer_Stack<LayerT>::
parameters()
{
}

template<class LayerT>
void
Layer_Stack<LayerT>::
apply(const float * input, float * output) const
{
}

template<class LayerT>
void
Layer_Stack<LayerT>::
apply(const double * input, double * output) const
{
}

template<class LayerT>
size_t
Layer_Stack<LayerT>::
fprop_temporary_space_required() const
{
    size_t result = 0;
    for (unsigned i = 0;  i < size();  ++i)
        result += layers_[i]->fprop_temporary_space_required();
    return result;
}

template<class LayerT>
distribution<float>
Layer_Stack<LayerT>::
fprop(const distribution<float> & inputs,
      float * temp_space, size_t temp_space_size) const
{
}

template<class LayerT>
distribution<double>
Layer_Stack<LayerT>::
fprop(const distribution<double> & inputs,
      double * temp_space, size_t temp_space_size) const
{
    distribution<double> prev_outputs = inputs;

    size_t temp_space_required = layers_[i]->fprop_temporary_space_required();

    double temp_space_start[temp_space_required];
    double * temp_space = temp_space_start;
    double * temp_space_end = temp_space_start + temp_space_required;

    for (unsigned i = 0;  i < size();  ++i) {
        int layer_temp_space_size
            = layers_[i]->fprop_temporary_space_required();

        prev_outputs = layers_[i]->fprop(prev_outputs, temp_space,
                                         layer_temp_space_size);
        
        temp_space += layer_temp_space_size;
        if (temp_space > temp_space_end
            || (i == size() - 1 && temp_space != temp_space_end))
            throw Exception("temp space out of sync");
    }

    return prev_outputs;
}
    
template<class LayerT>
void
Layer_Stack<LayerT>::
bprop(const distribution<float> & output_errors,
      float * temp_space, size_t temp_space_size,
      Parameters & gradient,
      distribution<float> & input_errors,
      double example_weight,
      bool calculate_input_errors) const
{
}

template<class LayerT>
void
Layer_Stack<LayerT>::
bprop(const distribution<double> & output_errors_in,
      double * temp_space_start, size_t temp_space_size,
      Parameters & gradient,
      distribution<double> & input_errors,
      double example_weight,
      bool calculate_input_errors) const
{
    double * temp_space_end = temp_space_start + temp_space_size;
    double * curr_temp_space = temp_space_end;

    distribution<double> output_errors = output_errors_in;

    for (int i = size() - 1;  i >= 0;  --i) {
        int layer_temp_space_size
            = layers_[i]->fprop_temporary_space_required();
        curr_temp_space -= layer_temp_space_size;

        if (curr_temp_space < temp_space_start)
            throw Exception("Layer temp space was out of sync");
        
        distribution<double> new_output_errors;

        layers_[i]->bprop(output_errors, temp_space, layer_temp_space_size,
                          gradient.submodel(i), new_output_errors,
                          example_weight,
                          (calculate_input_errors || i > 0));

        output_errors.swap(new_output_errors);
    }

    if (curr_temp_space != temp_space_start)
        throw Exception("Layer_Stack::bprop(): out of sync");
}

template<class LayerT>
void
Layer_Stack<LayerT>::
random_fill(float limit, Thread_Context & context)
{
    for (unsigned i = 0;  i < size();  ++i)
        layers_[i]->random_fill(limit, context);
}

template<class LayerT>
void
Layer_Stack<LayerT>::
zero_fill()
{
    for (unsigned i = 0;  i < size();  ++i)
        layers_[i]->zero_fill();
}

template<class LayerT>
size_t
Layer_Stack<LayerT>::
parameter_count() const
{
    size_t result = 0;
    for (unsigned i = 0;  i < size();  ++i)
        result += layers_[i]->parameter_count();
    return result;
}

template<class LayerT>
Layer_Stack<LayerT> *
Layer_Stack<LayerT>::
make_copy() const
{
    return new Layer_Stack<LayerT>(*this);
}

template<class LayerT>
Layer_Stack<LayerT> *
Layer_Stack<LayerT>::
deep_copy() const
{
    std::auto_ptr<Layer_Stack<LayerT> > result(new Layer_Stack<LayerT>(name()));
    for (unsigned i = 0;  i < size();  ++i)
        result->add(layers_[i]->deep_copy());
    return result.release();
}

} // namespace ML


#endif /* __jml__neural__layer_stack_impl_h__ */

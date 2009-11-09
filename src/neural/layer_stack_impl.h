/* layer_stack_impl.h                                              -*- C++ -*-
   Jeremy Barnes, 4 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Tools for a stack of layers.
*/

#ifndef __jml__neural__layer_stack_impl_h__
#define __jml__neural__layer_stack_impl_h__

#include "layer_stack.h"
#include "utils/smart_ptr_utils.h"
#include "db/persistent.h"

namespace ML {

/*****************************************************************************/
/* LAYER_STACK                                                               */
/*****************************************************************************/


template<class LayerT>
Layer_Stack<LayerT>::
Layer_Stack()
    : Layer("", 0, 0)
{
}


template<class LayerT>
Layer_Stack<LayerT>::
Layer_Stack(const std::string & name)
    : Layer(name, 0, 0)
{
}

template<class LayerT>
Layer_Stack<LayerT>::
Layer_Stack(const Layer_Stack & other)
    : Layer(other.name(), 0, 0)
{
    throw Exception("not finished");
}

template<class LayerT>
Layer_Stack<LayerT> &
Layer_Stack<LayerT>::
operator = (const Layer_Stack & other)
{
    Layer::operator = (other);
    layers_ = other.layers_;
    //if (typeid(*this) == typeid(Layer_Stack<LayerT>))
    //    add_parameters();
    throw Exception("not finished");
    return *this;
}

template<class LayerT>
void
Layer_Stack<LayerT>::
swap(Layer_Stack & other)
{
    throw Exception("not implemented");
}

template<class LayerT>
void
Layer_Stack<LayerT>::
add(LayerT * layer)
{
    add(make_sp(layer));
}

template<class LayerT>
void
Layer_Stack<LayerT>::
add(boost::shared_ptr<LayerT> layer)
{
    if (!layer)
        throw Exception("Layer_Stack::add(): added null layer");

    if (empty()) {
        this->inputs_ = layer->inputs();
        this->outputs_ = layer->outputs();
    }
    else {
        if (layer->inputs() != outputs())
            throw Exception("incompatible layer sizes");
        outputs_ = layer->outputs();
    }

    layers_.push_back(layer);
    layer->add_parameters(parameters_.subparams(layers_.size() - 1,
                                                layer->name()));
}

template<class LayerT>
void
Layer_Stack<LayerT>::
clear()
{
    layers_.clear();
    inputs_ = outputs_ = 0;
    update_parameters();
}

template<class LayerT>
std::string
Layer_Stack<LayerT>::
print() const
{
    std::string result = format("Layer_Stack: name \"%s\", %zd layers",
                                this->name().c_str(), size());
    for (unsigned i = 0;  i < size();  ++i)
        result += format("layer %d\n", i) + layers_[i]->print();
    return result;
}
    
template<class LayerT>
std::string
Layer_Stack<LayerT>::
class_id() const
{
    // All layer stacks serialize and reconstitute as the base; they can be
    // converted after reconstitution.
    return "Layer_Stack";
}

template<class LayerT>
void
Layer_Stack<LayerT>::
serialize(DB::Store_Writer & store) const
{
    using namespace DB;

    store << (char)0 // version
          << compact_size_t(layers_.size());

    for (unsigned i = 0;  i < size();  ++i)
        layers_[i]->poly_serialize(store);

    store << compact_size_t(1849202);
}

template<class LayerT>
void
Layer_Stack<LayerT>::
reconstitute(DB::Store_Reader & store)
{
    using namespace DB;

    char version;
    store >> version;
    if (version != 0)
        throw Exception("Layer_Stack::reconstitute(): invalid version");

    compact_size_t sz(store);
    layers_.resize(sz);

    for (unsigned i = 0;  i < sz;  ++i) {
        boost::shared_ptr<Layer> layer
            = Layer::poly_reconstitute(store);
        boost::shared_ptr<LayerT> cast
            = boost::dynamic_pointer_cast<LayerT>(layer);
        if (!cast)
            throw Exception("Layer_Stack::reconstitute(): couldn't convert");
        layers_[i] = cast;
    }

    compact_size_t canary(store);
    if (canary != 1849202)
        throw Exception("Layer_Stack::reconstitute(): invalid canary");
}

template<class LayerT>
void
Layer_Stack<LayerT>::
add_parameters(Parameters & params)
{
    for (unsigned i = 0;  i < layers_.size();  ++i)
        layers_[i]->add_parameters(params.subparams(i, layers_[i]->name()));
}

template<class LayerT>
void
Layer_Stack<LayerT>::
apply(const float * input, float * output) const
{
    float tmp1[max_width_], tmp2[max_width_];
    float * next_output = tmp1, * next_input = tmp2;
    for (unsigned l = 0;  l < layers_.size();  ++l) {
        const float * i = (l == 0 ? input : next_input);
        float * o = (l == layers_.size() - 1 ? next_output : output);

        layers_[l]->apply(i, o);
        std::swap(next_output, next_input);
    }
}

template<class LayerT>
void
Layer_Stack<LayerT>::
apply(const double * input, double * output) const
{
    double tmp1[max_width_], tmp2[max_width_];
    double * next_output = tmp1, * next_input = tmp2;
    for (unsigned l = 0;  l < layers_.size();  ++l) {
        const double * i = (l == 0 ? input : next_input);
        double * o = (l == layers_.size() - 1 ? next_output : output);

        layers_[l]->apply(i, o);
        std::swap(next_output, next_input);
    }
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
    distribution<float> prev_outputs = inputs;

    float * temp_space_start = temp_space;
    float * temp_space_end = temp_space_start + temp_space_size;

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
distribution<double>
Layer_Stack<LayerT>::
fprop(const distribution<double> & inputs,
      double * temp_space, size_t temp_space_size) const
{
    distribution<double> prev_outputs = inputs;

    double * temp_space_start = temp_space;
    double * temp_space_end = temp_space_start + temp_space_size;

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
      double * temp_space, size_t temp_space_size,
      Parameters & gradient,
      distribution<double> & input_errors,
      double example_weight,
      bool calculate_input_errors) const
{
    double * temp_space_start = temp_space;
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
                          gradient.subparams(i, layers_[i]->name()),
                          new_output_errors,
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
std::pair<float, float>
Layer_Stack<LayerT>::
targets(float maximum) const
{
    if (empty())
        throw Exception("targets(): doesn't work for empty stack");
    return layers_.back()->targets(maximum);
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
    std::auto_ptr<Layer_Stack<LayerT> > result
        (new Layer_Stack<LayerT>(this->name()));
    for (unsigned i = 0;  i < size();  ++i)
        result->add(layers_[i]->deep_copy());
    return result.release();
}

template<class LayerT>
bool
Layer_Stack<LayerT>::
equal_impl(const Layer & other) const
{
    if (typeid(*this) != typeid(other)) return false;
    const Layer_Stack & cast
        = reinterpret_cast<const Layer_Stack &>(other);
    return operator == (cast);
}

template<class LayerT>
bool
Layer_Stack<LayerT>::
operator == (const Layer_Stack & other) const
{
    if (!Layer::operator == (other)) return false;
    if (size() != other.size()) return false;
    for (unsigned i = 0;  i < size();  ++i) {
        if (layers_[i] == other.layers_[i]) continue;
        if (layers_[i] && other.layers_[i]
            && (layers_[i]->equal(*other.layers_[i]))) continue;
        return false;
    }
    return true;
}

} // namespace ML


#endif /* __jml__neural__layer_stack_impl_h__ */

/* layer_stack_impl.h                                              -*- C++ -*-
   Jeremy Barnes, 4 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Tools for a stack of layers.
*/

#ifndef __jml__neural__layer_stack_impl_h__
#define __jml__neural__layer_stack_impl_h__

#include "layer_stack.h"
#include "jml/utils/smart_ptr_utils.h"
#include "jml/db/persistent.h"
#include "auto_encoder_stack.h"

namespace ML {


/*****************************************************************************/
/* LAYER_STACK                                                               */
/*****************************************************************************/


template<class LayerT>
Layer_Stack<LayerT>::
Layer_Stack()
    : Layer("", 0, 0), max_width_(0), max_internal_width_(0)
{
}


template<class LayerT>
Layer_Stack<LayerT>::
Layer_Stack(const std::string & name)
    : Layer(name, 0, 0), max_width_(0), max_internal_width_(0)
{
}

template<class LayerT>
Layer_Stack<LayerT>::
Layer_Stack(const Layer_Stack & other)
    : Layer(other.name(), 0, 0), max_width_(0), max_internal_width_(0)
{
    for (unsigned i = 0;  i < other.size();  ++i)
        add(other.layers_[i]);
}

template<class LayerT>
Layer_Stack<LayerT>::
Layer_Stack(const Layer_Stack & other, Deep_Copy_Tag)
    : Layer(other.name(), 0, 0), max_width_(0), max_internal_width_(0)
{
    for (unsigned i = 0;  i < other.size();  ++i)
        add_cast(make_sp(other.layers_[i]->deep_copy()));
}

template<class LayerT>
Layer_Stack<LayerT>::
Layer_Stack(const Auto_Encoder_Stack & other)
    : Layer(other.name(), 0, 0), max_width_(0), max_internal_width_(0)
{
    for (unsigned i = 0;  i < other.size();  ++i)
        add_cast(other.layers_.layers_[i]);
}

template<class LayerT>
Layer_Stack<LayerT>::
Layer_Stack(const Auto_Encoder_Stack & other, Deep_Copy_Tag)
    : Layer(other.name(), 0, 0), max_width_(0), max_internal_width_(0)
{
    for (unsigned i = 0;  i < other.size();  ++i)
        add_cast(make_sp(other.layers_[i].deep_copy()));
}

template<class LayerT>
Layer_Stack<LayerT> &
Layer_Stack<LayerT>::
operator = (const Layer_Stack & other)
{
    Layer_Stack new_me(other);
    swap(new_me);
    return *this;
}

template<class LayerT>
Layer_Stack<LayerT> &
Layer_Stack<LayerT>::
operator = (const Auto_Encoder_Stack & other)
{
    Layer_Stack new_me(other);
    swap(new_me);
    return *this;
}

template<class LayerT>
void
Layer_Stack<LayerT>::
swap(Layer_Stack & other)
{
    Layer::swap(other);
    std::swap(max_width_, other.max_width_);
    std::swap(max_internal_width_, other.max_internal_width_);
    layers_.swap(other.layers_);
}

template<class LayerT>
void
Layer_Stack<LayerT>::
add(LayerT * layer)
{
    add(make_sp(layer));
}

template<class LayerT>
bool
Layer_Stack<LayerT>::
supports_missing_inputs() const
{
    if (empty()) return false;
    return layers_[0]->supports_missing_inputs();
}


template<class LayerT>
const Transfer_Function &
Layer_Stack<LayerT>::
transfer() const
{
    if (empty())
        throw Exception("no transfer function");
    return back().transfer();
}

template<class LayerT>
void
Layer_Stack<LayerT>::
add(std::shared_ptr<LayerT> layer)
{
    if (!layer)
        throw Exception("Layer_Stack::add(): added null layer");

    if (empty()) {
        this->inputs_ = layer->inputs();
        this->outputs_ = layer->outputs();
    }
    else {
        if (layer->inputs() != outputs()) {
            using namespace std;
            for (unsigned i = 0;  i < layers_.size();  ++i) {
                cerr << "layer " << i << ": inputs " << layers_[i]->inputs()
                     << " outputs " << layers_[i]->outputs() << " " << endl;
            }
            cerr << "inputs() = " << inputs() << " outputs " << outputs()
                 << endl;
            cerr << "layer->inputs() = " << layer->inputs()
                 << " outputs = " << layer->outputs() << endl;
            throw Exception("incompatible layer sizes");
        }
        outputs_ = layer->outputs();
    }

    layers_.push_back(layer);
    layer->add_parameters(parameters_.subparams(layers_.size() - 1,
                                                layer->name()));
    max_width_ = std::max(max_width_, layer->max_width());

    if (size() > 1) {
        max_internal_width_ = 0;
        for (unsigned i = 0;  i < size();  ++i) {
            size_t sz = (i == 0 ? layers_[i]->outputs()
                         : (i == size() - 1 ? layers_[i]->inputs()
                            : layers_[i]->max_width()));
            max_internal_width_ = std::max(max_internal_width_, sz);
        }
    }
}

template<class LayerT>
void
Layer_Stack<LayerT>::
clear()
{
    layers_.clear();
    inputs_ = outputs_ = 0;
    max_width_ = 0;
    max_internal_width_ = 0;
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

    store << (char)1 // version
          << name()
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
    if (version != 1) {
        using namespace std;
        cerr << "version = " << (int)version << endl;
        cerr << "store.offset() = " << store.offset() << endl;
        throw Exception("Layer_Stack::reconstitute(): invalid version");
    }

    store >> name_;

    clear();

    compact_size_t sz(store);
    layers_.reserve(sz);

    for (unsigned i = 0;  i < sz;  ++i) {
        std::shared_ptr<Layer> layer
            = Layer::poly_reconstitute(store);
        std::shared_ptr<LayerT> cast
            = std::dynamic_pointer_cast<LayerT>(layer);
        if (!cast)
            throw Exception("Layer_Stack::reconstitute(): couldn't convert");
        add(cast);
    }

    compact_size_t canary(store);
    if (canary != 1849202)
        throw Exception("Layer_Stack::reconstitute(): invalid canary");

    validate();
    update_parameters();
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
template<typename F>
void
Layer_Stack<LayerT>::
apply(const F * input, F * output) const
{
    F tmp[max_internal_width_];

    for (unsigned l = 0;  l < layers_.size();  ++l) {
        const F * i = (l == 0 ? input : tmp);
        F * o = (l == layers_.size() - 1 ? output : tmp);

        layers_[l]->apply(i, o);
    }
}

template<class LayerT>
void
Layer_Stack<LayerT>::
apply(const float * input, float * output) const
{
    apply<float>(input, output);
}

template<class LayerT>
void
Layer_Stack<LayerT>::
apply(const double * input, double * output) const
{
    apply<double>(input, output);
}

template<class LayerT>
size_t
Layer_Stack<LayerT>::
fprop_temporary_space_required() const
{
    // We need: inputs of all layers except the first,
    // temporary space of all layers in between.
    //
    // +-----------+-------+-------------+--------+---...
    // |   l0 tmp  | l0 out|  l1 tmp     |  l1 out| l2 tmp
    // +-----------+-------+-------------+--------+---...

    size_t result = 0;

    for (unsigned i = 0;  i < size();  ++i) {
        if (i != 0) result += layers_[i]->inputs();
        result += layers_[i]->fprop_temporary_space_required();
    }

    return result;
}

template<class LayerT>
template<class F>
void
Layer_Stack<LayerT>::
fprop(const F * inputs,
      F * temp_space, size_t temp_space_size,
      F * outputs) const
{
    F * temp_space_start = temp_space;
    F * temp_space_end = temp_space_start + temp_space_size;

    const F * curr_inputs = inputs;

    for (unsigned i = 0;  i < size();  ++i) {
        int layer_temp_space_size
            = layers_[i]->fprop_temporary_space_required();

        F * curr_outputs
            = (i == size() - 1
               ? outputs
               : temp_space + layer_temp_space_size);
        
        layers_[i]->fprop(curr_inputs, temp_space, layer_temp_space_size,
                          curr_outputs);

        curr_inputs = curr_outputs;

        temp_space += layer_temp_space_size;
        if (i != size() - 1) temp_space += layers_[i]->outputs();

        if (temp_space > temp_space_end
            || (i == size() - 1 && temp_space != temp_space_end))
            throw Exception("temp space out of sync");
    }
}

template<class LayerT>
void
Layer_Stack<LayerT>::
fprop(const float * inputs,
      float * temp_space, size_t temp_space_size,
      float * outputs) const
{
    return fprop<float>(inputs, temp_space, temp_space_size, outputs);
}

template<class LayerT>
void
Layer_Stack<LayerT>::
fprop(const double * inputs,
      double * temp_space, size_t temp_space_size,
      double * outputs) const
{
    return fprop<double>(inputs, temp_space, temp_space_size, outputs);
}

template<class LayerT>
template<typename F>
void
Layer_Stack<LayerT>::
bprop(const F * inputs,
      const F * outputs,
      const F * temp_space, size_t temp_space_size,
      const F * output_errors,
      F * input_errors,
      Parameters & gradient,
      double example_weight) const
{
    const F * temp_space_start = temp_space;
    const F * temp_space_end = temp_space_start + temp_space_size;
    const F * curr_temp_space = temp_space_end;

    const F * curr_outputs = outputs;

    // Storage for the errors kept between the layers
    F error_storage[max_internal_width() + 1];
    error_storage[max_internal_width()] = F(0.1234567);

    for (int i = size() - 1;  i >= 0;  --i) {
        int layer_temp_space_size
            = layers_[i]->fprop_temporary_space_required();

        curr_temp_space -= layer_temp_space_size;

        if (curr_temp_space < temp_space_start)
            throw Exception("Layer temp space was out of sync");

        const F * curr_inputs
            = (i == 0 ? inputs : curr_temp_space - layers_[i]->inputs());

        const F * curr_output_errors
            = (i == size() - 1 ? output_errors : error_storage);

        F * curr_input_errors
            = (i == 0 ? input_errors : error_storage);

        layers_[i]->bprop(curr_inputs, curr_outputs, curr_temp_space,
                          layer_temp_space_size, curr_output_errors,
                          curr_input_errors,
                          gradient.subparams(i, layers_[i]->name()),
                          example_weight);

        // Make sure that we didn't write outside of where we should have
        if (error_storage[max_internal_width()] != F(0.1234567))
            throw Exception("Layer_Stack::bprop(): layer bprop wrote too "
                            "far");

        curr_outputs = curr_inputs;
        if (i != 0) curr_temp_space -= layers_[i]->inputs();
    }

    if (curr_temp_space != temp_space_start)
        throw Exception("Layer_Stack::bprop(): out of sync");
}

template<class LayerT>
void
Layer_Stack<LayerT>::
bprop(const float * inputs,
      const float * outputs,
      const float * temp_space, size_t temp_space_size,
      const float * output_errors,
      float * input_errors,
      Parameters & gradient,
      double example_weight) const
{
    bprop<float>(inputs, outputs, temp_space, temp_space_size,
                 output_errors, input_errors, gradient, example_weight);
}

template<class LayerT>
void
Layer_Stack<LayerT>::
bprop(const double * inputs,
      const double * outputs,
      const double * temp_space, size_t temp_space_size,
      const double * output_errors,
      double * input_errors,
      Parameters & gradient,
      double example_weight) const
{
    bprop<double>(inputs, outputs, temp_space, temp_space_size,
                  output_errors, input_errors, gradient, example_weight);
}

template<class LayerT>
template<typename F>
void
Layer_Stack<LayerT>::
bbprop(const F * inputs,
       const F * outputs,
       const F * temp_space, size_t temp_space_size,
       const F * output_errors,
       const F * doutput_errors,
       F * input_errors,
       F * dinput_errors,
       Parameters & gradient,
       Parameters * dgradient,
       double example_weight) const
{
    if (dinput_errors == 0 && dgradient == 0)
        return bprop(inputs, outputs, temp_space, temp_space_size,
                     output_errors, input_errors, gradient, example_weight);
    
    const F * temp_space_start = temp_space;
    const F * temp_space_end = temp_space_start + temp_space_size;
    const F * curr_temp_space = temp_space_end;

    const F * curr_outputs = outputs;

    // Storage for the errors kept between the layers
    F error_storage[max_internal_width()];
    F derror_storage[max_internal_width()];

    for (int i = size() - 1;  i >= 0;  --i) {
        int layer_temp_space_size
            = layers_[i]->fprop_temporary_space_required();

        curr_temp_space -= layer_temp_space_size;

        const F * curr_inputs
            = (i == 0 ? inputs : curr_temp_space - layers_[i]->inputs());

        const F * curr_output_errors
            = (i == size() - 1 ? output_errors : error_storage);

        const F * curr_doutput_errors
            = (i == size() - 1 ? doutput_errors : derror_storage);

        F * curr_input_errors
            = (i == 0 ? input_errors : error_storage);
        F * curr_dinput_errors
            = (i == 0 ? dinput_errors : derror_storage);

        Parameters * current_dgradient
            = (dgradient ? &dgradient->subparams(i, layers_[i]->name()) : 0);

        layers_[i]->bbprop(curr_inputs, curr_outputs, curr_temp_space,
                           layer_temp_space_size,
                           curr_output_errors,
                           curr_doutput_errors,
                           curr_input_errors,
                           curr_dinput_errors,
                           gradient.subparams(i, layers_[i]->name()),
                           current_dgradient,
                           example_weight);

        if (curr_temp_space < temp_space_start)
            throw Exception("Layer temp space was out of sync");

        curr_outputs = curr_inputs;
        if (i != 0) curr_temp_space -= layers_[i]->inputs();
    }

    if (curr_temp_space != temp_space_start)
        throw Exception("Layer_Stack::bprop(): out of sync");
}
 
template<class LayerT>
void
Layer_Stack<LayerT>::
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
    bbprop<float>(inputs, outputs, temp_space, temp_space_size,
                  output_errors, d2output_errors, input_errors,
                  d2input_errors, gradient, dgradient, example_weight);
}
 
template<class LayerT>
void
Layer_Stack<LayerT>::
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
    bbprop<double>(inputs, outputs, temp_space, temp_space_size,
                   output_errors, d2output_errors, input_errors,
                   d2input_errors, gradient, dgradient, example_weight);
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
        result->add_cast(make_sp(layers_[i]->deep_copy()));
    return result.release();
}

template<class LayerT>
bool
Layer_Stack<LayerT>::
equal_impl(const Layer & other) const
{
    using namespace std;
    cerr << "equal_impl: typeid(*this).name() = "
         << typeid(*this).name()
         << " typeid(other).name() "
         << typeid(other).name()
         << endl;

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
#if 0
    using namespace std;
    cerr << "operator ==" << endl;
    cerr << "typeid(*this).name() = "
         << typeid(*this).name()
         << " typeid(other).name() "
         << typeid(other).name()
         << endl;

    cerr << "inputs_ = " << inputs_ << endl;
    cerr << "other.inputs_ = " << other.inputs_ << endl;
    cerr << "outputs_ = " << outputs_ << endl;
    cerr << "other.outputs_ = " << other.outputs_ << endl;
#endif
    if (!Layer::operator == (other)) return false;
    //cerr << "layers same" << endl;
    if (size() != other.size()) return false;
    //cerr << "size same" << endl;
    for (unsigned i = 0;  i < size();  ++i) {
        if (layers_[i] == other.layers_[i]) continue;
        if (layers_[i] && other.layers_[i]
            && (layers_[i]->equal(*other.layers_[i]))) continue;
        //cerr << "layer " << i << " not equal" << endl;
        return false;
    }
    return true;
}

} // namespace ML


#endif /* __jml__neural__layer_stack_impl_h__ */

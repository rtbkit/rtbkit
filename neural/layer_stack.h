/* layer_stack.h                                                   -*- C++ -*-
   Jeremy Barnes, 4 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Stack of neural network layers where each one feeds its output as in input
   into the next.
*/

#ifndef __jml__neural__layer_stack_h__
#define __jml__neural__layer_stack_h__

#include "layer.h"
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits/is_base_of.hpp>
#include "jml/arch/demangle.h"


namespace ML {

/// Tag structure to force a deep copy
struct Deep_Copy_Tag {
};

struct Auto_Encoder_Stack;


/*****************************************************************************/
/* LAYER_STACK                                                               */
/*****************************************************************************/

/** This class is a stack of layers connected in a simple fashion: each
    layer takes as its input the output of the previous layer.

    NOTE: this class does *not* own its layers; it merely keeps a reference
    to them.  If you copy the class using the default operators, the new
    copy will refer to the same layer objects.  If you need to perform a
    deep copy, you should use one of the methods associated with it.
    
    Another class will be written for more complicated arrangements.
*/

template<class LayerT>
struct Layer_Stack : public Layer {

    Layer_Stack();
    Layer_Stack(const std::string & name);
    Layer_Stack(const Layer_Stack & other);
    Layer_Stack(const Layer_Stack & other, Deep_Copy_Tag);
    Layer_Stack(const Auto_Encoder_Stack & other);
    Layer_Stack(const Auto_Encoder_Stack & other, Deep_Copy_Tag);

    template<class OtherLayer>
    Layer_Stack(const Layer_Stack<OtherLayer> & other)
        : Layer(other.name(), 0, 0), max_width_(0), max_internal_width_(0)
    {
        for (unsigned i = 0;  i < other.size();  ++i)
            add_cast(other.layers_[i]);
    }

    template<class OtherLayer>
    Layer_Stack(const Layer_Stack<OtherLayer> & other,
                Deep_Copy_Tag)
        : Layer(other.name(), 0, 0), max_width_(0), max_internal_width_(0)
    {
        for (unsigned i = 0;  i < other.size();  ++i)
            add_cast(other.layers_[i]->deep_copy());
    }

    Layer_Stack & operator = (const Layer_Stack & other);
    Layer_Stack & operator = (const Auto_Encoder_Stack & other);

    template<class OtherLayer>
    Layer_Stack &
    operator = (const Layer_Stack<OtherLayer> & other)
    {
        Layer_Stack new_me(other);
        swap(new_me);
        return *this;
    }

    void swap(Layer_Stack & other);

    size_t size() const { return layers_.size(); }
    bool empty() const { return layers_.empty(); }

    void clear();

    virtual size_t max_width() const { return max_width_; }
    size_t max_internal_width() const { return max_internal_width_; }

    virtual bool supports_missing_inputs() const;

    virtual const Transfer_Function & transfer() const;

    const LayerT & operator [] (int index) const { return *layers_.at(index); }
    LayerT & operator [] (int index) { return *layers_.at(index); }

    LayerT & back() { return operator [] (size() - 1); }
    const LayerT & back() const { return operator [] (size() - 1); }

    template<typename As>
    const As & get_as(int index) const
    {
        return dynamic_cast<const As &>(*layers_.at(index));
    }

    template<typename As>
    As & get_as(int index)
    {
        return dynamic_cast<As &>(*layers_.at(index));
    }

    std::shared_ptr<LayerT> share(int index)
    {
        return layers_.at(index);
    }

    /** Add a layer to the stack.  Checks the preconditions first.  The
        ownership of the pointer passes to the Layer_Stack object. */
    void add(LayerT * layer);

    void add(std::shared_ptr<LayerT> layer);

    /** Add a layer to the stack, performing the necessary upcast. */
    template<typename LayerT2>
    typename boost::disable_if<boost::is_base_of<LayerT, LayerT2>, void>::type
    add_cast(std::shared_ptr<LayerT2> layer)
    {
        if (!layer)
            throw Exception("no layer");
        std::shared_ptr<LayerT> cast
            = std::dynamic_pointer_cast<LayerT>(layer);
        if (!cast)
            throw Exception("Layer_Stack::add_cast(): type "
                            + demangle(typeid(*layer).name())
                            + " can't be upcast to "
                            + demangle(typeid(LayerT).name()));
        add(cast);
    }
    
    /** Add a layer to the stack.  No cast is necessary, as the type is
        derived. */
    template<typename LayerT2>
    typename boost::enable_if<boost::is_base_of<LayerT, LayerT2>, void>::type
    add_cast(std::shared_ptr<LayerT2> layer)
    {
        add(layer);
    }
    
    /** Dump as ASCII.  This will be big. */
    virtual std::string print() const;
    
    /** Return the name of the type */
    virtual std::string class_id() const;

    virtual void serialize(DB::Store_Writer & store) const;
    virtual void reconstitute(DB::Store_Reader & store);

    /** Add all of our parameters to the given parameters object. */
    virtual void add_parameters(Parameters & params);


    /*************************************************************************/
    /* APPLY                                                                 */
    /*************************************************************************/

    template<typename F>
    void apply(const F * input, F * output) const;

    virtual void apply(const float * input, float * output) const;
    virtual void apply(const double * input, double * output) const;

    using Layer::apply;


    /*************************************************************************/
    /* FPROP                                                                 */
    /*************************************************************************/

    /** Return the amount of space necessary to save temporary results for the
        forward prop.  There will be an array of the given precision (double
        or single) provided.

        Default implementation returns outputs().
    */

    virtual size_t fprop_temporary_space_required() const;

    using Layer::fprop;

    template<typename F>
    void fprop(const F * inputs,
               F * temp_space, size_t temp_space_size,
               F * outputs) const;

    virtual void
    fprop(const float * inputs,
          float * temp_space, size_t temp_space_size,
          float * outputs) const;

    virtual void
    fprop(const double * inputs,
          double * temp_space, size_t temp_space_size,
          double * outputs) const;

               

    /*************************************************************************/
    /* BPROP                                                                 */
    /*************************************************************************/

    using Layer::bprop;

    template<typename F>
    void bprop(const F * inputs,
               const F * outputs,
               const F * temp_space, size_t temp_space_size,
               const F * output_errors,
               F * input_errors,
               Parameters & gradient,
               double example_weight) const;

    virtual void bprop(const float * inputs,
                       const float * outputs,
                       const float * temp_space, size_t temp_space_size,
                       const float * output_errors,
                       float * input_errors,
                       Parameters & gradient,
                       double example_weight) const;

    virtual void bprop(const double * inputs,
                       const double * outputs,
                       const double * temp_space, size_t temp_space_size,
                       const double * output_errors,
                       double * input_errors,
                       Parameters & gradient,
                       double example_weight) const;

    template<typename F>
    void bbprop(const F * inputs,
                const F * outputs,
                const F * temp_space, size_t temp_space_size,
                const F * output_errors,
                const F * d2output_errors,
                F * input_errors,
                F * d2input_errors,
                Parameters & gradient,
                Parameters * dgradient,
                double example_weight) const; 
 
    virtual void bbprop(const float * inputs,
                        const float * outputs,
                        const float * temp_space, size_t temp_space_size,
                        const float * output_errors,
                        const float * d2output_errors,
                        float * input_errors,
                        float * d2input_errors,
                        Parameters & gradient,
                        Parameters * dgradient,
                        double example_weight) const;
 
    virtual void bbprop(const double * inputs,
                        const double * outputs,
                        const double * temp_space, size_t temp_space_size,
                        const double * output_errors,
                        const double * d2output_errors,
                        double * input_errors,
                        double * d2input_errors,
                        Parameters & gradient,
                        Parameters * dgradient,
                        double example_weight) const;

    virtual void random_fill(float limit, Thread_Context & context);

    virtual void zero_fill();

    virtual size_t parameter_count() const;

    virtual std::pair<float, float> targets(float maximum) const;

    virtual Layer_Stack * make_copy() const;

    virtual Layer_Stack * deep_copy() const;

    virtual bool equal_impl(const Layer & other) const;

    bool operator == (const Layer_Stack & other) const;
    bool operator != (const Layer_Stack & other) const
    {
        return ! operator == (other);
    }

protected:
    /// The actual layers
    std::vector<std::shared_ptr<LayerT> > layers_;

    /// Maximum width over the entire stack
    size_t max_width_;

    /// Maximum width, excluding the input and the output
    size_t max_internal_width_;
    template<class LayerT2> friend class Layer_Stack;
};

extern template class Layer_Stack<Layer>;

JML_IMPL_SERIALIZE_RECONSTITUTE_TEMPLATE(class LayerT, Layer_Stack<LayerT>);


} // namespace ML


#endif /* __jml__neural__layer_stack_h__ */

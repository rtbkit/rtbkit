/* auto_encoder_stack.h                                            -*- C++ -*-
   Jeremy Barnes, 11 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Stack of auto-encoders.
*/

#ifndef __jml__neural__auto_encoder_stack_h__
#define __jml__neural__auto_encoder_stack_h__


#include "auto_encoder.h"
#include "layer_stack.h"

namespace ML {

// This is instantiated in the .cc file to save work for the compiler
extern template class Layer_Stack<Auto_Encoder>;


/*****************************************************************************/
/* AUTO_ENCODER_STACK                                                        */
/*****************************************************************************/

/** A stack of auto-encoders, that composes their actions. */

struct Auto_Encoder_Stack : public Auto_Encoder {

    Auto_Encoder_Stack();
    Auto_Encoder_Stack(const std::string & name);

    Auto_Encoder_Stack(const Auto_Encoder_Stack & other, Deep_Copy_Tag);

    template<class OtherLayer>
    Auto_Encoder_Stack(const Layer_Stack<OtherLayer> & other)
        : Layer(other.name(), 0, 0)
    {
        for (unsigned i = 0;  i < other.size();  ++i)
            add_cast(other.layers_[i]);
    }

    template<class OtherLayer>
    Auto_Encoder_Stack(const Layer_Stack<OtherLayer> & other,
                Deep_Copy_Tag)
        : Layer(other.name(), 0, 0)
    {
        for (unsigned i = 0;  i < other.size();  ++i)
            add_cast(other.layers_[i]->deep_copy());
    }

    void swap(Auto_Encoder_Stack & other);

    operator const Layer_Stack<Auto_Encoder> & () const { return layers_; }

    size_t size() const { return layers_.size(); }
    bool empty() const { return layers_.empty(); }

    void clear();

    virtual size_t max_width() const { return layers_.max_width(); }
    size_t max_internal_width() const { return layers_.max_internal_width(); }

    const Auto_Encoder & operator [] (int index) const
    {
        return layers_[index];
    }

    Auto_Encoder & operator [] (int index)
    {
        return layers_[index];
    }
    
    template<typename As>
    const As & get_as(int index) const
    {
        return dynamic_cast<const As &>(layers_[index]);
    }

    template<typename As>
    As & get_as(int index)
    {
        return dynamic_cast<As &>(layers_[index]);
    }

    /** Add a layer to the stack.  Checks the preconditions first.  The
        ownership of the pointer passes to the Auto_Encoder_Stack object. */
    void add(Auto_Encoder * layer);

    void add(std::shared_ptr<Auto_Encoder> layer);

    /** Add a layer to the stack, performing the necessary upcast. */
    template<typename LayerT2>
    typename boost::disable_if<boost::is_base_of<Auto_Encoder, LayerT2>, void>::type
    add_cast(std::shared_ptr<LayerT2> layer)
    {
        if (!layer)
            throw Exception("no layer");
        std::shared_ptr<Auto_Encoder> cast
            = std::dynamic_pointer_cast<Auto_Encoder>(layer);
        if (!cast)
            throw Exception("Auto_Encoder_Stack::add_cast(): type "
                            + demangle(typeid(*layer).name())
                            + " can't be upcast to "
                            + demangle(typeid(Auto_Encoder).name()));
        add(cast);
    }
    
    /** Add a layer to the stack.  No cast is necessary, as the type is
        derived. */
    template<typename LayerT2>
    typename boost::enable_if<boost::is_base_of<Auto_Encoder, LayerT2>, void>::type
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
    /* FORWARD DIRECTION                                                     */
    /*************************************************************************/

    virtual void apply(const float * input, float * output) const;
    virtual void apply(const double * input, double * output) const;

    using Layer::apply;


    virtual size_t fprop_temporary_space_required() const;

    using Layer::fprop;

    virtual void
    fprop(const float * inputs,
          float * temp_space, size_t temp_space_size,
          float * outputs) const;

    virtual void
    fprop(const double * inputs,
          double * temp_space, size_t temp_space_size,
          double * outputs) const;

               
    using Layer::bprop;

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


    /*************************************************************************/
    /* INVERSE DIRECTION                                                     */
    /*************************************************************************/

    virtual std::pair<float, float> itargets(float maximum) const;
    virtual bool supports_missing_outputs() const;

    virtual void iapply(const float * input, float * output) const;
    virtual void iapply(const double * input, double * output) const;

    template<typename F>
    void iapply(const F * output, F * input) const;

    using Auto_Encoder::iapply;

    /* Note that these functions will not normally be used, as a stacked
       autoencoder would normally be trained greedily one layer at a
       time.  Using these functions will transfer the gradient over the
       entire stack, training all layers at once.
    */

    virtual size_t ifprop_temporary_space_required() const;

    virtual void
    ifprop(const float * outputs,
           float * temp_space, size_t temp_space_size,
           float * inputs) const;

    virtual void
    ifprop(const double * outputs,
           double * temp_space, size_t temp_space_size,
           double * inputs) const;

    template<typename F>
    void
    ifprop(const F * outputs,
           F * temp_space, size_t temp_space_size,
           F * inputs) const;

    virtual void ibprop(const float * outputs,
                        const float * inputs,
                        const float * temp_space, size_t temp_space_size,
                        const float * input_errors,
                        float * output_errors,
                        Parameters & gradient,
                        double example_weight) const;
    
    virtual void ibprop(const double * outputs,
                        const double * inputs,
                        const double * temp_space, size_t temp_space_size,
                        const double * input_errors,
                        double * output_errors,
                        Parameters & gradient,
                        double example_weight) const;

    template<typename F>
    void ibprop(const F * outputs,
                const F * inputs,
                const F * temp_space, size_t temp_space_size,
                const F * input_errors,
                F * output_errors,
                Parameters & gradient,
                double example_weight) const;

    virtual void random_fill(float limit, Thread_Context & context);

    virtual void zero_fill();

    virtual size_t parameter_count() const;

    virtual std::pair<float, float> targets(float maximum) const;

    virtual bool supports_missing_inputs() const;

    virtual Auto_Encoder_Stack * make_copy() const;

    virtual Auto_Encoder_Stack * deep_copy() const;

    virtual bool equal_impl(const Layer & other) const;

    bool operator == (const Auto_Encoder_Stack & other) const;
    bool operator != (const Auto_Encoder_Stack & other) const
    {
        return ! operator == (other);
    }

    Layer_Stack<Auto_Encoder> layers_;
};


} // namespace ML

#endif /* __jml__neural__auto_encoder_stack_h__ */

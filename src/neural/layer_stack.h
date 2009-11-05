/* layer_stack.h                                                   -*- C++ -*-
   Jeremy Barnes, 4 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Stack of neural network layers where each one feeds its output as in input
   into the next.
*/

#ifndef __jml__neural__layer_stack_h__
#define __jml__neural__layer_stack_h__

#include "layer.h"

namespace ML {

template<class LayerT>
struct Layer_Stack : public Layer {

    Layer_Stack(const std::string & name);

    template<class OtherLayer>
    Layer_Stack(const Layer_Stack<OtherLayer> & other);
                
    

    size_t size() const { return layers_.size(); }
    bool empty() const { return layers_.empty(); }

    const LayerT & operator [] (int index) const { return *layers_.at(index); }
    LayerT & operator [] (int index) { return *layers_.at(index); }

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

    /** Add a layer to the stack.  Checks the preconditions first. */
    void add(const LayerT * layer);

    void add(const boost::shared_ptr<LayerT> & layer);


    /** Dump as ASCII.  This will be big. */
    virtual std::string print() const;
    
    /** Return the name of the type */
    virtual std::string type() const;

    virtual void serialize(DB::Store_Writer & store) const;
    virtual void reconstitute(DB::Store_Reader & store);

    virtual boost::shared_ptr<Parameters> parameters();


    /*************************************************************************/
    /* APPLY                                                                 */
    /*************************************************************************/

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

    /** These functions perform a forward propagation.  They also save whatever
        information is necessary to perform an efficient backprop at a later
        period in time.

        Default implementation calls apply() and saves the outputs only in the
        temporary space.
    */
    virtual distribution<float>
    fprop(const distribution<float> & inputs,
          float * temp_space, size_t temp_space_size) const;

    virtual distribution<double>
    fprop(const distribution<double> & inputs,
          double * temp_space, size_t temp_space_size) const;
    
               

    /*************************************************************************/
    /* BPROP                                                                 */
    /*************************************************************************/

    /** Perform a back propagation.  Given the derivative of the error with
        respect to each of the errors, they compute the gradient of the
        parameter space.
    */

    virtual void bprop(const distribution<float> & output_errors,
                       float * temp_space, size_t temp_space_size,
                       Parameters & gradient,
                       distribution<float> & input_errors,
                       double example_weight,
                       bool calculate_input_errors) const;

    virtual void bprop(const distribution<double> & output_errors,
                       double * temp_space, size_t temp_space_size,
                       Parameters & gradient,
                       distribution<double> & input_errors,
                       double example_weight,
                       bool calculate_input_errors) const;

    /** Fill with random weights. */
    virtual void random_fill(float limit, Thread_Context & context);

    virtual void zero_fill();

    /** Return the number of parameters (degrees of freedom) for the
        layer. */
    virtual size_t parameter_count() const;

    /** Copy the object */
    virtual Layer_Stack * make_copy() const;

    /** Copy the object, making a deep copy of all of the underlying objects */
    virtual Layer_Stack * deep_copy() const;

protected:
    std::vector<boost::shared_ptr<LayerT> > layers_;
};

extern template class Layer_Stack<Layer>;


} // namespace ML


#endif /* __jml__neural__layer_stack_h__ */

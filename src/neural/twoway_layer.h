/* twoway_layer.h                                                  -*- C++ -*-
   Jeremy Barnes, 4 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Two way layer, that can generate its data.
*/

#ifndef __jml__neural__twoway_layer_h__
#define __jml__neural__twoway_layer_h__


#include "dense_layer.h"
#include "layer_stack.h"


namespace ML {

template<class LayerT> class Layer_Stack;


/*****************************************************************************/
/* TWOWAY_LAYER_UPDATES                                                      */
/*****************************************************************************/

struct Twoway_Layer;

struct Twoway_Layer_Updates {

    Twoway_Layer_Updates();

    Twoway_Layer_Updates(bool train_generative,
                         const Twoway_Layer & layer);

    Twoway_Layer_Updates(bool use_dense_missing,
                         bool train_generative,
                         int inputs, int outputs);

    void zero_fill();

    void init(bool train_generative,
              const Twoway_Layer & layer);

    void init(bool use_dense_missing, bool train_generative,
              int inputs, int outputs);

    int inputs() const { return weights.shape()[0]; }
    int outputs() const { return weights.shape()[1]; }

    Twoway_Layer_Updates & operator += (const Twoway_Layer_Updates & other);

    bool use_dense_missing;
    bool train_generative;
    boost::multi_array<double, 2> weights;
    distribution<double> bias;
    distribution<double> missing_replacements;
    std::vector<distribution<double> > missing_activations;
    distribution<double> ibias;
    distribution<double> iscales;
    distribution<double> hscales;
};


/*****************************************************************************/
/* TWOWAY_LAYER                                                              */
/*****************************************************************************/

/** A perceptron layer that has both a forward and a reverse direction.  It's
    both a discriminative model (in the forward direction) and a generative
    model (in the reverse direction).
*/

struct Twoway_Layer : public Dense_Layer<float> {
    typedef Dense_Layer<float> Base;

    Twoway_Layer();

    Twoway_Layer(const std::string & name,
                 size_t inputs, size_t outputs,
                 Transfer_Function_Type transfer,
                 Missing_Values missing_values,
                 Thread_Context & context,
                 float limit = -1.0);

    Twoway_Layer(const std::string & name,
                 size_t inputs, size_t outputs,
                 Transfer_Function_Type transfer,
                 Missing_Values missing_values);

    virtual boost::shared_ptr<Parameters> parameters();

    /// Bias for the reverse direction
    distribution<float> ibias;

    /// Scaling factors for the reverse direction
    distribution<float> iscales;
    distribution<float> hscales;

    distribution<double> iapply(const distribution<double> & output) const;
    distribution<float> iapply(const distribution<float> & output) const;

    distribution<double> ipreprocess(const distribution<double> & output) const;
    distribution<float> ipreprocess(const distribution<float> & output) const;

    distribution<double> iactivation(const distribution<double> & output) const;
    distribution<float> iactivation(const distribution<float> & output) const;

    distribution<double>
    itransfer(const distribution<double> & activation) const;
    distribution<float>
    itransfer(const distribution<float> & activation) const;

    distribution<double> iderivative(const distribution<double> & input) const;
    distribution<float> iderivative(const distribution<float> & input) const;

    void update(const Twoway_Layer_Updates & updates, double learning_rate);

    virtual void random_fill(float limit, Thread_Context & context);

    virtual void zero_fill();

    virtual void serialize(DB::Store_Writer & store) const;
    virtual void reconstitute(DB::Store_Reader & store);

    /** Dump as ASCII.  This will be big. */
    virtual std::string print() const;
    
    /** Backpropagate the given example.  The gradient will be acculmulated in
        the output.  Fills in the errors for the next stage at input_errors. */
    void backprop_example(const distribution<double> & outputs,
                          const distribution<double> & output_deltas,
                          const distribution<double> & inputs,
                          distribution<double> & input_deltas,
                          Twoway_Layer_Updates & updates) const;

    /** Inverse direction backpropagation of the given example.  Again, the
        gradient will be acculmulated in the output.  Fills in the errors for
        the next stage at input_errors. */
    void ibackprop_example(const distribution<double> & outputs,
                           const distribution<double> & output_deltas,
                           const distribution<double> & inputs,
                           distribution<double> & input_deltas,
                           Twoway_Layer_Updates & updates) const;

    /** Trains a single iteration on the given data with the selected
        parameters.  Returns a moving estimate of the RMSE on the
        training set. */
    std::pair<double, double>
    train_iter(const std::vector<distribution<float> > & data,
               float prob_cleared,
               Thread_Context & thread_context,
               int minibatch_size, float learning_rate,
               int verbosity,
               float sample_proportion,
               bool randomize_order);

    /** Tests on the given dataset, returning the exact and noisy RMSE.  If
        data_out is non-empty, then it will also fill it in with the
        hidden representations for each of the inputs (with no noise added).
        This information can be used to train the next layer. */
    std::pair<double, double>
    test_and_update(const std::vector<distribution<float> > & data_in,
                    std::vector<distribution<float> > & data_out,
                    float prob_cleared,
                    Thread_Context & thread_context,
                    int verbosity) const;

    /** Tests on the given dataset, returning the exact and noisy RMSE. */
    std::pair<double, double>
    test(const std::vector<distribution<float> > & data,
         float prob_cleared,
         Thread_Context & thread_context,
         int verbosity) const
    {
        std::vector<distribution<float> > dummy;
        return test_and_update(data, dummy, prob_cleared, thread_context,
                               verbosity);
    }

    bool operator == (const Twoway_Layer & other) const;
};

IMPL_SERIALIZE_RECONSTITUTE(Twoway_Layer);

extern template class Layer_Stack<Twoway_Layer>;

} // namespace ML


#endif /* __jml__neural__twoway_layer_h__ */

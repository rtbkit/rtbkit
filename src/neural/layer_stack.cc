/* layer_stack.cc
   Jeremy Barnes, 4 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Stack of layers.
*/

#include "layer_stack.h"
#include "dense_layer.h"
#include "layer_stack_impl.h"
#include "boosting/registry.h"

namespace ML {

template class Layer_Stack<Layer>;
template class Layer_Stack<Dense_Layer<float> >;
template class Layer_Stack<Dense_Layer<double> >;

namespace {

Register_Factory<Layer, Layer_Stack<Layer> >
LAYER_STACK_REGISTER("Layer_Stack");

} // file scope



#if 0

/*****************************************************************************/
/* DNAE_STACK                                                                */
/*****************************************************************************/

/** A stack of denoising autoencoder layers, to create a deep encoder. */

struct DNAE_Stack : public std::vector<Twoway_Layer> {

    distribution<float> apply(const distribution<float> & input) const;
    distribution<double> apply(const distribution<double> & input) const;

    distribution<float> iapply(const distribution<float> & output) const;
    distribution<double> iapply(const distribution<double> & output) const;

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);

    /** Train a single example.  Returns the RMSE in the first and the
        output value (which can be used to calculate the AUC) in the
        second.
    */
    std::pair<double, double>
    train_discrim_example(const distribution<float> & data,
                          float label,
                          DNAE_Stack_Updates & udpates) const;

    /** Trains a single iteration on the given data with the selected
        parameters.  Returns a moving estimate of the RMSE on the
        training set. */
    std::pair<double, double>
    train_discrim_iter(const std::vector<distribution<float> > & data,
                       const std::vector<float> & labels,
                       Thread_Context & thread_context,
                       int minibatch_size, float learning_rate,
                       int verbosity,
                       float sample_proportion,
                       bool randomize_order);

    /** Perform backpropagation given an error gradient.  Note that doing
        so will adversely affect the performance of the autoencoder, as
        the reverse weights aren't modified when performing this training.

        Returns the best training and testing error.
    */
    std::pair<double, double>
    train_discrim(const std::vector<distribution<float> > & training_data,
                  const std::vector<float> & training_labels,
                  const std::vector<distribution<float> > & testing_data,
                  const std::vector<float> & testing_labels,
                  const Configuration & config,
                  ML::Thread_Context & thread_context);

    /** Update given the learning rate and the gradient. */
    void update(const DNAE_Stack_Updates & updates, double learning_rate);
    
    /** Test the discriminative power of the network.  Returns the RMS error
        or AUC depending upon whether it's a regression or classification
        task.
    */
    std::pair<double, double>
    test_discrim(const std::vector<distribution<float> > & data,
                 const std::vector<float> & labels,
                 ML::Thread_Context & thread_context,
                 int verbosity);
    

    /** Train (unsupervised) as a stack of denoising autoencoders. */
    void train_dnae(const std::vector<distribution<float> > & training_data,
                    const std::vector<distribution<float> > & testing_data,
                    const Configuration & config,
                    ML::Thread_Context & thread_context);

    /** Tests on both pristine and noisy inputs.  The first returned is the
        error on pristine inputs.  The second is the error on noisy inputs.
        the prob_cleared parameter describes the probability that noise will
        be added to any given input.
    */
    std::pair<double, double>
    test_dnae(const std::vector<distribution<float> > & data,
              float prob_cleared,
              ML::Thread_Context & thread_context,
              int verbosity) const;
    
    bool operator == (const DNAE_Stack & other) const;
};


IMPL_SERIALIZE_RECONSTITUTE(DNAE_Stack);


/*****************************************************************************/
/* DNAE_STACK                                                                */
/*****************************************************************************/

distribution<float>
DNAE_Stack::
apply(const distribution<float> & input) const
{
    distribution<float> output = input;
    
    // Go down the stack
    for (unsigned l = 0;  l < size();  ++l)
        output = (*this)[l].apply(output);
    
    return output;
}

distribution<double>
DNAE_Stack::
apply(const distribution<double> & input) const
{
    distribution<double> output = input;
    
    // Go down the stack
    for (unsigned l = 0;  l < size();  ++l)
        output = (*this)[l].apply(output);
    
    return output;
}

distribution<float>
DNAE_Stack::
iapply(const distribution<float> & output) const
{
    distribution<float> input = output;
    
    // Go down the stack
    for (int l = size() - 1;  l >= 0;  --l)
        input = (*this)[l].iapply(input);
    
    return input;
}

distribution<double>
DNAE_Stack::
iapply(const distribution<double> & output) const
{
    distribution<double> input = output;
    
    // Go down the stack
    for (int l = size() - 1;  l >= 0;  --l)
        input = (*this)[l].iapply(input);
    
    return input;
}

void
DNAE_Stack::
serialize(ML::DB::Store_Writer & store) const
{
    store << (char)1; // version
    store << compact_size_t(size());
    for (unsigned i = 0;  i < size();  ++i)
        (*this)[i].serialize(store);
}

void
DNAE_Stack::
reconstitute(ML::DB::Store_Reader & store)
{
    char version;
    store >> version;
    if (version != 1) {
        cerr << "version = " << (int)version << endl;
        throw Exception("DNAE_Stack::reconstitute(): invalid version");
    }
    compact_size_t sz(store);
    resize(sz);

    for (unsigned i = 0;  i < sz;  ++i)
        (*this)[i].reconstitute(store);
}

void
DNAE_Stack::
update(const DNAE_Stack_Updates & updates, double learning_rate)
{
    if (updates.size() != size())
        throw Exception("DNAE_Stack::update(): updates have the wrong size");

    for (unsigned i = 0;  i < size();  ++i)
        (*this)[i].update(updates[i], learning_rate);
}


#endif

} // namespace ML


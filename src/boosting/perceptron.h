/* perceptron.h                                                    -*- C++ -*-
   Jeremy Barnes, 16 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   The multi layer perceptron algorithm, implemented as a classifier.
*/

#ifndef __boosting__perceptron_h__
#define __boosting__perceptron_h__


#include "config.h"
#include "classifier.h"
#include "stump.h"
#include "utils/pair_utils.h"
#include <boost/multi_array.hpp>
#include "perceptron_defs.h"
#include <boost/shared_ptr.hpp>

namespace ML {


class Label;
class Thread_Context;


/*****************************************************************************/
/* PERCEPTRON                                                                */
/*****************************************************************************/

/** The boosted decision stumps classifier.  It has a whole bunch of decision
    stumps (decision trees with only one decision), which it combines together
    linearly (with weights based upon the boosting algorithm) in order to
    determine the best output.
*/

class Perceptron : public Classifier_Impl {
public:
    Perceptron();

    Perceptron(DB::Store_Reader & store,
               const boost::shared_ptr<const Feature_Space> & feature_space);
    
    Perceptron(const boost::shared_ptr<const Feature_Space> & feature_space,
               const Feature & predicted);

    /** Swap two Perceptron objects.  Guaranteed not to throw an
        exception. */
    void swap(Perceptron & other)
    {
        throw Exception("Perceptron::swap(): not implemented");
    }

    using Classifier_Impl::predict;

    /** Predict the score for a single class. */
    virtual float predict(int label, const Feature_Set & features) const;
    
    /** Predict the score for all classes. */
    virtual distribution<float> predict(const Feature_Set & features) const;

    /** Calculate the accuracy.  This method takes a set of decorrelated
        samples.  The accuracy can be calculated much faster in this case
        as there is no need to decorrelate nor extract the features. */
    float accuracy(const boost::multi_array<float, 2> & decorrelated,
                   const std::vector<Label> & labels,
                   const distribution<float> & example_weights
                       = UNIFORM_WEIGHTS) const;

    /** Apply the first layer to a dataset to decorrelate it. */
    boost::multi_array<float, 2> decorrelate(const Training_Data & data) const;
        
    using Classifier_Impl::accuracy;

    virtual std::string print() const;

    virtual std::vector<Feature> all_features() const;

    virtual Output_Encoding output_encoding() const;

    /** Serialization and reconstitution. */
    virtual void serialize(DB::Store_Writer & store) const;
    virtual void
    reconstitute(DB::Store_Reader & store,
                 const boost::shared_ptr<const Feature_Space> & features);

    virtual std::string class_id() const { return "PERCEPTRON"; }
    
    virtual Perceptron * make_copy() const;

    /** Parse the architecture. */
    static std::vector<int> parse_architecture(const std::string & arch);


    /*************************************************************************/
    /* INTERNAL VARIABLES                                                    */
    /*************************************************************************/

    /* The internal data flow is as follows:

       Features -> feature vector -> decorrelation -> layer1 -> (...) -> output

       Note that the decorrelation is simply an extra layer with the identity
       function and no bias input.
    */

    /* Variables... */
    std::vector<Feature> features;  ///< Features to use as input
    
    /** This structure holds a layer of neurons. */
    struct Layer {
        Layer();
        Layer(const Layer & other);
        Layer(size_t inputs, size_t units, Activation activation);

        Layer & operator = (const Layer & other)
        {
            weights = other.weights;
            bias = other.bias;
            activation = other.activation;
            return *this;
        }

        boost::multi_array<double, 2> weights;
        distribution<double> bias;
        Activation activation;   ///< Activation function
        
        /** Dump as ASCII.  This will be big. */
        std::string print() const;
        
        void serialize(DB::Store_Writer & store) const;
        void reconstitute(DB::Store_Reader & store);

        /** Apply the layer to the input and return an output. */
        distribution<float> apply(const distribution<float> & input) const;
        
        void apply(const distribution<float> & input,
                   distribution<float> & output) const;

        void apply_stochastic(const distribution<float> & input,
                              distribution<float> & output,
                              Thread_Context & context) const;

        void apply(const float * input, float * output) const;

        void apply_stochastic(const float * input, float * output,
                              Thread_Context & context) const;

        /** Generate a single stochastic Gibbs sample from the stocastic
            distribution, starting from the given input values.  It will
            modify both the input and the output of the new sample.

            Performs the given number of iterations.
        */
        void sample(distribution<float> & input,
                    distribution<float> & output,
                    Thread_Context & context,
                    int num_iterations) const;

        /** Transform the given value according to the transfer function. */
        void transform(distribution<float> & values) const;

        /** Return the derivative of the given value according to the transfer
            function. */
        distribution<float> derivative(const distribution<float> & outputs) const;

        void deltas(const float * outputs, const float * errors,
                    float * deltas) const;

        /** Fill with random weights. */
        void random_fill(float limit = 0.1);

        /** Return the number of parameters (degrees of freedom) for the
            layer. */
        size_t parameters() const
        {
            return bias.size() + (inputs() * outputs());
        }

        size_t inputs() const { return weights.shape()[0]; }
        size_t outputs() const { return weights.shape()[1]; }
    };
    
    std::vector<boost::shared_ptr<Layer> > layers;  ///< Different layers
    
    size_t max_units;  ///< Number of units in layer with highest number

    /** Perform the transformation given the activation function. */
    static void transform(distribution<float> & values,
                          Activation activation);

    /** Perform the transformation given the activation function. */
    static void transform(float * values, size_t nv, Activation activation);

    /** Perform the transformation given the activation function. */
    static void derivative(distribution<float> & values,
                           Activation activation);

    static std::pair<float, float> targets(float maximum, int activation);
    
    void add_layer(const boost::shared_ptr<Layer> & layer);

    void clear();

    size_t parameters() const;

    template<class OutputIterator>
    void extract_features(const Feature_Set & fs, OutputIterator result) const
    {
        Feature_Set::const_iterator it = fs.begin(), end = fs.end();
        int i = 0, ni = features.size();

        for(; i < ni && it != end;  ++it) {
            if (it.feature() == features[i]) {
                *result++ = it.value();
                ++i;
            }
            else if (i != 0 && it.feature() == features[i - 1])
                throw Exception("duplicate feature values");
            else if (features[i] < it.feature())
                throw Exception("feature missing value");
        }

        if (i != ni)
            throw Exception("feature missing value 2");
    }

private:
    /** For reconstituting old classifiers only */
    Perceptron(const boost::shared_ptr<const Feature_Space>
                   & feature_space,
               const Feature & predicted,
               size_t label_count);

};

} // namespace ML

#endif /* __boosting__perceptron_h__ */

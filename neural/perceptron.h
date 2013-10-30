/* perceptron.h                                                    -*- C++ -*-
   Jeremy Barnes, 16 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   The multi layer perceptron algorithm, implemented as a classifier.
*/

#ifndef __boosting__perceptron_h__
#define __boosting__perceptron_h__


#include "jml/boosting/config.h"
#include "jml/boosting/classifier.h"
#include "layer.h"
#include "layer_stack.h"
#include "output_encoder.h"
#include "jml/utils/pair_utils.h"
#include <boost/multi_array.hpp>
#include <boost/shared_ptr.hpp>


namespace ML {


class Label;
class Thread_Context;


/*****************************************************************************/
/* OUTPUT_TRANSFORM                                                          */
/*****************************************************************************/

/** A structure used by the perceptron to transform its output from one range
    to another.  For example, if we wanted to predict movies on a scale of
    1 to 5 with a tanh algorithm (scale -1 to 1), this would scale the final
    output by multiplying by 2 and adding 3.
*/

struct Output_Transform {
    float bias;
    float slope;

    float apply(float input) const;
    float inverse(float input) const;

    distribution<float> apply(const distribution<float> & input) const;
    distribution<float> inverse(const distribution<float> & input) const;

    void apply(distribution<float> & input) const;
    void inverse(distribution<float> & input) const;

    void serialize(DB::Store_Writer & store) const;
    void reconstitute(DB::Store_Reader & store);
};


/*****************************************************************************/
/* PERCEPTRON                                                                */
/*****************************************************************************/

/** The perceptron classifier.
*/

class Perceptron : public Classifier_Impl {
public:
    Perceptron();

    Perceptron(DB::Store_Reader & store,
               const std::shared_ptr<const Feature_Space> & feature_space);
    
    Perceptron(const std::shared_ptr<const Feature_Space> & feature_space,
               const Feature & predicted);

    /** Swap two Perceptron objects.  Guaranteed not to throw an
        exception. */
    void swap(Perceptron & other)
    {
        Classifier_Impl::swap(other);
        features.swap(other.features);
        layers.swap(other.layers);
        output.swap(other.output);
    }

    // Implement copying explicitly to use the deep copies
    Perceptron(const Perceptron & other);
    Perceptron & operator = (const Perceptron & other);

    using Classifier_Impl::predict;

    /** Predict the score for a single class. */
    virtual float predict(int label, const Feature_Set & features,
                          PredictionContext * context = 0) const;
    
    /** Predict the score for all classes. */
    virtual distribution<float>
    predict(const Feature_Set & features,
            PredictionContext * context = 0) const;

    /** Apply the first layer to a dataset to decorrelate it. */
    boost::multi_array<float, 2> decorrelate(const Training_Data & data) const;
        
    virtual std::string print() const;

    virtual std::vector<Feature> all_features() const;

    virtual Output_Encoding output_encoding() const;

    /** Serialization and reconstitution. */
    virtual void serialize(DB::Store_Writer & store) const;
    virtual void
    reconstitute(DB::Store_Reader & store,
                 const std::shared_ptr<const Feature_Space> & features);

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
    
    Layer_Stack<Layer> layers;

    Output_Encoder output;
    
    void add_layer(const std::shared_ptr<Layer> & layer);

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
    Perceptron(const std::shared_ptr<const Feature_Space>
                   & feature_space,
               const Feature & predicted,
               size_t label_count);

};

} // namespace ML

#endif /* __boosting__perceptron_h__ */

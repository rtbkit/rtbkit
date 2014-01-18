/* stump_training_core.h                                           -*- C++ -*-
   Jeremy Barnes, 20 February 2004
   Copyright (c) 2004 Jeremy Barnes.  All rights reserved.
   $Source$

   Core routines of the stump training.  Header file only, since it is
   templated on the parts which can be specialised.
*/

#ifndef __boosting__stump_training_core_h__
#define __boosting__stump_training_core_h__

#include "config.h"
#include <vector>
#include <boost/multi_array.hpp>
#include "training_data.h"
#include "feature_set.h"
#include "jml/arch/exception.h"
#include "jml/utils/vector_utils.h"
#include "jml/utils/pair_utils.h"
#include "stump.h"
#include "stump_training.h"
#include "training_index.h"
#include "jml/utils/guard.h"
#include <boost/bind.hpp>
#include "thread_context.h"

namespace ML {

extern size_t num_real, num_boolean, num_presence, num_categorical;
extern size_t num_bucketed, num_non_bucketed;
extern size_t num_real_early, num_real_not_early;
extern size_t num_bucket_early, num_bucket_not_early;



/** Compare two Z values, and return true if they are equal within a small
    relative tolerance. */
inline bool z_equal(double z1, double z2, double tolerance = 1e-3)
{
    double interval = tolerance * (1.0 - std::max(z1, z2) + 1e-3);
    return std::abs(z1 - z2) < interval;
}

/** Very lightweight array that calculates its offsets much more easily than a
    multi array.  Can speed up some code by four times. */
template<typename T>
struct LW_Array {
    template<typename T2>
    LW_Array(const boost::multi_array<T2, 2> & array)
        : base(array.data()), stride(array.shape()[1]) {}
                                     
    T * base;
    size_t stride;
    
    JML_ALWAYS_INLINE T * operator [] (size_t i) const { return base + i * stride; }
};

/*****************************************************************************/
/* TRACING                                                                   */
/*****************************************************************************/

/** When we want all tracing to turn into a no-op and go away, we use this
    tracer object.
*/

struct No_Trace {
    /** Are we tracing? */
    JML_ALWAYS_INLINE operator bool () const { return false; }

    /** Return the stream to which we trace.

        \param module       The name of the module we are tracing.
        \param level        The verbosity level of the message.
    */
    std::ostream & operator () (const char * module, int level)
    {
        throw Exception("Tracing should never be called from No_Trace");
    }
};

/** When we want to trace to an ostream, we use this object. */
struct Stream_Tracer {
    /** Construct the tracer.

        \param trace        Are we actually tracing?  Allows it to be enabled/
                            disabled at runtime.
        \param stream       Stream to dump the tracing information to.
    */
    Stream_Tracer(bool trace = true, std::ostream & stream = std::cerr,
                  size_t start_message = 0)
        : trace(trace), stream(stream), message(0),
          start_message(start_message)
    {
    }
    
    /** Are we tracing? */
    JML_ALWAYS_INLINE operator bool () const
    {
        return trace && message++ >= start_message;
    }

    /** Return the stream to which we trace.

        \param module       The name of the module we are tracing.
        \param level        The verbosity level of the message.
    */
    std::ostream & operator () (const char * module, int level)
    {
        return std::cerr << message << " " << module << " " << level << ": ";
    }
    
    bool trace;              ///< Is tracing enabled?
    std::ostream & stream;   ///< Which stream do we write to?
    mutable size_t message;  ///< Which message number are we up to?
    size_t start_message;    ///< Which message number do we start with?
};


/*****************************************************************************/
/* STUMP_TRAINER                                                             */
/*****************************************************************************/

/** This class performs the overall training of a decision stumps object.
    It presents candidate (feature, arg, W) tuples to a results object
    (one of the template parameters), which is free to do with them as it
    pleases.

    \param W               The type of the object which holds the weights
                           (broken down by label, predicate value, and
                           correct/incorrect).
*/

template<class T>
struct LW_Array;

/** What is our advance in memory to move from label to label within the
    weights array?  If we have a binary symmetric problem, we will only
    store the weights for label 0, since label 1 will have the same
    value.  In this case, we use an advance of 0 which keeps us pointing
    to the same value.
*/
JML_ALWAYS_INLINE int get_advance(const boost::multi_array<float, 2> & weights)
{
    return (weights.shape()[1] == 1 ? 0 : 1);
}

template<class T>
JML_ALWAYS_INLINE int get_advance(const LW_Array<T> & weights)
{
    return (weights.stride == 1 ? 0 : 1);
}

/** When the weights are like this, it's always regression or one
    dimensional, so the advance doesn't matter. */
JML_ALWAYS_INLINE int get_advance(const std::vector<float> & weights)
{
    return 1;
}

/** Ditto. */
JML_ALWAYS_INLINE int get_advance(const std::vector<const float *> & weights)
{
    return 1;  // assume not binsym
}

template<class W, class Z, class Tracer=No_Trace>
struct Stump_Trainer {
    Stump_Trainer() {}
    
    Stump_Trainer(const Tracer & tracer)
        : tracer(tracer)
    {
    }

    mutable Tracer tracer;  ///< Object to which we trace

    /** This is an object used for example weights which acts as a vector
        of all 1s.  It specifies that each example counts for the same
        amount, without needing to use any memory.
    */
    struct All_Examples {
        float operator [] (int) const { return 1.0f; }
    };

    /** Test all of the given features.  This will iterate over all features
        given and test each of them, accumulating the results of each in the
        results object.

        \param features     List of features to test.
        \param data         Training data to test over.
        \param predicted    Feature we are trying to predict.
        \param weights      Array of weights for each label for each example.
                            For a regression problem, there is one weight
                            per sample.  Must be accessible via the syntax
                            weights[ex][label].
        \param results      Results object into which we accumulate possible
                            split points and their Z values.
    */
    template<class Results, class Weights>
    void test_all(const std::vector<Feature> & features,
                  const Training_Data & data,
                  const Feature & predicted,
                  const Weights & weights,
                  Results & results,
                  int advance = -1) const
    {
        using namespace std;

        if (advance == -1) advance = get_advance(weights);

        /* Pre-calculate the bucket weights for each label. */
        W default_w = calc_default_w(data, predicted, All_Examples(), weights,
                                     advance);
        
        if (tracer) {
            tracer("stump training", 1)
                << "test all: " << features.size() << " features" << endl;
            tracer("stump training", 2)
                << "default w: " << endl
                << default_w.print() << endl;
        }
        
        for (unsigned i = 0;  i < features.size();  ++i)
            test(features[i], data, predicted, weights, All_Examples(),
                 default_w, results, advance);
    }

    /** Test all of the given features.  This will iterate over all features
        given and test each of them, accumulating the results of each in the
        results object.

        An extra parameter is provided which allows us to give the weights
        of each example.  This can either be used to weight training examples,
        or used as a boolean to indicate which examples are to be trained on
        (with a weight of 1) and which are to be ignored.

        \param features     List of features to test.
        \param data         Training data to test over.
        \param predicted    Feature we are trying to predict.
        \param weights      Array of weights for each label for each example.
                            For a regression problem, there is one weight
                            per sample.  Must be accessible via the syntax
                            weights[ex][label].
        \param results      Results object into which we accumulate possible
                            split points and their Z values.
    */
    template<class Results, class Weights>
    void test_all(const std::vector<Feature> & features,
                  const Training_Data & data,
                  const Feature & predicted,
                  const Weights & weights,
                  const distribution<float> & in_class,
                  Results & results,
                  int advance = -1) const
    {
        using namespace std;

        if (advance == -1) advance = get_advance(weights);

        W default_w = calc_default_w(data, predicted, in_class, weights,
                                     advance);

        if (tracer) {
            tracer("stump training", 1)
                << "test all: " << features.size() << " features" << endl;
            tracer("stump training", 2)
                << "default w: " << endl
                << default_w.print() << endl;
        }
        
        for (unsigned i = 0;  i < features.size();  ++i)
            test(features[i], data, predicted, weights, in_class, default_w,
                 results, advance);
    }

    template<class Results, class Weights>
    struct Test_Feature_Job {
        const Stump_Trainer * parent;
        Feature feature;
        const Training_Data & data;
        const Feature & predicted;
        const Weights & weights;
        const distribution<float> & in_class;
        Results & results;
        int advance;
        const W & default_w;

        Test_Feature_Job(const Stump_Trainer * parent,
                         const Feature & feature,
                         const Training_Data & data,
                         const Feature & predicted,
                         const Weights & weights,
                         const distribution<float> & in_class,
                         const W & default_w,
                         Results & results,
                         int advance)
            : parent(parent), feature(feature), data(data),
              predicted(predicted),
              weights(weights), in_class(in_class), results(results),
              advance(advance), default_w(default_w)
        {
        }

        void operator () ()
        {
            parent->test(feature, data, predicted, weights, in_class,
                         default_w, results, advance);
        }
    };

    /** Like test_all, but in parallel using the worker task. */
    template<class Results, class Weights>
    void test_all(Thread_Context & context,
                  const std::vector<Feature> & features,
                  const Training_Data & data,
                  const Feature & predicted,
                  const Weights & weights,
                  const distribution<float> & in_class,
                  Results & results,
                  int advance = -1) const
    {
        using namespace std;

        if (advance == -1) advance = get_advance(weights);

        W default_w = calc_default_w(data, predicted, in_class, weights,
                                     advance);

        if (tracer) {
            tracer("stump training", 1)
                << "test all: " << features.size() << " features" << endl;
            tracer("stump training", 2)
                << "default w: " << endl
                << default_w.print() << endl;
        }

        Worker_Task & worker = context.worker();

        int group = worker.get_group(NO_JOB,
                                     "test all group",
                                     context.group());
        {
            Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                         boost::ref(worker),
                                         group));
            
            for (unsigned i = 0;  i < features.size();  ++i)
                worker.add(Test_Feature_Job<Results, Weights>
                           (this, features[i], data, predicted,
                            weights, in_class, default_w,
                            results, advance),
                           "test feature job",
                           group);
        }

        worker.run_until_finished(group);
    }

    /* Test all of the given features, and return them sorted by their best
       Z score. */
    template<class Results, class Weights>
    void test_all_and_sort(std::vector<Feature> & features,
                           const Training_Data & data,
                           const Feature & predicted,
                           const Weights & weights,
                           Results & results,
                           int advance = -1) const
    {
        using namespace std;

        if (advance == -1) advance = get_advance(weights);

        W default_w = calc_default_w(data, predicted, All_Examples(), weights,
                                     advance);

        if (tracer) {
            tracer("stump training", 1)
                << "test all: " << features.size() << " features" << endl;
            tracer("stump training", 2)
                << "default w: " << endl
                << default_w.print() << endl;
        }
        
        std::vector<std::pair<int, float> > feature_scores;
        feature_scores.reserve(features.size());
        
        for (unsigned i = 0;  i < features.size();  ++i) {
            float z = test(features[i], data, predicted, weights, All_Examples(),
                           default_w, results, advance);
            feature_scores.push_back(std::make_pair(i, z));
        }

        sort_on_second_ascending(feature_scores);
        std::vector<Feature> new_features;
        new_features.reserve(features.size());

        for (unsigned i = 0;  i < features.size();  ++i)
            new_features.push_back(features[feature_scores[i].first]);
        
        features.swap(new_features);
    }

    /* Test all of the given features, and return them sorted by their best
       Z score. */
    template<class Results, class Weights>
    void test_all_and_sort(std::vector<Feature> & features,
                           const Training_Data & data,
                           const Feature & predicted,
                           const Weights & weights,
                           const distribution<float> & in_class,
                           Results & results, int advance = -1) const
    {
        using namespace std;

        if (advance == -1) advance = get_advance(weights);

        W default_w = calc_default_w(data, predicted, in_class, weights, advance);

        if (tracer) {
            tracer("stump training", 1)
                << "test all: " << features.size() << " features" << endl;
            tracer("stump training", 2)
                << "default w: " << endl
                << default_w.print() << endl;
        }
        
        std::vector<std::pair<int, float> > feature_scores;
        feature_scores.reserve(features.size());
        
        for (unsigned i = 0;  i < features.size();  ++i) {
            float z = test(features[i], data, predicted, weights, in_class,
                           default_w, results);
            //cerr << " feat " << features[i] << " z " << z << endl;
            if (z != Z::none) feature_scores.push_back(std::make_pair(i, z));
        }
        
        sort_on_second_ascending(feature_scores);
        std::vector<Feature> new_features;
        new_features.reserve(feature_scores.size());
        
        for (unsigned i = 0;  i < feature_scores.size();  ++i)
            new_features.push_back(features[feature_scores[i].first]);
        
        features.swap(new_features);
    }
    
    /** Test the single given feature, calling the appropriate routine for the
        type of feature.

        \param feature      Feature we are testing.
        \param data         Training data to test over.
        \param predicted    Feature we are trying to predict.
        \param weights      Array of weights for each label for each example.
                            For a regression problem, there is one weight
                            per sample.  Must be accessible via the syntax
                            weights[ex][label].
        \param ex_weights   Array of weights for each sample.  The weights
                            must be accessible via the syntax ex_weights[ex].
        \param default_w    The starting W value.  Passed in as it can be
                            calculated once and used for each feature.  
                            See calc_default_w.
        \param results      Results object into which we accumulate possible
                            split points and their Z values.

        \returns            The Z value of the best split point.
    */
    template<class Results, class ExampleWeights, class Weights>
    float test(const Feature & feature,
               const Training_Data & data,
               const Feature & predicted,
               const Weights & weights,
               const ExampleWeights & ex_weights,
               const W & default_w, Results & results,
               int advance = -1) const
    {
        using namespace std;

        if (advance == -1) advance = get_advance(weights);

        /* Don't predict the label with the label! */
        if (feature == predicted) return Z::worst;

        std::shared_ptr<const Feature_Space> fs = data.feature_space();

        if (tracer)
            tracer("stump training", 1)
                << "testing feature " << fs->print(feature)
                << "(" << feature << ") info "
                << fs->info(feature) << endl;

        Feature_Info info = fs->info(feature);
        
        switch (info.type()) {
        case PRESENCE:
            return test_presence(feature, data, predicted, weights, ex_weights,
                                 default_w, results, advance);
            
        case BOOLEAN:
            return test_boolean(feature, data, predicted, weights, ex_weights,
                                default_w, results, advance);

        case CATEGORICAL:
        case STRING:
            return test_categorical(feature, data, predicted, weights,
                                    ex_weights, default_w, results, advance);

        case REAL:
            return test_real(feature, data, predicted, weights, ex_weights,
                             default_w, results, advance);
            
        case INUTILE: {
            double missing;
            if (!results.start(feature, default_w, missing))
                return Z::worst;
            float z = results.add(feature, default_w, -INFINITY, missing);
            results.finish(feature);
            return z;
        }

        default:
            throw Exception("Unknown feature info type " + info.print()
                            + " in Stump::test");
        }
    }

    /** Calculate the default weights of the buckets.  This will accumulate
        the example weights into the MISSING bucket for each label.  This
        is the starting point for all of the test_* routines; we only do it
        once.

        \param data         Training data to test over.
        \param predicted    Feature we are trying to predict.
        \param ex_weights   Array of weights for each sample.  The weights
                            must be accessible via the syntax ex_weights[ex].
                            Normally, this will be either a
                            distribution<float> or All_Examples object.
        \param weights      Array of weights for each label for each example.
                            For a regression problem, there is one weight
                            per sample.  Must be accessible via the syntax
                            weights[ex][label].  Normally this will be a
                            boost::multi_array<float, 2>.

        \returns            The W object with all weight in the MISSING
                            buckets, distributed according to weights and
                            ex_weights.
    */
    template<class Weights, class ExampleWeights>
    W calc_default_w(const Training_Data & data,
                     const Feature & predicted,
                     const ExampleWeights & ex_weights,
                     const Weights & weights,
                     int advance = -1) const
    {
        if (advance == -1) advance = get_advance(weights);

        int nl = data.label_count(predicted);
        W result(nl);

        const std::vector<Label> & labels = data.index().labels(predicted);
        
        for (unsigned i = 0;  i < data.example_count();  ++i) {
            if (ex_weights[i] == 0.0) continue;
            int correct_label = labels[i];
#if 0
            using namespace std;
            cerr << "default W: example " << i << " label " << correct_label
                 << " weights " << ex_weights[i] << " " << weights[i][0]
                 << endl;
#endif
            result.add(correct_label, MISSING, ex_weights[i], &weights[i][0],
                       advance);
        }
        
        //using namespace std;
        //cerr << "default w: " << endl
        //     << result.print() << endl;

        return result;
    }

    /** Adjust the weight for a given feature before testing split points.
        This function transfers weight from the MISSING bucket to the
        def bucket (specified) for all examples where the given feature is
        not missing.  This is done before testing the split points for each
        feature.

        \param W            W value to modify.  This will probably be the
                            return from default_w.  It is modified in place.
        \param data         Training data we are training over.
        \param predicted    Feature we are trying to predict.
        \param weights      Array of weights for each label for each example.
                            For a regression problem, there is one weight
                            per sample.  Must be accessible via the syntax
                            weights[ex][label].  Normally this will be a
                            boost::multi_array<float, 2>.
        \param ex_weights   Array of weights for each sample.  The weights
                            must be accessible via the syntax ex_weights[ex].
                            Normally, this will be either a
                            distribution<float> or All_Examples object.
        \param feature      Feature we are adjusting for (that we are about
                            to test).
        \param def          Bucket to transfer weight to for non-missing
                            examples.  Weight is always transfered from the
                            MISSING bucket.

        \returns            Number of non-missing examples that contain this
                            feature in the training data.
    */
    template<class Weights, class ExampleWeights>
    int adjust_w(W & w,
                 const Training_Data & data,
                 const Feature & predicted,
                 const Weights & weights,
                 const ExampleWeights & ex_weights,
                 const Feature & feature, bool def, int advance) const
    {
        using namespace std;
        
        /* Fix it up for the ones where the feature occurs. */
        Joint_Index index
            = data.index().joint(predicted, feature, BY_EXAMPLE,
                                 IC_EXAMPLE | IC_LABEL | IC_DIVISOR);
        
#if 0
        if (tracer)
            tracer("adjust_w", 3)
                << " feature_data.exactly_one = " << feature_index.exactly_one()
                << " feature_data.dense() = " << feature_index.dense()
                << " feature_data.size() = " << examples.size()
                << " data.example_count() = " << data.example_count()
                << endl;
#endif
        
        unsigned i = 0;
        int result = 0;

        for (;  i < index.size();  ++i) {
            if (index[i].missing()) continue;

            int example = index[i].example();
            if (ex_weights[example] == 0.0) continue;

            int label = index[i].label();

            /* If we had the same feature multiple times in the dataset, then
               we need to spread its weight out over all of them (otherwise
               the one feature will have too much weight). */
            double divisor = ex_weights[example] * index[i].divisor();

            /* Transfer (weighted) the weight from the MISSING bucket to the
               def bucket for this label. */
            w.transfer(label, MISSING, def, divisor, &weights[example][0],
                       advance);
            ++result;
        }
        
        /* Compensate for any accumulated rounding errors. */
        w.clip(MISSING);
        
        return result;
    }

    /** Test a boolean variable. */
    template<class Results, class Weights, class ExampleWeights>
    float test_boolean(const Feature & feature,
                       const Training_Data & data,
                       const Feature & predicted,
                       const Weights & weights,
                       const ExampleWeights & ex_weights,
                       const W & default_w, Results & results, int advance) const
    {
        using namespace std;

        ++num_boolean;

        /* See if we can do it by buckets.  We only do so if more than 20% of
           the examples include this feature (otherwise it will probably be
           slower).
        */
        if (data.index().density(feature) > 0.2)
            return test_buckets(feature, data, predicted, weights, ex_weights,
                                default_w, results, 2,
                                false /* categorical; false since doesn't
                                         matter and faster if false */,
                                advance);

        /* Fix it up for the ones where the feature occurs. */
        Joint_Index index
            = data.index().joint(predicted, feature, BY_EXAMPLE,
                                 IC_EXAMPLE | IC_LABEL | IC_DIVISOR | IC_VALUE);

        if (index.empty()) return Z::none;
        

        ++num_non_bucketed;

        W w(default_w);
        
        unsigned i = 0;
        for (;  i < index.size();  ++i) {
            if (index[i].missing()) continue;
            int example = index[i].example();
            if (ex_weights[example] == 0.0) continue;
            int label = index[i].label();
            bool bucket = index[i].value() > 0.5;

            /* If we had the same feature multiple times in the dataset, then
               we need to spread its weight out over all of them (otherwise
               the one feature will have too much weight). */
            double divisor = ex_weights[example] * index[i].divisor();
            
            w.transfer(label, MISSING, bucket, divisor, &weights[example][0],
                       advance);
        }
        
        /* Compensate for any accumulated rounding errors. */
        w.clip(MISSING);

        double missing;
        if (!results.start(feature, w, missing)) return Z::worst;
        float Z = results.add(feature, w, 0.5, missing);
        results.finish(feature);
        
        return Z;
    }

    /** Test a presence variable. */
    template<class Results, class ExampleWeights, class Weights>
    float test_presence(const Feature & feature,
                        const Training_Data & data,
                        const Feature & predicted,
                        const Weights & weights,
                        ExampleWeights ex_weights,
                        const W & default_w, Results & results,
                        int advance) const
    {
        using namespace std;

        ++num_presence;

        W w(default_w);
        int ex = adjust_w(w, data, predicted, weights, ex_weights, feature,
                          true, advance);
        if (ex == 0) return Z::none; // no examples
        
        if (tracer)
            tracer("test_presence", 1)
                << "w = " << endl << w.print() << endl;
        
        double missing;

        if (!results.start(feature, w, missing)) return Z::worst;
        float Z = results.add_presence(feature, w, 0.5, missing);
        results.finish(feature);
        
        return Z;
    }

    template<class Results, class Weights, class ExampleWeights>
    float test_categorical(const Feature & feature,
                           const Training_Data & data,
                           const Feature & predicted,
                           const Weights & weights,
                           const ExampleWeights & ex_weights,
                           const W & default_w, Results & results,
                           int advance) const
    {
        ++num_categorical;
        
        /* See if we can do it by buckets.  We only do so if more than 20% of
           the examples include this feature (otherwise it will probably be
           slower).
        */
        //if (data.index().density(feature) > 0.2)
        //    return test_buckets(feature, data, predicted, weights, ex_weights,
        //                        default_w, results,
        //                        255 /* num_buckets; TODO: configurable */,
        //                        true /* categorical */, advance);

        Joint_Index index
            = data.index().joint(predicted, feature, BY_VALUE,
                                 IC_EXAMPLE | IC_LABEL | IC_EXAMPLE
                                 | IC_DIVISOR);
        
        if (index.empty()) return Z::none;
        
        W w(default_w);
        int ex = adjust_w(w, data, predicted, weights, ex_weights, feature,
                          false, advance);
        
        if (ex == 0) return Z::none;

        double missing;

        if (!results.start(feature, w, missing)) return Z::worst;
        
        /* Save this W value so we can get back to it after each value. */
        W w_start(w);
        
        bool debug = false;

        int i = 0;
        /* Skim off any missing ones from the start. */
        while (i < index.size() && index[i].missing()) ++i;
        
        using namespace std;

        if (debug) {
            cerr << " feature "
                 << data.feature_space()->print(feature)
                 << endl;
            cerr << "i = " << i << " of " << index.size() << endl;
        }

        /* One candidate split point is -INF, which lets us split only based
           upon missing or not. */
        float Z = Z::worst;

        if (i != 0) {
            Z = results.add(feature, w, -INFINITY, missing);
        
            if (debug)
                cerr << "added split " << -INFINITY << " with " << missing
                     << " missing and score " << Z << endl;
        }

        float prev = index[i].value();
        
        while (i < index.size()) {
            
            int nex = 0;
            /* Look for a unique split point. */
            while (i < index.size() && index[i].value() == prev) {
                int example = index[i].example();

                if (ex_weights[example] == 0.0) { ++i; continue; }

                int label = index[i].label();
                float divisor = ex_weights[example] * index[i].divisor();
                
                /* Transfer weight from predicate not holding to predicate 
                   holding. */
                w.transfer(label, false, true, divisor, &weights[example][0],
                           advance);
                
                ++i;  ++nex;
            }
            
            /* Fix up any rounding errors that took it below zero. */
            w.clip(false);
            
            /* Add this split point. */
            float arg = prev;
            float new_Z = results.add(feature, w, arg, missing);
            Z = std::min(Z, new_Z);
            
            if (debug && new_Z == Z) {
                cerr << "i = " << i << endl;
                cerr << "added split "
                     << data.feature_space()->print(feature, arg)
                     << " with " << missing
                     << " missing and score " << new_Z
                     << (new_Z == Z ? " *** BEST ***" : "")
                     << endl;

                cerr << "nex = " << nex << endl;
                cerr << "w_start = " << w_start.print() << endl;
                cerr << "W = " << w.print() << endl;

            }
            
            /* Reset back to old values for next value */
            w = w_start;
            
            if (i < index.size()) prev = index[i].value();
        }
        
        results.finish(feature);

        return Z;
    }

    template<class Results, class Weights, class ExampleWeights>
    float test_real(const Feature & feature,
                    const Training_Data & data,
                    const Feature & predicted,
                    const Weights & weights,
                    const ExampleWeights & ex_weights,
                    const W & default_w,
                    Results & results,
                    int advance) const
    {
        using namespace std;
        //cerr << "test_real" << endl;

        /* See if we can do it by buckets.  We only do so if more than 20% of
           the examples include this feature (otherwise it will probably be
           slower).
        */

        if (data.index().density(feature) > 0.2)
            return test_buckets(feature, data, predicted, weights, ex_weights,
                                default_w, results,
                                255 /* num_buckets; TODO: configurable */,
                                false /* categorical */,
                                advance);

        ++num_real;

        bool debug = false;
        //debug = (data.feature_space()->print(feature) == "language|all_diff_prob_lb");
        //debug = (feature.type() == 10);

        using namespace std;

        if (debug) {
            cerr << "feature " << data.feature_space()->print(feature)
                 << endl;
        }

        ++num_non_bucketed;
        ++num_real_early;

        Joint_Index index
            = data.index().joint(predicted, feature, BY_VALUE,
                                 IC_VALUE | IC_LABEL | IC_EXAMPLE | IC_DIVISOR);

        if (index.empty()) return Z::none;
        
        W w(default_w);
        int ex = adjust_w(w, data, predicted, weights, ex_weights, feature,
                          true, advance);

        if (ex == 0) return Z::none;

        if (debug)
            cerr << "ex = " << ex << " adjusted w " << endl << w.print() << endl;
        //cerr << "adjusted w = " << endl << w.print() << endl;

        double missing;
        if (!results.start(feature, w, missing)) return Z::worst;

        --num_real_early;
        ++num_real_not_early;
        
        int i = 0;
        /* Skim off any missing ones from the start. */
        while (i < index.size() && index[i].missing()) ++i;
        
        if (debug) {
            cerr << "i = " << i << " of " << index.size() << endl;
        }

        // TODO: not missing
        float Z = Z::worst;
#if 0
        /* One candidate split point is -INF, which lets us split only based
           upon missing or not. */
        float Z = results.add(feature, w, -INFINITY, missing);
#endif
        
        float prev = index[i].value();
        float max_value = index.back().value();
        
        while (i < index.size() && index[i].value() < max_value) {
            
            /* Look for a unique split point. */
            for (; i < index.size() && index[i].value() < max_value
                     && index[i].value() == prev;  ++i) {
                
                int example = index[i].example();
                if (ex_weights[example] == 0.0) continue;
                int label = index[i].label();
                float divisor = ex_weights[example] * index[i].divisor();
                
                /* Transfer weight from predicate not holding to predicate 
                   holding. */
                w.transfer(label, true, false, divisor, &weights[example][0],
                           advance);
            }
            
            /* Fix up any rounding errors that took it below zero. */
            w.clip(true);

            /* Add this split point. */
            float arg = (index[i].value() + prev) * 0.5;
            
            if (arg == prev || arg == index[i].value()) {
                arg = index[i].value();
#if 0 // TODO: should be equal to lower or highest?
                cerr << "feature: " << data.feature_space()->print(feature)
                     << endl;
                cerr << "arg: " << format("arg: %.10f (0x%08x) "
                                          "prev: %.10f (0x%08x) "
                                          "val: %.10f (0x%08x)",
                                          arg, reinterpret_as_int(arg),
                                          prev, reinterpret_as_int(prev),
                                          index[i].value(),
                                          reinterpret_as_int(index[i].value()))
                     << endl;
                cerr << "density: " << data.index().density(feature) << endl;
                cerr << "examples: " << data.example_count() << endl;
                throw Exception("adjacent floats");
#endif // equal to lower or highest?
            }

            float new_Z = results.add(feature, w, arg, missing);
            if (debug) {
                int i1 = int_float(prev).i;
                int i2 = int_float(index[i].value()).i;

                int dist = i2 - i1;

                cerr << "arg = " << arg << "  Z = " << new_Z
                     << "  w = " << endl
                     << w.print() << endl;
                cerr << format("prev = %f (%0x8d) curr = %f (%0x8d) "
                               "dist = %d ulps",
                               prev, i1, index[i].value(), i2, dist)
                     << endl;
            }
            Z = std::min(Z, new_Z);
            
            prev = index[i].value();
        }
        
        results.finish(feature);

        return Z;
    }

    template<class Results, class Weights, class ExampleWeights>
    float test_buckets(const Feature & feature,
                       const Training_Data & data,
                       const Feature & predicted,
                       const Weights & weights,
                       const ExampleWeights & ex_weights,
                       const W & default_w,
                       Results & results,
                       int num_buckets,
                       bool categorical,
                       int advance) const
    {
        ++num_bucketed;
        using namespace std;

        bool debug = false;
        //debug = (feature.type() == 10);

        Joint_Index index
            = data.index().joint(predicted, feature, BY_EXAMPLE,
                                 IC_LABEL | IC_EXAMPLE | IC_BUCKET | IC_DIVISOR,
                                 num_buckets);

        W w_empty(default_w.nl());
        int nb = index.bucket_count();

        std::vector<W> buckets(nb, w_empty);

        W w = default_w;

        int nl JML_UNUSED = default_w.nl();

        if (debug) {
            cerr << "feature " << data.feature_space()->print(feature)
                 << endl;
        }

        for (unsigned i = 0;  i < index.size();  ++i) {
            int example = index[i].example();

            if (ex_weights[example] == 0.0) continue;
            int label = index[i].label();
            
            double divisor = ex_weights[example] * index[i].divisor();
            
            int bucket = index[i].bucket();

            //cerr << "i " << i << " example " << example << " label "
            //     << label << " bucket " << bucket << endl;

            buckets[bucket].add(label, true, divisor, &weights[example][0],
                                advance);
            w.transfer(label, MISSING, true, divisor, &weights[example][0],
                       advance);
        }

        /* Compensate for any accumulated rounding errors. */
        w.clip(MISSING);

        ++num_bucket_early;

        double missing;
        if (!results.start(feature, w, missing)) return Z::worst;

        --num_bucket_early;
        ++num_bucket_not_early;

        /* We need at least 2 buckets or one bucket and some missing values
           in order to make a split. */
        if (nb + (missing > 0.0) < 2) return Z::none;

        /* One candidate split point is -INF, which lets us split only based
           upon missing or not. */
        float Z = Z::worst;

        if (missing > 0.0)
            results.add(feature, w, -INFINITY, missing);
        float best_arg = -INFINITY;

        if (debug) {
            cerr << "missing = " << missing << endl;
            cerr << "nb = " << nb << endl;
            cerr << "added default split " << -INFINITY << " with "
                 << missing << " missing and score " << Z
                 << endl;
        }

        W w_start = w;  // only necessary for categorical

        /* Go through the buckets and select the best one. */
        for (int i = 0;  i < nb - 1;  ++i) {

            if (debug) {
                cerr << "before: " << endl << w.print() << endl;
                cerr << "bucket contents: " << endl
                     << buckets[i].print() << endl;
            }
            /* Transfer the whole bucket. */
            w.transfer(true, false, buckets[i]);

            if (debug) {
                cerr << "after: " << endl << w.print() << endl;
            }

            /* Fix up any rounding errors that took it below zero. */
            w.clip(true);
            
            if (debug) {
                cerr << "after clip: " << endl << w.print() << endl;
            }

            /* Add this split point. */
            float arg = index.bucket_vals()[i];
            float new_Z = results.add(feature, w, arg, missing);

            if (categorical) w = w_start;
            
            if (debug) {
                cerr << "i = " << i << endl;
                cerr << "added split " << arg << " with " << missing
                     << " missing and score " << new_Z
                     << (new_Z < Z ? " *** BEST ***" : "")
                     << endl;
                if (new_Z < Z) best_arg = arg;
            }
            
            Z = std::min(Z, new_Z);

        }
        
        if (debug && !finite(best_arg)) {
            cerr << "*** best_arg non finite" << endl;
        }

        results.finish(feature);
        return Z;
    }
};


} // namespace ML


#endif /* __boosting__stump_training_core_h__ */

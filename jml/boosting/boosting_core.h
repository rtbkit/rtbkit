/* boosting_core.h                                                 -*- C++ -*-
   Jeremy Barnes, 4 March 2004
   Copyright (c) 2004 Jeremy Barnes.  All rights reserved.
   $Source$

   This is the core of the boosting library.  It contains the functions
   needed to do the updating and reweighting.
*/

#ifndef __boosting__boosting_core_h__
#define __boosting__boosting_core_h__


#include "jml/compiler/compiler.h"
#include "training_data.h"
#include "stump.h"
#include "training_index.h"
#include "evaluation.h"
#include "stump_predict.h"

namespace ML {

/*****************************************************************************/
/* LOSS FUNCTIONS                                                            */
/*****************************************************************************/

/** Each of these objects calculate the weight associated with a single
    prediction, according to the given loss function.
*/

/** The boosting loss function.  It is exponential in the margin. */
struct Boosting_Loss {
    JML_ALWAYS_INLINE
    float operator () (int label, int corr, float pred, float current) const
    {
        int correct = (corr == label);
        pred -= 2.0 * pred * correct;
        //*((int *)&pred) ^= (correct << 31);  // flip sign bit if correct
        return current * exp(pred);
    }
};

/** The logistic boost loss function.  Requires the z function from the boosting
    loss to work; fortunately this is provided by the stumps. */
struct Logistic_Loss {
    Logistic_Loss(double z) : z(z) {}
    
    JML_ALWAYS_INLINE
    float operator () (int label, int corr, float pred, float current) const
    {
        int correct = (corr == label);
        pred -= 2.0 * pred * correct;
        //*((int *)&pred) ^= (correct << 31);  // flip sign bit if correct
        float qz = exp(pred);
        return 1.0 / ((z/qz * ((1.0 / current) - 1.0)) - 1.0);
    }
    
    double z;
};

/** A "loss" function used to update a set of prediction weights for some
    training data. */
struct Boosting_Predict {
    JML_ALWAYS_INLINE
    float operator () (int label, int corr, float pred, float current) const
    {
        return current + pred;
    }
};


/*****************************************************************************/
/* WEIGHTS UPDATERS                                                          */
/*****************************************************************************/

/** These objects update a row of weights at a time.  They are separate
    objects as it is possible to speed it up significantly (twice as fast)
    in the case of a binary symmetric classifier (which gives x and -x, or
    equivalently x and 1-x for its two outputs).
*/

template<class Fn>
struct Binsym_Updater {
    Binsym_Updater(const Fn & fn = Fn()) : fn(fn) {}
    Fn fn;

    template<class FeatureIt, class WeightIt>
    float operator () (const Stump & stump,
                       const Optimization_Info & opt_info,
                       float cl_weight, int corr,
                       FeatureIt ex_start, FeatureIt ex_range,
                       WeightIt weight_begin, int advance) const
    {
        float pred = stump.predict(0, ex_start, ex_range) * cl_weight;

        //cerr << "    cl_weights = " << cl_weight << " corr = " << corr << endl;
        //cerr << "    pred = { " << pred << " " << -pred << " }" << endl;
        //cerr << "    weight_begin = " << *weight_begin << endl;

        *weight_begin = fn(0, corr, pred, *weight_begin);
        return *weight_begin * 2.0;
    }

    template<class WeightIt>
    float operator () (const Classifier_Impl & classifier,
                       const Optimization_Info & opt_info,
                       float cl_weight, int corr,
                       const Feature_Set & features,
                       WeightIt weight_begin,
                       int advance) const
    {
        float pred = classifier.predict(0, features, opt_info) * cl_weight;
        *weight_begin = fn(0, corr, pred, *weight_begin);
        return *weight_begin * 2.0;
    }

    template<class PredIt, class WeightIt>
    float operator () (PredIt pred_it, PredIt pred_end, float cl_weight,
                       int corr,
                       WeightIt weight_begin, int advance) const
    {
        *weight_begin = fn(0, corr, (*pred_it) * cl_weight, *weight_begin);
        return *weight_begin * 2.0;
    }
}; 

template<class Fn>
struct Normal_Updater {
    Normal_Updater(size_t nl, const Fn & fn = Fn())
        : fn(fn), nl(nl) {}
    Fn fn;
    size_t nl;

    /* Make a prediction and return an update. */
    template<class FeatureIt, class WeightIt>
    float operator () (const Stump & stump,
                       const Optimization_Info & opt_info,
                       float cl_weight, int corr,
                       FeatureIt ex_start, FeatureIt ex_range,
                       WeightIt weight_begin, int advance) const
    {
        distribution<float> pred(nl);
        stump.predict(pred, ex_start, ex_range);
        if (cl_weight != 1.0) pred *= cl_weight;

        //cerr << "    pred = " << pred << endl;
        
        return operator () (pred.begin(), pred.end(), corr, weight_begin,
                            advance);
    }

    template<class WeightIt>
    float operator () (const Classifier_Impl & classifier,
                       const Optimization_Info & opt_info,
                       float cl_weight, int corr,
                       const Feature_Set & features,
                       WeightIt weight_begin,
                       int advance) const
    {
        distribution<float> pred = classifier.predict(features, opt_info);
        if (cl_weight != 1.0) pred *= cl_weight;

        return operator () (pred.begin(), pred.end(), corr, weight_begin,
                            advance);
    }
    
    /* Perform an update once the predictions are already known. */
    template<class PredIt, class WeightIt>
    float operator () (PredIt pred_it, PredIt pred_end, int corr,
                       WeightIt weight_begin, int advance) const
    {
        float total = 0.0;
        
        if (advance) {
            for (unsigned l = 0;  l < nl;  ++l) {
                *weight_begin = fn(l, corr, pred_it[l], *weight_begin);
                total += *weight_begin;
                weight_begin += advance;
            }
        }
        else
            total = Binsym_Updater<Fn>(fn)(pred_it, pred_end, 1.0, corr,
                                           weight_begin, 0);
        
        return total;
    }
};


/** Small object used to find an example number in an index. */
struct Find_Example {
    bool operator () (const Index_Iterator & it, int what) const
    {
        return it.example() < what;
    }
};
        


/*****************************************************************************/
/* UPDATE_WEIGHTS                                                            */
/*****************************************************************************/

/** This class updates a weights matrix in response to a new stump having
    been learned.
*/

template<class Updater>
struct Update_Weights {
    Update_Weights(const Updater & updater = Updater()) : updater(updater) {}
    Updater updater;

    /** Apply the given stump to the given weights, given the training
        data.  This will update all of the weights. */
    float operator () (const Stump & stump,
                       const Optimization_Info & opt_info,
                       float cl_weight,
                       boost::multi_array<float, 2> & weights,
                       const Training_Data & data,
                       int start_x = 0, int end_x = -1) const
    {
        if (end_x == -1) end_x = data.example_count();

        Joint_Index index
            = data.index().joint(stump.predicted(), stump.split.feature(),
                                 BY_EXAMPLE,
                                 IC_VALUE | IC_EXAMPLE);
        
        double total = 0.0;

        Index_Iterator ex_start
            = (start_x == 0
               ? index.begin()
               : std::lower_bound(index.begin(), index.end(), start_x,
                                  Find_Example()));
        
        Index_Iterator ex_end   = index.end();

        const std::vector<Label> & labels
            = data.index().labels(stump.predicted());

        int advance = (weights.shape()[1] == 1 ? 0 : 1);

        for (unsigned x = start_x;  x < end_x;  ++x) {

            /* Find the number of examples that we have. */
            Index_Iterator ex_range = ex_start;
            while (ex_range != ex_end && ex_range->example() == x)
                ++ex_range;

            float t = updater(stump, opt_info, cl_weight, labels[x],
                              ex_start, ex_range,
                              &weights[x][0], advance);
            total += t;

            __builtin_prefetch(&weights[x][0] + 48, 1, 3);
            ex_start = ex_range;
        }
        
        return total;
    }

    /** Apply the given population of stumps with the given classifier weight
        distribution to the given sample weights, given the training data.  */
    float operator () (const std::vector<Stump> & stumps,
                       const std::vector<Optimization_Info> & opt_infos,
                       const std::vector<float> & cl_weights,
                       boost::multi_array<float, 2> & sample_weights,
                       const Training_Data & data) const
    {
        double total = 0.0;
        for (unsigned i = 0;  i < stumps.size();  ++i) {
            total = operator () (stumps[i], opt_infos[i],
                                 cl_weights.at(i), sample_weights,
                                 data);
            //cerr << "after stump " << i << ": total = " << total << endl;
        }
        return total;
    }

    /** Apply the given classifier to the given weights, given the training
        data.  This will update all of the weights. */
    float operator () (const Classifier_Impl & classifier,
                       const Optimization_Info & opt_info,
                       float cl_weight,
                       boost::multi_array<float, 2> & weights,
                       const Training_Data & data,
                       int start_x = 0, int end_x = -1) const
    {
        if (end_x == -1) end_x = data.example_count();

        double total = 0.0;

        const std::vector<Label> & labels
            = data.index().labels(classifier.predicted());
        
        int advance = (weights.shape()[1] == 1 ? 0 : 1);

        using namespace std;
        //map<float, int> output;

        for (unsigned x = start_x;  x < end_x;  ++x) {
            //float val = classifier.predict(0, data[x]);
            //output[val] += 1;
            
            float t = updater(classifier, opt_info, cl_weight, labels[x],
                              data[x], &weights[x][0], advance);
            total += t;

            __builtin_prefetch(&weights[x][0] + 48, 1, 3);
        }

#if 0
        for (map<float, int>::const_iterator it = output.begin(),
                 end = output.end();  it != end;  ++it)
            cerr << "  value " << it->first << " seen " << it->second
                 << " times" << endl;
        
        cerr << "updater: total = " << total << endl;
#endif

        return total;
    }
};


/*****************************************************************************/
/* SCORERS                                                                   */
/*****************************************************************************/

/* Simple class that tells us if we got it right or not for the binary
   symmetric case. */

struct Binsym_Scorer {
    template<class WeightIterator>
    JML_ALWAYS_INLINE
    float
    operator () (int label, WeightIterator begin, WeightIterator end) const
    {
        float pred = *begin;
        pred -= 2 * label * pred;
        // 1/2 a point for being zero, another 1/2 for being greater
        float result = 0.5f * ((int)(pred >= -0.5e-6f) + (int)(pred >= 0.5e-6f));
        //cerr << "label = " << label << " pred = " << pred << " *begin = "
        //     << *begin << " result = " << result << endl;
        return result;
    }
};

struct Normal_Scorer {
    template<class WeightIterator>
    JML_ALWAYS_INLINE
    float
    operator () (int label, WeightIterator begin, WeightIterator end) const
    {
        return correctness(begin, end, label).correct;
    }
};


/*****************************************************************************/
/* UPDATE_WEIGHTS_AND_SCORES                                                 */
/*****************************************************************************/

/** This class updates a weights matrix in response to a new stump having
    been learned, and updates a scores matrix based on the output of the
    classifier, as well as calculating the accuracy of the classifier.
*/

template<class Weights_Updater, class Output_Updater, class Scorer>
struct Update_Weights_And_Scores {
    Update_Weights_And_Scores(const Weights_Updater & weights_updater
                                  = Weights_Updater(),
                              const Output_Updater & output_updater
                                  = Output_Updater())
        : weights_updater(weights_updater), output_updater(output_updater)
    {
    }

    Weights_Updater weights_updater;
    Output_Updater output_updater;
    Scorer scorer;

    /** Apply the given stump to the given weights, given the training
        data.  This will update all of the weights. */
    float operator () (const Stump & stump,
                       const Optimization_Info & opt_info,
                       float cl_weight,
                       boost::multi_array<float, 2> & weights,
                       boost::multi_array<float, 2> & output,
                       const Training_Data & data,
                       const distribution<float> & example_weights,
                       double & accuracy,
                       int start_x = 0, int end_x = -1) const
    {
        if (end_x == -1) end_x = data.example_count();

        size_t nl = output.shape()[1];

        Joint_Index index
            = data.index().joint(stump.predicted(),
                                 stump.split.feature(),
                                 BY_EXAMPLE,
                                 IC_VALUE | IC_EXAMPLE);

        double total = 0.0;

        const std::vector<Label> & labels
            = data.index().labels(stump.predicted());

        int advance = (weights.shape()[1] == 1 ? 0 : 1);
        //cerr << "advance = " << advance << endl;
        //cerr << "cl_weight = " << cl_weight << endl;

        double correct = 0.0;

        Index_Iterator ex_start
            = (start_x == 0
               ? index.begin()
               : std::lower_bound(index.begin(), index.end(), start_x,
                                  Find_Example()));
        Index_Iterator ex_end   = index.end();

        for (unsigned x = start_x;  x < end_x;  ++x) {
            /* Find the number of examples that we have. */
            Index_Iterator ex_range = ex_start;
            while (ex_range != ex_end && ex_range->example() == x)
                ++ex_range;

            if (example_weights[x] == 0.0) {
                ex_start = ex_range;
                continue;  // must be zero...
            }

            //if (x < 10)
            //    cerr << "x = " << x << " weight = " << example_weights[x]
            //         << " advance = " << advance << " cl_weight = "
            //         << cl_weight << " classifier.predict(data[x]) = "
            //         << stump.predict(data[x]) << endl;

            float t = weights_updater(stump, opt_info, cl_weight, labels[x],
                                      ex_start, ex_range,
                                      &weights[x][0], advance);
            
            output_updater(stump, opt_info, cl_weight, labels[x],
                           ex_start, ex_range,
                           &output[x][0], advance);

            correct += scorer(labels[x], &output[x][0], &output[x][0] + nl)
                * example_weights[x];

            //cerr << "  updater returned total " << t << endl;
            total += t;
            
            __builtin_prefetch(&weights[x][0] + 24, 1, 3);
            __builtin_prefetch(&output[x][0] + 24, 1, 3);
            ex_start = ex_range;
        }
        
        accuracy = correct / example_weights.total();

        return total;
    }

    /** Apply the given stump to the given weights, given the training
        data.  This will update all of the weights. */
    float operator () (const Classifier_Impl & classifier,
                       const Optimization_Info & opt_info,
                       float cl_weight,
                       boost::multi_array<float, 2> & weights,
                       boost::multi_array<float, 2> & output,
                       const Training_Data & data,
                       const distribution<float> & example_weights,
                       double & accuracy,
                       int start_x = 0, int end_x = -1) const
    {
        if (end_x == -1) end_x = data.example_count();
        
        size_t nl = output.shape()[1];

        double correct = 0.0;
        double total = 0.0;

        const std::vector<Label> & labels
            = data.index().labels(classifier.predicted());

        int advance = (weights.shape()[1] == 1 ? 0 : 1);


        using namespace std;
        //map<float, int> output_freq;

        for (unsigned x = start_x;  x < end_x;  ++x) {
            if (example_weights[x] == 0.0) continue;  // must be zero...

            //float val = classifier.predict(0, data[x]);
            //output_freq[val] += 1;
            
            //if (x < 10)
            //    cerr << "x = " << x << " weight = " << example_weights[x]
            //         << " advance = " << advance << " cl_weight = "
            //         << cl_weight << " classifier.predict(data[x]) = "
            //         << classifier.predict(data[x]) << endl;

            float t = weights_updater(classifier, opt_info,
                                      cl_weight, labels[x],
                                      data[x], &weights[x][0], advance);
            
            output_updater(classifier, opt_info, cl_weight, labels[x],
                           data[x], &output[x][0], advance);
            
            correct += scorer(labels[x], &output[x][0], &output[x][0] + nl)
                * example_weights[x];
            
            //cerr << "  updater returned total " << t << endl;
            total += t;
            
            __builtin_prefetch(&weights[x][0] + 24, 1, 3);
            __builtin_prefetch(&output[x][0] + 24, 1, 3);
        }

#if 0        
        for (map<float, int>::const_iterator it = output_freq.begin(),
                 end = output_freq.end();  it != end;  ++it)
            cerr << "  value " << it->first << " seen " << it->second
                 << " times" << endl;
        
        cerr << "updater: total = " << total << endl;
#endif
        
        accuracy = correct / example_weights.total();
        
        return total;
    }
};


/*****************************************************************************/
/* UPDATE_SCORES                                                             */
/*****************************************************************************/

/** This class updates a scores matrix based on the output of the
    classifier, as well as calculating the accuracy of the classifier.
*/

template<class Output_Updater, class Scorer>
struct Update_Scores {
    Update_Scores(const Output_Updater & output_updater
                      = Output_Updater())
        : output_updater(output_updater)
    {
    }

    Output_Updater output_updater;
    Scorer scorer;

    /** Apply the given stump to the given weights, given the training
        data.  This will update all of the weights.  Returns the total
        correct.
    */
    double operator () (const Stump & stump,
                        const Optimization_Info & opt_info,
                        float cl_weight,
                        boost::multi_array<float, 2> & output,
                        const Training_Data & data,
                        const distribution<float> & example_weights,
                        int start_x = 0, int end_x = -1) const
    {
        if (end_x == -1) end_x = data.example_count();

        size_t nl = output.shape()[1];
        
        Joint_Index index
            = data.index().joint(stump.predicted(),
                                 stump.split.feature(),
                                 BY_EXAMPLE,
                                 IC_VALUE | IC_EXAMPLE);

        const std::vector<Label> & labels
            = data.index().labels(stump.predicted());

        Index_Iterator ex_start
            = (start_x == 0
               ? index.begin()
               : std::lower_bound(index.begin(), index.end(), start_x,
                                  Find_Example()));
        Index_Iterator ex_end   = index.end();

        int advance = (output.shape()[1] == 1 ? 0 : 1);

        double correct = 0.0;

        for (unsigned x = start_x;  x < end_x;  ++x) {
            /* Find the number of examples that we have. */
            Index_Iterator ex_range = ex_start;
            while (ex_range != ex_end && ex_range->example() == x)
                ++ex_range;

            if (example_weights[x] == 0.0) {
                ex_start = ex_range;
                continue;  // must be zero...
            }

            output_updater(stump, opt_info, cl_weight, labels[x],
                           ex_start, ex_range,
                           &output[x][0], advance);
            
            correct += scorer(labels[x], &output[x][0], &output[x][0] + nl)
                * example_weights[x];
            
            __builtin_prefetch(&output[x][0] + 24, 1, 3);
            ex_start = ex_range;
        }

        return correct;
    }

    /** Apply the given classifier to the given weights, given the training
        data.  This will update all of the weights.  Returns the total
        correct.
    */
    double operator () (const Classifier_Impl & classifier,
                        const Optimization_Info & opt_info,
                        float cl_weight,
                        boost::multi_array<float, 2> & output,
                        const Training_Data & data,
                        const distribution<float> & example_weights,
                        int start_x = 0, int end_x = -1) const
    {
        if (end_x == -1) end_x = data.example_count();

        size_t nl = output.shape()[1];

        const std::vector<Label> & labels
            = data.index().labels(classifier.predicted());

        int advance = (output.shape()[1] == 1 ? 0 : 1);

        double correct = 0.0;

        for (unsigned x = start_x;  x < end_x;  ++x) {
            if (example_weights[x] == 0.0) continue;  // must be zero...

            output_updater(classifier, opt_info, cl_weight, labels[x],
                           data[x], &output[x][0], advance);
            
            correct += scorer(labels[x], &output[x][0], &output[x][0] + nl)
                * example_weights[x];
            
            __builtin_prefetch(&output[x][0] + 24, 1, 3);
        }
        
        return correct;
    }
};

} // namespace ML



#endif /* __boosting__boosting_core_h__ */

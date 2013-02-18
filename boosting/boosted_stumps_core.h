/* boosted_stumps_core.h                                           -*- C++ -*-
   Jeremy Barnes, 23 February 2004
   Copyright (c) 2004 Jeremy Barnes.  All rights reserved.

   This file is part of "Jeremy's Machine Learning Library", copyright (c)
   1999-2005 Jeremy Barnes.
   
   This program is available under the GNU General Public License, the terms
   of which are given by the file "license.txt" in the top level directory of
   the source code distribution.  If this file is missing, you have no right
   to use the program; please contact the author.

   This program is distributed in the hope that it will be useful, but
   WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
   or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
   for more details.

   ---

   Core of the boosted stumps routines.
*/

#ifndef __boosting__boosted_stumps_core_h__
#define __boosting__boosted_stumps_core_h__

#include "jml/compiler/compiler.h"
#include "training_data.h"
#include "training_index.h"
#include "stump.h"
#include "stump_predict.h"

namespace ML {

/** The boosting loss function.  It is exponential in the margin. */
struct Boosting_Loss {
    JML_ALWAYS_INLINE
    float operator () (int label, int corr, float pred, float current) const
    {
        int correct = (corr == label);
        //cerr << "label = " << label << " corr = " << corr << " pred = "
        //     << pred << " correct = " << correct;
        *((int *)&pred) ^= (correct << 31);  // flip sign bit if correct
        //cerr << " flipped = " << pred << " current = " << current
        //     << " result = " << current * exp(pred) << endl;
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
        *((int *)&pred) ^= (correct << 31);  // flip sign bit if correct
        float qz = exp(pred);
        return 1.0 / ((z/qz * ((1.0 / current) - 1.0)) - 1.0);
    }
    
    double z;
};

/* A "loss" function used to update a set of prediction weights for some
   training data. */
struct Boosting_Predict {
    JML_ALWAYS_INLINE
    float operator () (int label, int corr, float pred, float current) const
    {
        return current + pred;
    }
};

template<class Fn>
struct Binsym_Update {
    Binsym_Update(const Fn & fn = Fn()) : fn(fn) {}
    Fn fn;

    template<class FeatureIt, class WeightIt>
    float operator () (const Stump & stump, float cl_weight, int corr,
                       FeatureIt ex_start, FeatureIt ex_range,
                       WeightIt weight_begin, int advance) const
    {
        float pred = stump.predict(0, ex_start, ex_range) * cl_weight;

        //cerr << "    pred = { " << pred << " " << -pred << " }" << endl;

        *weight_begin = fn(0, corr, pred, *weight_begin);
        return *weight_begin * 2.0;
    }

    template<class PredIt, class WeightIt>
    float operator () (PredIt pred_it, PredIt pred_end, int corr,
                       WeightIt weight_begin, int advance) const
    {
        *weight_begin = fn(0, corr, *pred_it, *weight_begin);
        return *weight_begin * 2.0;
    }
}; 

template<class Fn>
struct Normal_Update {
    Normal_Update(const Fn & fn = Fn()) : fn(fn) {}
    Fn fn;
    mutable distribution<float> pred;

    /* Make a prediction and return an update. */
    template<class FeatureIt, class WeightIt>
    float operator () (const Stump & stump, float cl_weight, int corr,
                       FeatureIt ex_start, FeatureIt ex_range,
                       WeightIt weight_begin, int advance) const
    {
        size_t nl = stump.label_count();
        if (pred.size() != nl) pred = distribution<float>(nl);

        stump.predict(pred, ex_start, ex_range);
        if (cl_weight != 1.0) pred *= cl_weight;

        //cerr << "    pred = " << pred << endl;
        
        return operator () (pred.begin(), pred.end(), corr, weight_begin,
                            advance);
    }
    
    /* Perform an update once the predictions are already known. */
    template<class PredIt, class WeightIt>
    float operator () (PredIt pred_it, PredIt pred_end, int corr,
                       WeightIt weight_begin, int advance) const
    {
        size_t nl = pred_end - pred_it;
        //cerr << "nl = " << nl << " advance = " << advance << endl;
        //cerr << __PRETTY_FUNCTION__ << endl;
        float total = 0.0;
        
        if (advance) {
            for (unsigned l = 0;  l < nl;  ++l) {
                *weight_begin = fn(l, corr, pred_it[l], *weight_begin);
                total += *weight_begin;
                weight_begin += advance;
            }
        }
        else
            total = Binsym_Update<Fn>(fn)(pred_it, pred_end, corr,
                                          weight_begin, 0);
        
        return total;
    }
};


/*****************************************************************************/
/* UPDATE_WEIGHTS                                                            */
/*****************************************************************************/

/** This class updates a weights matrix in response to a new stump having
    been learned.
*/

template<class Loss>
struct Update_Weights {
    Update_Weights(const Loss & loss = Loss()) : loss(loss) {}
    Loss loss;

    /** Apply the given stump to the given weights, given the training
        data.  This will update all of the weights. */
    float operator () (const Stump & stump, float cl_weight,
                       fixed_array<float, 2> & weights,
                       const Training_Data & data) const
    {
        Joint_Index index
            = data.index().joint(stump.predicted(), stump.feature, BY_EXAMPLE);

#if 0
        cerr << "update_weights:" << endl;
        for (Index_Iterator it = index.begin();  it != index.end();  ++it) {
            cerr << "i " << it - index.begin() << " example = "
                 << it->example() << " label = " << it->label() << endl;
        }
#endif
        
        double total = 0.0;

        Index_Iterator ex_start = index.begin();
        Index_Iterator ex_end   = index.end();

        const std::vector<Label> & labels
            = data.index().labels(stump.predicted());

        int advance = (weights.dim(1) == 1 ? 0 : 1);
        //cerr << "advance = " << advance << endl;
        //cerr << "cl_weight = " << cl_weight << endl;

        for (unsigned x = 0;  x < data.example_count();  ++x) {

            /* Find the number of examples that we have. */
            Index_Iterator ex_range = ex_start;
            while (ex_range != ex_end && ex_range->example() == x)
                ++ex_range;

#if 0
            cerr << "updating example " << x << " with range of "
                 << (ex_range - ex_start) << endl;

            if (ex_range != ex_start || true) {
                cerr << "  updated from "
                     << distribution<float>(&weights[x][0],
                                            &weights[x][0] + weights.dim(1))
                     << endl;
            }
#endif

            float t = loss(stump, cl_weight, labels[x],
                           ex_start, ex_range,
                           &weights[x][0], advance);
            //cerr << "  loss returned total " << t << endl;
            total += t;

#if 0
            if (ex_range != ex_start || true) {
                cerr << "  updated to "
                     << distribution<float>(&weights[x][0],
                                            &weights[x][0] + weights.dim(1))
                     << endl;
            }

            cerr << "  total now " << total << endl;
#endif

            __builtin_prefetch(&weights[x][0] + 48, 1, 3);
            ex_start = ex_range;
        }
        
        //cerr << "total = " << total << endl;

        return total;
    }

    /** Apply the given population of stumps with the given classifier weight
        distribution to the given sample weights, given the training data.  */
    float operator () (const std::vector<Stump> & stumps,
                       const std::vector<float> & cl_weights,
                       fixed_array<float, 2> & sample_weights,
                       const Training_Data & data) const
    {
        double total = 0.0;
#if 0  // doesn't work for logistic
        /* This version is specially unrolled to first predict all of the
           stumps on each example, and then only update the weights one time.
           It doesn't make much of a difference, except when the committee
           size is large (in which case much less time is spent calculating
           exponentials), or when the weights array is very large (in which
           case we make much better use of the cache).
        */
        /* We need to predict all the stumps first, so we use this object. */
        Normal_Update<Boosting_Predict> update;
        
        /* This contains the action of the stumps on the current example.  We
           use double precision as it won't lost accuracy over large
           numbers of rules. */
        distribution<double> pred(data.label_count());

        /* Get data for the feature for each of our stumps. */
        std::vector<const Training_Data::Feature_Data *>
            feature_data(stumps.size());
        std::vector<std::shared_ptr<const example_data_type> >
            sorted(stumps.size());
        std::vector<example_data_type::const_iterator> ex_start(stumps.size());
        std::vector<example_data_type::const_iterator> ex_end(stumps.size());
        for (unsigned i = 0;  i < stumps.size();  ++i) {
            feature_data[i] = &data[stumps[i].feature];
            sorted[i] = get_example_data(*feature_data[i]);
            ex_start[i] = sorted[i]->begin();
            ex_end[i] = sorted[i]->end();
        }
        
        int advance = (sample_weights.dim(1) == 1 ? 0 : 1);
        
        for (unsigned x = 0;  x < data.example_count();  ++x) {
            /* Clear the predictions on the current example. */
            std::fill(pred.begin(), pred.end(), 0.0);
            
            /* Predict this example with each stump. */
            for (unsigned i = 0;  i < stumps.size();  ++i) {

                /* Find the number of examples that we have. */
                example_data_type::const_iterator ex_range = ex_start[i];
                while (ex_range != ex_end[i]
                       && (*ex_range)->example() == x)
                    ++ex_range;

                update(stumps[i], cl_weights[i], data.data[x]->label,
                       ex_start[i], ex_range, pred.begin(), advance);
                
                ex_start[i] = ex_range;
                
                if (ex_end[i] != ex_range)
                    __builtin_prefetch(*ex_range, 1, 3);
            }
            __builtin_prefetch(&sample_weights[x][0] + 48, 1, 3);
            __builtin_prefetch(&data.data[x] + 2, 1, 3);
            
            /* Now, we apply the loss function to our weights, given our
               combined prediction(s). */
            total += loss(pred.begin(), pred.end(), data.data[x]->label,
                          &sample_weights[x][0], advance);
        }

#else
        for (unsigned i = 0;  i < stumps.size();  ++i) {
            total = operator () (stumps[i], cl_weights.at(i), sample_weights,
                                 data);
            //cerr << "after stump " << i << ": total = " << total << endl;
        }
#endif
        return total;
    }
};

} // namespace ML



#endif /* __boosting__boosted_stumps_core_h__ */

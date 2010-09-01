/* evaluation.h                                                    -*- C++ -*-
   Jeremy Barnes, 16 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$

   Functions to ease evaluation of correctness.
*/

#ifndef __boosting__evaluation_h__
#define __boosting__evaluation_h__

#include "config.h"
#include "jml/stats/distribution.h"
#include "feature_set.h"
#include <algorithm>
#include <boost/multi_array.hpp>


namespace ML {

struct Correctness {
    Correctness(float correct = 0.0, float possible = 0.0, float margin = 0.0)
        : correct(correct), possible(possible), margin(margin)
    {
    }

    float correct;    ///< Is it correct or not?
    float possible;   ///< Is it possible to get it correct or not?
    float margin;     ///< It is correct by what margin?

    std::string print() const;
};

std::ostream & operator << (std::ostream & stream, const Correctness & corr);

/** Returns the correctness of the given results vector.
    First return value: correctness: 0.0 (completely incorrect) to 1.0
    (completely correct).

    returns: element 0: correctness (0.0 to 1.0)
             element 1: possible (0.0 = label is missing, otherwise 1.0)
             element 2: margin (-1.0 to 1.0)
*/
Correctness
correctness(const distribution<float> & results,
            const Feature & label,
            const Feature_Set & features,
            double tolerance = 1e-6);

template<class ResultIterator>
Correctness
correctness(ResultIterator rbegin, ResultIterator rend, int label,
            double tolerance = 1e-6)
{
    using namespace std;

    //cerr << "rend - rbegin = " << rend - rbegin << endl;

    if (label >= (rend - rbegin))  // label not known about
        return Correctness(0.0, 1.0, *std::max_element(rbegin, rend));
    
    double max_correct = rbegin[label];
    double max_incorrect = -INFINITY;
    int num_incorrect_equal = 0;
    int num_correct = 0;
    for (int i = 0; rbegin != rend;  ++rbegin, ++i) {
        //cerr << "i = " << i << " val = " << *rbegin
        //     << " label = " << label << " max_correct = "
        //     << max_correct << " max_incorrect = " << max_incorrect
        //     << " num_inc_eq = " << num_incorrect_equal << endl;
        if (label == i) { ++num_correct; continue; } // correct...
        //cerr << "updating... val = " << *rbegin << endl;
        max_incorrect = std::max<double>(max_incorrect, *rbegin);

        //cerr << "max_correct - *rbegin = " << max_correct - *rbegin
        //     << endl;
        if (abs(max_correct - *rbegin) < tolerance)
            ++num_incorrect_equal;
    }

    //cerr << " label = " << label << " max_correct = "
    //     << max_correct << " max_incorrect = " << max_incorrect
    //     << " num_inc_eq = " << num_incorrect_equal << endl;
    
    if (max_correct - max_incorrect > tolerance)  // correct
        return Correctness(1.0, 1.0, max_correct - max_incorrect);
    else if (num_incorrect_equal > 0)  // tied
        return Correctness((double)num_correct / (num_incorrect_equal + num_correct), 1.0, 0.0);
    else return Correctness(0.0, 1.0, max_correct - max_incorrect); // inc
}

/** Returns the margin of the given predictor. */
float margin(const distribution<float> & results,
             const Feature & label,
             const Feature_Set & features,
             double tolerance = 1e-6);

/** Distribution to use to indicate that weights are uniform.  It is
    empty. */
extern const distribution<float> UNIFORM_WEIGHTS;


/** Calculate the prediction accuracy over a training set, using a set of
    already cached predictions.

    \param output             the output of the classifier for each of the
                              examples in \p data.
    \param data               the training data used to calculate the
                              accuracy over.
    \param example_weights    a weighting of the examples.  The examples
                              will count as if they were duplicated this
                              number of times.  An empty distribution is
                              the same as all ones.
    \returns                  a number between 0 and 1, giving the
                              proportion of the examples in data that were
                              correct.

    \pre                      data.size() == output.size()
    \pre                      \f$\forall\f$ i:
                              output[i].size() == label_count()

    Note that this method cannot be overridden, as it is a static
    method.  It is provided mainly for the boosting algorithm, which can
    calculate the accuracy on each training iteration in constant time
    per iteration rather than a linearly increasing time if this method
    is used.
*/
float accuracy(const std::vector<distribution<float> > & output,
               const Training_Data & data,
               const Feature & label,
               const distribution<float> & example_weights
                   = UNIFORM_WEIGHTS);

/** Calculate the prediction accuracy over a training set, using a set of
    already cached predictions.

    \param output             the output of the classifier for each of the
                              examples in \p data.
    \param data               the training data used to calculate the
                              accuracy over.
    \param example_weights    a weighting of the examples.  The examples
                              will count as if they were duplicated this
                              number of times.  An empty distribution is
                              the same as all ones.
    \returns                  a number between 0 and 1, giving the
                              proportion of the examples in data that were
                              correct.
    \pre                      data.size() == output.shape()[0]
    \pre                      label_count() == output.shape()[1]

    Note that this method cannot be overridden, as it is a static
    method.  Again, it is provided for the boosting algorithm.
*/
float accuracy(const boost::multi_array<float, 2> & output,
               const Training_Data & data,
               const Feature & label,
               const distribution<float> & example_weights
                   = UNIFORM_WEIGHTS);

} // namespace ML


#endif /* __boosting__evaluation_h__ */

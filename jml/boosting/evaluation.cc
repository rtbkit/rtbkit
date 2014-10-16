/* evaluation.cc
   Jeremy Barnes, 16 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$

   Utility functions for evaluation of correctness of classifiers.
*/

#include "evaluation.h"
#include "training_data.h"


using namespace std;


namespace ML {

std::string Correctness::print() const
{
    return format("(corr: %4.12f poss: %4.12f marg: %+5.12f)",
                  correct, possible, margin);
}

std::ostream & operator << (std::ostream & stream, const Correctness & corr)
{
    return stream << corr.print();
}

const distribution<float> UNIFORM_WEIGHTS;


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
            double tolerance)
{
    /* Make sure we handle:
       1.  Multiple correct answers (only need one of them)
       2.  Multiple top answers (divided between)
       3.  No correct answers (not correct)
    */
    
    /* Find the range of all correct answers. */
    std::pair<Feature_Set::const_iterator, Feature_Set::const_iterator> range
        = features.find(label);

    int num_correct = range.second - range.first;

    if (num_correct == 0)
        return Correctness(0.0, 0.0, -results.max());
    
    else if (num_correct == 1) {
        return correctness(results.begin(), results.end(),
                           (int)range.first.value(), tolerance);
    }
    else
        throw Exception("correctness(): can't handle multiple labels yet");
}

float margin(const distribution<float> & results,
             const Feature & label,
             const Feature_Set & features,
             double tolerance)
{
    return correctness(results, label, features).margin;
}

float 
accuracy(const std::vector<distribution<float> > & output,
         const Training_Data & data,
         const Feature & label,
         const distribution<float> & example_weights)
{
    if (output.size() != data.example_count())
        throw Exception("accuracy: output and data sizes don't match");

    if (!example_weights.empty()
        && example_weights.size() != data.example_count())
        throw Exception("Classifier_Impl::accuracy(): dataset and weight "
                        "vector sizes don't match");

    double correct = 0.0;
    double total = 0.0;

    for (unsigned i = 0;  i < data.example_count();  ++i) {
        double w = (example_weights.empty() ? 1.0 : example_weights[i]);
        const distribution<float> & result = output[i];
        
        Correctness c = correctness(result, label, data[i]);
        correct += w * c.possible * c.correct;
        total += w * c.possible;
    }

    float result = correct / total;
    
    return result;
}

float 
accuracy(const boost::multi_array<float, 2> & output,
         const Training_Data & data,
         const Feature & label,
         const distribution<float> & example_weights)
{
    unsigned nx = output.shape()[0];
    if (nx != data.example_count())
        throw Exception("accuracy: data set and output size don't match");

    unsigned nl = output.shape()[1];
    /* TODO: put this check back in */
    //if (nl != label_count_)
    //    throw Exception("accuracy: data set and output labels differ");
    
    double correct = 0.0, total = 0.0;

#if 0  // doesn't work until we have a dense results vector
    if (nl == 2) {
        for (unsigned i = 0;  i < nx;  ++i) {
            double w = (example_weights.empty() ? 1.0 : example_weights[i]);
            //cerr << "data[i].label = " << data[i].label << endl;
            //cerr << "data[i].label.is(1) = " << data[i].label.is(1) << endl;
            bool c = output[i][data[i].label.is(1)]
                > output[i][1 - data[i].label.is(1)];
            __builtin_prefetch(&output[i][0] + 48, 0, 0);
            if (c) correct += w;
            total += w;
        }
    }
#else
    if (false) ;
#endif
    else {
        for (unsigned i = 0;  i < nx;  ++i) {
            //cerr <<"***** ex " << i << endl;
            distribution<float> result(&output[i][0], &output[i][0] + nl);
            double w = (example_weights.empty() ? 1.0 : example_weights[i]);
            Correctness c = correctness(result, label, data[i]);
            correct += w * c.possible * c.correct;
            total += w * c.possible;
            //cerr << "ex " << i << " poss " << c.possible << " corr "
            //     << c.correct << " res " << result << endl;
            //cerr << data.feature_space()->print(data[i]) << endl;
        }
    }
    
    float result = correct / total;
    
    return result;
}

} // namespace ML


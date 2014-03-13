/* boosted_stumps_testing.h                                        -*- C++ -*-
   Jeremy Barnes, 23 February 2004
   Copyright (c) 2004 Jeremy Barnes.  All rights reserved.
   $Source$

   Testing code for the boosted stumps.  Contains instrumented versions of
   various structures that are used in boosted stumps training.
*/

#ifndef __boosting__boosted_stumps_testing_h__
#define __boosting__boosted_stumps_testing_h__


namespace ML {


template<class Loss>
struct Update_Weights_Basic {
    
    /** The loss function we are using. */
    Loss loss;

    float operator () (const Stump & stump, boost::multi_array<float, 2> & weights,
                       const Training_Data & data) const
    {
        for (unsigned m = 0;  m < data.example_count();  ++m) {
        
            distribution<float> predictions
                = stump.predict(data.data[m]->features);
            int corr = data.data[m]->label;
            
            for (unsigned l = 0;  l < weights.shape()[1];  ++l) {
                float prediction = predictions[l];
                weights[m][l] = loss(l, corr, prediction, weights[m][l]);
                total += weights[m][l];
            }
        }

        /* If we had a 1d weights vector, we need to double the total. */
        if (nl == 2 && weights.shape()[1] == 1) total *= 2.0;
        
        return total;
    }
};

/* Non-optimised version of the boosting loss function. */
struct Boosting_Loss_Test {
    JML_ALWAYS_INLINE
    float operator () (bool correct, float pred, float current) const
    {
        float sign = (correct == l ? 1.0 : -1.0);
        current *= exp(-sign * pred);
        return current;
    }
};

#if 0
    float operator () ()
    {


        //if (ex_range != ex_end && (*ex_range)->example() <= x)
        //    throw Exception("ex_range");

        if (bin_sym) {
            /* Binary symmetric version. */
            float prediction = stump.predict(1, ex_start, ex_range);
            int corr = data.data[x]->label;
            float sign = (corr == 1 ? -1.0 : 1.0);
            float factor = exp(sign * prediction);
            weights[x][0] *= factor;
            total += weights[x][0] * 2.0;
        }
        else if (label_count() == 2) {
            /* Binary version.  We can speed it up. */
            float prediction = stump.predict(1, ex_start, ex_range);
            int corr = data.data[x]->label;
            float sign = (corr == 1 ? -1.0 : 1.0);
            float factor = exp(sign * prediction);
            weights[x][0] *= factor;
            weights[x][1] *= factor;
            total += weights[x][0] + weights[x][1];
        }
        else {
            stump.predict(pred, ex_start, ex_range);
            int corr = data.data[x]->label;
            for (unsigned l = 0;  l < label_count();  ++l) {
                float prediction = pred[l];
                float sign = (corr == l ? 1.0 : -1.0);
                weights[x][l] *= exp(-sign * prediction);
                total += weights[x][l];
            }
        }
    }
};
#endif




} // namespace ML


#endif /* __boosting__boosted_stumps_testing_h__ */

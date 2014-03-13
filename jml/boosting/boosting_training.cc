/* boosting_training.cc
   Jeremy Barnes, 16 March 2006
   Copyright (c) 2006 Jeremy Barnes  All rights reserved.
   $Source$

*/

#include "boosting_training.h"
#include "jml/math/xdiv.h"
#include "jml/arch/simd_vector.h"
#include "jml/utils/floating_point.h"
#include "jml/utils/vector_utils.h"
#include "boosting_core.h"
#include "jml/utils/smart_ptr_utils.h"


using namespace std;


namespace ML {

void 
update_weights(boost::multi_array<float, 2> & weights,
               const Stump & stump,
               const Optimization_Info & opt_info,
               const Training_Data & data,
               Cost_Function cost,
               bool bin_sym,
               int parent)
{
    vector<Stump> stumps(1, stump);
    vector<Optimization_Info> opt_infos(1, opt_info);
    distribution<float> cl_weights(1, 1.0 / stump.Z);
    update_weights(weights, stumps, opt_infos,
                   cl_weights, data, cost, bin_sym,
                   parent);
}

void 
update_weights(boost::multi_array<float, 2> & weights,
               const std::vector<Stump> & stumps,
               const std::vector<Optimization_Info> & opt_infos,
               const distribution<float> & cl_weights,
               const Training_Data & data,
               Cost_Function cost,
               bool bin_sym,
               int parent)
{
    /* Work out the weights.  This depends upon the 1/Z score. */
    float total_z = 0.0;
    for (unsigned s = 0;  s < stumps.size();  ++s) {
        float Z = stumps[s].Z;
        if (Z < 1e-5) ;
        else { total_z += 1.0 / Z; }
    }
    
    /* Get the average Z score, which is needed by the logistic update. */
    float avg_z = total_z / cl_weights.total();
    
    /* Update the d distribution. */
    double total = 0.0;
    size_t nl = weights.shape()[1];

    if (cost == CF_EXPONENTIAL) {
        typedef Boosting_Loss Loss;
        if (bin_sym) {
            //PROFILE_FUNCTION(t_update);
            typedef Binsym_Updater<Loss> Updater;
            typedef Update_Weights<Updater> Update;
            Update update;
            
            total = update(stumps, opt_infos, cl_weights, weights, data);
        }
        else {
            //PROFILE_FUNCTION(t_update);
            typedef Normal_Updater<Loss> Updater;
            typedef Update_Weights<Updater> Update;
            Updater updater(nl);
            Update update(updater);
            
            total = update(stumps, opt_infos, cl_weights, weights, data);
        }
    }
    else if (cost == CF_LOGISTIC) {
        typedef Logistic_Loss Loss;
        Loss loss(avg_z);

        if (bin_sym) {
            //PROFILE_FUNCTION(t_update);
            typedef Normal_Updater<Loss> Updater;
            typedef Update_Weights<Updater> Update;
            Updater updater(nl, loss);
            Update update(updater);
            
            total = update(stumps, opt_infos, cl_weights, weights, data);
        }
        else {
            //PROFILE_FUNCTION(t_update);
            typedef Normal_Updater<Loss> Updater;
            typedef Update_Weights<Updater> Update;
            Updater updater(nl, loss);
            Update update(updater);
            
            total = update(stumps, opt_infos, cl_weights, weights, data);
        }
    }
    else throw Exception("update_weights: unknown cost function");

    for (unsigned m = 0;  m < data.example_count();  ++m)
        for (unsigned l = 0;  l < weights.shape()[1];  ++l)
            weights[m][l] /= total;
}

void 
update_scores(boost::multi_array<float, 2> & example_scores,
              const Training_Data & data,
              const Stump & stump,
              const Optimization_Info & opt_info,
              int parent)
{
    //PROFILE_FUNCTION(t_update);
    size_t nl = stump.label_count();
    if (nl != example_scores.shape()[1])
        throw Exception("update_scores: label counts don't match");

    size_t nx = data.example_count();
    if (nx != example_scores.shape()[0])
        throw Exception("update_scores: example counts don't match");

    typedef Normal_Updater<Boosting_Predict> Updater;
    typedef Update_Weights<Updater> Update;
    Updater updater(nl);
    Update update(updater);

    update(stump, opt_info, 1.0, example_scores, data);
}

void 
update_scores(boost::multi_array<float, 2> & example_scores,
              const Training_Data & data,
              const Classifier_Impl & classifier,
              const Optimization_Info & opt_info,
              int parent)
{
    //PROFILE_FUNCTION(t_update);
    size_t nl = classifier.label_count();
    if (nl != example_scores.shape()[1])
        throw Exception("update_scores: label counts don't match");

    size_t nx = data.example_count();
    if (nx != example_scores.shape()[0])
        throw Exception("update_scores: example counts don't match");

    typedef Normal_Updater<Boosting_Predict> Updater;
    typedef Update_Weights<Updater> Update;
    Updater updater(nl);
    Update update(updater);

    update(classifier, opt_info, 1.0, example_scores, data);
}

void 
update_scores(boost::multi_array<float, 2> & example_scores,
              const Training_Data & data,
              const std::vector<Stump> & trained_stumps,
              const std::vector<Optimization_Info> & opt_infos,
              int parent)
{
    if (trained_stumps.empty()) return;

    //PROFILE_FUNCTION(t_update);
    size_t nl = trained_stumps[0].label_count();
    if (nl != example_scores.shape()[1])
        throw Exception("update_scores: label counts don't match");

    size_t nx = data.example_count();
    if (nx != example_scores.shape()[0])
        throw Exception("update_scores: example counts don't match");

    if (trained_stumps.empty())
        throw Exception("update_scores: no training has been done");

    for (unsigned i = 0;  i < nx;  ++i) {
        distribution<float> total(nl, 0.0);
        for (unsigned s = 0;  s < trained_stumps.size();  ++s)
            total += trained_stumps[s].predict(data[i], opt_infos[s]);
        total /= trained_stumps.size();
        std::transform(total.begin(), total.end(), &example_scores[i][0],
                       &example_scores[i][0], std::plus<float>());
    }
}

} // namespace ML

ENUM_INFO_NAMESPACE
  
const Enum_Opt<ML::Cost_Function>
Enum_Info<ML::Cost_Function>::OPT[2] = {
    { "exponential",      ML::CF_EXPONENTIAL   },
    { "logistic",         ML::CF_LOGISTIC      } };

const char * Enum_Info<ML::Cost_Function>::NAME
   = "Cost_Function";

END_ENUM_INFO_NAMESPACE

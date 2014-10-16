/* stump_generator.cc
   Jeremy Barnes, 15 March 2006
   Copyright (c) 2006 Jeremy Barnes  All rights reserved.
   $Source$

   Generator for boosted stumps.
*/

#include "stump_generator.h"
#include "registry.h"
#include <boost/timer.hpp>
#include <boost/progress.hpp>
#include "training_index.h"
#include "weighted_training.h"
#include "committee.h"
#include "stump_training.h"
#include "stump_training_core.h"
#include "stump_training_bin.h"
#include "stump_training_multi.h"
#include "stump_training_parallel.h"
#include "stump_accum.h"
#include "stump_regress.h"
#include "binary_symmetric.h"
#include "jml/utils/environment.h"
#include "jml/utils/info.h"
#include "jml/arch/tick_counter.h"
#include "jml/utils/smart_ptr_utils.h"
#include <boost/bind.hpp>

using namespace std;


namespace ML {

double non_missing_ticks = 0.0;
size_t non_missing_calls = 0;

namespace {

size_t ticks_train_weighted = 0;

#if 0
struct Ticks {
    ~Ticks() {
        cerr << "train_weighted: " << ticks_train_weighted
             << "t, " << ticks_train_weighted * seconds_per_tick << "s" << endl;
        cerr << "non_missing: " << (size_t)non_missing_ticks << "t, "
             << non_missing_ticks * seconds_per_tick << "s, "
             << non_missing_calls << "c, "
             << non_missing_ticks / non_missing_calls << "t/c" << endl;
    }
} print_ticks;
#endif

}

/*****************************************************************************/
/* STUMP_GENERATOR                                                           */
/*****************************************************************************/

Stump_Generator::
Stump_Generator()
{
    defaults();
}

Stump_Generator::~Stump_Generator()
{
}

void
Stump_Generator::
configure(const Configuration & config)
{
    Classifier_Generator::configure(config);
    
    config.find(committee_size,       "committee_size");
    config.find(feature_prop,         "feature_prop");
    config.find(trace,                "trace");
    config.find(update_alg,           "update_alg");
    config.find(ignore_highest,       "ignore_highest");
}

void
Stump_Generator::
defaults()
{
    Classifier_Generator::defaults();
    committee_size = 1;
    feature_prop = 1.0;
    trace = 0;
    update_alg = Stump::NORMAL;
    ignore_highest = 0.0;
}

Config_Options
Stump_Generator::
options() const
{
    Config_Options result = Classifier_Generator::options();
    result
        .add("committee_size", committee_size, ">=1",
             "learn a committee of size N at once")
        .add("feature_prop", feature_prop, "0.0<N<=1.0",
             "try lazy training on only this proportion of features per iter")
        .add("update_alg", update_alg,
             "select the harshness of the update algorithm")
        .add("ignore_highest", ignore_highest, "0.0<=N<1.0",
             "ignore the examples witht the highest N% of weights")
        .add("trace", trace, "0-",
             "trace training (very detailed) to given level");

    return result;
}

void
Stump_Generator::
init(std::shared_ptr<const Feature_Space> fs, Feature predicted)
{
    Classifier_Generator::init(fs, predicted);
    model = Stump(fs, predicted);
}

std::shared_ptr<Classifier_Impl>
Stump_Generator::
generate(Thread_Context & context,
         const Training_Data & training_set,
         const Training_Data & validation_set,
         const distribution<float> & training_ex_weights,
         const distribution<float> & validate_ex_weights,
         const std::vector<Feature> & features_, int) const
{
    vector<Feature> features = features_;

    boost::timer timer;

    Feature predicted = model.predicted();

    Stump stumps(training_set.feature_space(), predicted);

    boost::multi_array<float, 2> weights
        = expand_weights(training_set, training_ex_weights, predicted);

    if (committee_size == 1) {
        Stump current
            = train_weighted(context, training_set, weights, features_);
        return make_sp(current.make_copy());
    }
    else {
        /* Train a committee of stumps. */
        vector<Stump> all_trained
            = train_all(context, training_set, weights, features_);

        /* Work out the weights.  This depends upon the 1/Z score. */
        distribution<float> cl_weights(all_trained.size());
        float total_z = 0.0;
        for (unsigned s = 0;  s < all_trained.size();  ++s) {
            float Z = all_trained[s].Z;
            if (Z < 1e-5) cl_weights[s] = 0.0;
            else { cl_weights[s] = 1.0 / Z;  total_z += 1.0 / Z; }
        }
        if (cl_weights.total() == 0.0)
            throw Exception("Boosted_Stumps::train_iteration_fair: zero weight");
        
        cl_weights.normalize();
        
        Committee result(model.feature_space(), model.predicted());
        for (unsigned i = 0;  i < all_trained.size();  ++i)
            result.add(make_sp(all_trained[i].make_copy()), cl_weights[i]);
        
        return make_sp(result.make_copy());
    }
    
}

std::shared_ptr<Classifier_Impl>
Stump_Generator::
generate(Thread_Context & context,
         const Training_Data & training_set,
         const boost::multi_array<float, 2> & weights,
         const std::vector<Feature> & features_,
         float & Z,
         int recursion) const
{
    vector<Feature> features = features_;

    boost::timer timer;

    Feature predicted = model.predicted();

    Stump stumps(training_set.feature_space(), predicted);

    if (committee_size == 1) {
        Stump current
            = train_weighted(context, training_set, weights, features_);
        Z = current.Z;
        return make_sp(current.make_copy());
    }
    else {
        /* Train a committee of stumps. */
        vector<Stump> all_trained
            = train_all(context, training_set, weights, features_);

        /* Work out the weights.  This depends upon the 1/Z score. */
        distribution<float> cl_weights(all_trained.size());
        float total_z = 0.0;
        for (unsigned s = 0;  s < all_trained.size();  ++s) {
            float Z = all_trained[s].Z;
            if (Z < 1e-5) cl_weights[s] = 0.0;
            else { cl_weights[s] = 1.0 / Z;  total_z += 1.0 / Z; }
        }
        if (cl_weights.total() == 0.0)
            throw Exception("Boosted_Stumps::train_iteration_fair: zero weight");
        
        cl_weights.normalize();
        
        Committee result(model.feature_space(), model.predicted());
        for (unsigned i = 0;  i < all_trained.size();  ++i)
            result.add(make_sp(all_trained[i].make_copy()), cl_weights[i]);

        Z = all_trained.front().Z;

        return make_sp(result.make_copy());
    }
}

Stump
Stump_Generator::
get_bias(const Training_Data & data,
         const boost::multi_array<float, 2> & weights,
         const Feature & predicted,
         int trace, Stump::Update update_alg)
{
    size_t nx = data.example_count();
    
    distribution<float> example_weights(nx, 1.0);

    /* Allow the user to specify in which order we try the features. */
    vector<Feature> features(1, MISSING_FEATURE);
    //size_t nf = 1;
    
    /* Check for binary symmetric.  We can optimise various things in this
       case. */
    bool bin_sym = is_bin_sym(weights, data, predicted, features);
    if (bin_sym) example_weights /= 2.0;

    int nl = data.label_count(predicted);

    if (nl == 1) {
        /* Regression problem. */

        typedef W_regress W;
        typedef Z_regress Z;
        typedef C_regress C;

        typedef Bias_Accum<W, Z, C, Stream_Tracer> Accum;
        typedef Stump_Trainer<W, Z, Stream_Tracer> Trainer;
        
        Accum accum(*data.feature_space(), nl, C(), trace);
        Trainer trainer(trace);
        
        W default_w = trainer.calc_default_w(data, predicted, example_weights,
                                             weights);
        trainer.test(MISSING_FEATURE, data, predicted, weights, example_weights,
                     default_w, accum);

        return accum.result(data, predicted);
    }
    else {
        /* If we have a single dimensional weights array, we need to expand
           it. */
        boost::multi_array<float, 2> weights_ = weights;  // shallow copy
        
        if (weights.shape()[1] == 1) {
            weights_.resize(boost::extents[weights.shape()[0]][nl]);
            for (unsigned x = 0;  x < nx;  ++x)
                for (unsigned l = 0;  l < nl;  ++l)
                    weights_[x][l] = weights[x][0];
        }
        
        typedef W_normal W;
        typedef Z_normal Z;
        typedef C_any C;

        typedef Bias_Accum<W, Z, C, Stream_Tracer> Accum;
        typedef Stump_Trainer<W, Z, Stream_Tracer> Trainer;
        
        Accum accum(*data.feature_space(), nl, update_alg, trace);
        Trainer trainer(trace);
        
        W default_w = trainer.calc_default_w(data, predicted, example_weights,
                                             weights);
        trainer.test(MISSING_FEATURE, data, predicted, weights, example_weights,
                     default_w, accum);
        
        return accum.result(data, predicted);
    }
}

std::vector<Stump>
Stump_Generator::
train_all(Thread_Context & context,
          const Training_Data & data,
          const boost::multi_array<float, 2> & weights,
          const std::vector<Feature> & features_) const
{
    size_t nx = data.example_count();
    size_t nl = model.label_count();

    const Feature_Space & feature_space = *model.feature_space();
    
    /* Skip out the low weights, if we were asked to. */
    vector<pair<int, float> > reweighted(nx);
    for (unsigned i = 0;  i < nx;  ++i) {
        reweighted[i].first = i;
        for (unsigned l = 0;  l < weights.shape()[1];  ++l)
            reweighted[i].second += weights[i][l]; 
    }
    //sort_on_second_descending(reweighted);

    double max_weight = ignore_highest / nx;

    distribution<float> example_weights(nx);
    double kept = 0.0, left = 0.0;
    for (unsigned i = 0;  i < nx;  ++i) {
        //if (i >= ignore_highest * nx) {
        if (reweighted[i].second <= max_weight || ignore_highest <= 1.0) {
            example_weights[reweighted[i].first] = 1.0;
            kept += reweighted[i].second;
        }
        else left += reweighted[i].second;
    }

    //cerr << "kept = " << kept << " left = " << left << " total = "
    //     << kept + left << endl;

    //cerr << "ignore_highest = " << ignore_highest << endl;
    //cerr << "example_weights.total() = " << example_weights.total() << endl;
    example_weights /= kept;
    //cerr << "example_weights: min = " << example_weights.min()
    //     << " max = " << example_weights.max() << " total = "
    //     << example_weights.total() << endl;


    /* Allow the user to specify in which order we try the features. */
    vector<Feature> features = features_;

    /* Test the given proportion of possible features, and return the best. */
    if (feature_prop < 1.0) {
        vector<Feature> features2;
        features2.reserve((size_t)(features.size() * (feature_prop + 0.1) + 10));

        size_t nf = (size_t)((float)features.size() * feature_prop);

        if (nf == 0)
            throw Exception("Stump::train_weighted: attempt to train with no "
                            "features (or feature_prop too low)");
        
        if (features.size() < 100) nf = features.size();
        
        double norm = 1.0 / (1 << 24);
        unsigned mask = (1 << 24) - 1;
        //cerr << "norm = " << norm << " mask = " << mask << endl;

        for (unsigned i = 0;  i < features.size();  ++i)
            if ((rand() & mask) * norm < feature_prop)
                features2.push_back(features[i]);

        //cerr << "features.size() = " << features.size()
        //     << " features2.size() = " << features2.size()
        //     << " nf = " << nf << " feature_prop = " << feature_prop << endl;

        features.swap(features2);
    }

    /* Check for binary symmetric.  We can optimise various things in this
       case. */
    bool bin_sym = is_bin_sym(weights, data, model.predicted(), features);
    if (bin_sym) example_weights /= 2.0;

    vector<Stump> all_best;

    bool fair = true;

    //cerr << "features.size() = " << features.size() << endl;

    //PROFILE_FUNCTION(t_train_stump);

    //cerr << "nl = " << nl << " bin_sym = " << bin_sym << " trace = " << trace
    //     << endl;
    
    static Worker_Task & worker = Worker_Task::instance(num_threads() - 1);

    if (nl == 1) {
        /* Regression problem. */

        typedef W_regress W;
        typedef Z_regress Z;
        typedef C_regress C;

        typedef Stump_Accum<W, Z, C, Stream_Tracer, Locked> Accum;
        typedef Stump_Trainer_Parallel<W, Z, Stream_Tracer> Trainer;
        
        Accum accum(feature_space, fair, committee_size, C(), trace);
        Trainer trainer(trace, worker);
        
        Trainer::Test_All_Job<Accum, LW_Array<const float>, distribution<float> >
            job(features, data, model.predicted(), weights,
                example_weights, accum, trainer,
                NO_JOB, context.group());
        
        // Wait until we have finished
        worker.run_until_finished(job.group);
        
        all_best = accum.results(data, model.predicted());
    }
    else if (nl > 11) {
        /* If we have a single dimensional weights array, we need to expand
           it. */
        boost::multi_array<float, 2> weights_ = weights;  // shallow copy
        
        if (weights.shape()[1] == 1) {
            weights_.resize(boost::extents[weights.shape()[0]][nl]);
            for (unsigned x = 0;  x < nx;  ++x)
                for (unsigned l = 0;  l < nl;  ++l)
                    weights_[x][l] = weights[x][0];
        }
        
        //typedef W_multi<float> W;
        //typedef Z_multi<float> Z;
        typedef W_multi<double> W;
        typedef Z_multi<double> Z;
        //typedef W_normal W;
        //typedef Z_normal Z;
        typedef C_any C;
        
        typedef Stump_Accum<W, Z, C, Stream_Tracer, Locked> Accum;
        typedef Stump_Trainer_Parallel<W, Z, Stream_Tracer> Trainer;
        Accum accum(feature_space, fair, committee_size, update_alg, trace);
        Trainer trainer(trace, worker);

        Trainer::Test_All_Job<Accum, LW_Array<const float>, distribution<float> >
            job(features, data, model.predicted(), weights,
                example_weights, accum, trainer,
                NO_JOB, context.group());
        
        // Wait until we have finished
        worker.run_until_finished(job.group);
            
        all_best = accum.results(data, model.predicted());
    }
    else if (!bin_sym) {
        /* If we have a single dimensional weights array, we need to expand
           it. */
        boost::multi_array<float, 2> weights_ = weights;
        if (weights.shape()[1] == 1) {
            weights_.resize(boost::extents[weights.shape()[0]][nl]);
            for (unsigned x = 0;  x < nx;  ++x)
                for (unsigned l = 0;  l < nl;  ++l)
                    weights_[x][l] = weights[x][0];
        }

        typedef W_normal W;
        typedef Z_normal Z;
        typedef C_any C;

        typedef Stump_Accum<W, Z, C, Stream_Tracer, Locked> Accum;
        typedef Stump_Trainer_Parallel<W, Z, Stream_Tracer> Trainer;
        
        //cerr << "trace = " << trace << endl;

        Accum accum(feature_space, fair, committee_size, update_alg, trace);
        Trainer trainer(trace, worker);
        
        //cerr << "trainer = " << &trainer << endl;
        //cerr << "accum.tracer() = " << accum.tracer.operator bool() << endl;
        //cerr << "trainer.tracer() = " << trainer.tracer.operator bool() << endl;

        Trainer::Test_All_Job<Accum, LW_Array<const float>, distribution<float> >
            job(features, data, model.predicted(), weights,
                example_weights, accum, trainer,
                NO_JOB, context.group());
            
        // Wait until we have finished
        worker.run_until_finished(job.group);
        
        //cerr << "done training" << endl;

        all_best = accum.results(data, model.predicted());

#if 0
        if (feature_prop == 1.0)
            trainer.test_all_and_sort(features, data, model.predicted(), weights_,
                                      example_weights, accum);
        else trainer.test_all(features, data, model.predicted(), weights_,
                              example_weights, accum);
#endif
    }
    else {
        /* Use the optimised versions for binary symmetric. */
        typedef W_binsym W;
        typedef Z_binsym Z;
        typedef C_any C;
        
        if (trace) {
            typedef Stump_Accum<W, Z, C, Stream_Tracer, Locked> Accum;
            typedef Stump_Trainer_Parallel<W, Z, Stream_Tracer> Trainer;
            
            Accum accum(feature_space, fair, committee_size, update_alg, trace);
            Trainer trainer(trace, worker);
            
            Trainer::Test_All_Job<Accum, LW_Array<const float>,
                                  distribution<float> >
                job(features, data, model.predicted(), weights,
                    example_weights, accum, trainer,
                    NO_JOB, context.group());
            
            // Wait until we have finished
            worker.run_until_finished(job.group);
            
            all_best = accum.results(data, model.predicted());
        }
        else {
            typedef Stump_Accum<W, Z, C, No_Trace, Locked> Accum;
            typedef Stump_Trainer_Parallel<W, Z, No_Trace> Trainer;
            
            Accum accum(feature_space, fair, committee_size, update_alg);
            Trainer trainer(worker);

            Trainer::Test_All_Job<Accum, LW_Array<const float>, distribution<float> >
                job(features, data, model.predicted(), weights,
                    example_weights, accum, trainer,
                    NO_JOB, context.group());
            
            // Wait until we have finished
            worker.run_until_finished(job.group);
            
            all_best = accum.results(data, model.predicted());
        }
    }
    
    /* If we asked for the n highest features, then we remove those with a
       Z score too low. */
    if (committee_size > 1) {
        std::sort(all_best.begin(), all_best.end(), Sort_Z());
        while (all_best.size() > committee_size
               && !z_equal(all_best[committee_size - 1].Z, all_best.back().Z))
            all_best.pop_back();
    }

    return all_best;
}

Stump
Stump_Generator::
train_weighted(Thread_Context & context,
               const Training_Data & data,
               const boost::multi_array<float, 2> & weights,
               const std::vector<Feature> & features) const
{
    uint64_t before = ticks();

    size_t nl = model.label_count();

    vector<Stump> all_best
        = train_all(context, data, weights, features);

    /* If we have a feature_proportion of < 1.0, it is possible that
       all_best is empty.  In this case, we don't update anything
       and simply return 1.0 for Z.
    */

    if (all_best.empty()) {
        //throw Exception("Stump::train_weighted(): no stumps trained");
        Stump result = model;
        result.split = Split();
        result.action.pred_true    = distribution<float>(nl);
        result.action.pred_false   = distribution<float>(nl);
        result.action.pred_missing = distribution<float>(nl);
        return result;
    }
    
    /* Choose the one which has the highest feature count.  This way we bias
       it towards choosing features which occur more often.  This seems to
       add significantly to the overall accuracy. */
    vector<int> all_highest;
    int highest_count = 0;

    for (unsigned i = 0;  i < all_best.size();  ++i) {
        int count = data.index().count(all_best[i].split.feature());
        if (count > highest_count) {
            all_highest.clear();
            all_highest.push_back(i);
            highest_count = count;
        }
    }
    
    assert(!all_highest.empty());

    ticks_train_weighted += (ticks() - before);
    
    if (all_highest.size() == 1)
        return all_best[all_highest[0]];
    
    else {
        /* Choose the one with the smallest printed feature, so that we are
           stable. */
        int best = -1;
        string best_name;

        for (unsigned i = 0;  i < all_highest.size();  ++i) {
            string name = all_best[all_highest[i]].print();
            if (best == -1 || name < best_name) {
                name = best_name;
                best = i;
            }
        }

        return all_best[best];
    }
}


/*****************************************************************************/
/* REGISTRATION                                                              */
/*****************************************************************************/

namespace {

Register_Factory<Classifier_Generator, Stump_Generator>
    STUMP_REGISTER("stump");

} // file scope

} // namespace ML

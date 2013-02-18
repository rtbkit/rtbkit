/* boosted_stumps_generator.cc
   Jeremy Barnes, 15 March 2006
   Copyright (c) 2006 Jeremy Barnes  All rights reserved.
   $Source$

   Generator for boosted stumps.
*/

#include "boosted_stumps_generator.h"
#include "registry.h"
#include <boost/timer.hpp>
#include <boost/progress.hpp>
#include "training_index.h"
#include "weighted_training.h"
#include "jml/arch/simd_vector.h"
#include "boosted_stumps_impl.h"
#include "boosting_core.h"
#include "boosting_core_parallel.h"
#include "stump_predict.h"
#include "binary_symmetric.h"
#include "jml/arch/tick_counter.h"
#include "jml/utils/smart_ptr_utils.h"
#include <boost/scoped_ptr.hpp>
#include "stump_predict.h"

using namespace std;


namespace ML {

namespace {
uint64_t bin_sym_ticks = 0, weak_learner_ticks = 0, update_ticks = 0;

#if 0
struct Prof {
    ~Prof()
    {
        cerr << format("bin_sym:       %14ld (%8.5fs)\n", bin_sym_ticks,
                       bin_sym_ticks * seconds_per_tick); 
        cerr << format("weak learner:  %14ld (%8.5fs)\n",
                       weak_learner_ticks,
                       weak_learner_ticks * seconds_per_tick); 
        cerr << format("update:        %14ld (%8.5fs)\n", update_ticks,
                       update_ticks * seconds_per_tick); 
    }
} prof;
#endif

} // file scope


/*****************************************************************************/
/* BOOSTED_STUMPS_GENERATOR                                                  */
/*****************************************************************************/

Boosted_Stumps_Generator::
Boosted_Stumps_Generator()
{
    defaults();
}

Boosted_Stumps_Generator::~Boosted_Stumps_Generator()
{
}

void
Boosted_Stumps_Generator::
configure(const Configuration & config)
{
    Early_Stopping_Generator::configure(config);
    
    config.find(max_iter,             "max_iter");
    config.find(min_iter,             "min_iter");
    config.find(true_only,            "true_only");
    config.find(fair,                 "fair");
    config.find(cost_function,        "cost_function");
    config.find(output_function,      "output_function");
    config.find(short_circuit_window, "short_circuit_window");
    config.find(trace_training_acc,   "trace_training_acc");
}

void
Boosted_Stumps_Generator::
defaults()
{
    Early_Stopping_Generator::defaults();
    max_iter = 500;
    min_iter = 10;
    true_only = false;
    fair = false;
    cost_function = CF_EXPONENTIAL;
    output_function = Boosted_Stumps::RAW;
    short_circuit_window = 0;
}

Config_Options
Boosted_Stumps_Generator::
options() const
{
    Config_Options result = Early_Stopping_Generator::options();
    result
        .add("min_iter", min_iter, "1-max_iter",
             "minimum number of training iterations to run")
        .add("max_iter", max_iter, ">=min_iter",
             "maximum number of training iterations to run")
        .add("true_only", true_only,
             "don't allow missing predicates to infer labels")
        .add("fair", fair,
             "treat multiple equivalent features in a symmetric manner")
        .add("cost_function", cost_function,
             "select cost function for boosting weight update")
        .add("output_function", output_function,
             "select output function of classifier")
        .add("short_circuit_window", short_circuit_window, "0-",
             "short circuit (stop) training if no improvement for N iter "
             "(0 off)")
        .add("trace_training_acc", trace_training_acc,
             "trace the accuracy of the training set as well as validation")
        .add(weak_learner.options());

    return result;
}

void
Boosted_Stumps_Generator::
init(std::shared_ptr<const Feature_Space> fs, Feature predicted)
{
    Classifier_Generator::init(fs, predicted);
    model = Boosted_Stumps(fs, predicted);
    weak_learner.init(fs, predicted);
}

std::shared_ptr<Classifier_Impl>
Boosted_Stumps_Generator::
generate(Thread_Context & context,
         const Training_Data & training_set,
         const Training_Data & validation_set,
         const distribution<float> & training_ex_weights,
         const distribution<float> & validate_ex_weights,
         const std::vector<Feature> & features_, int) const
{
    return make_sp(generate_stumps(context, training_set, validation_set,
                                   training_ex_weights, validate_ex_weights,
                                   features_).make_copy());
}

Boosted_Stumps
Boosted_Stumps_Generator::
generate_stumps(Thread_Context & context,
                const Training_Data & training_set,
                const Training_Data & validation_set,
                const distribution<float> & training_ex_weights,
                const distribution<float> & validate_ex_weights,
                const std::vector<Feature> & features_) const
{
    const Feature_Space & fs = *training_set.feature_space();

    vector<Feature> features = features_;

    boost::timer timer;

    Feature predicted = model.predicted();

    unsigned nl = training_set.label_count(predicted);
    Boosted_Stumps stumps(training_set.feature_space(), predicted);
    Boosted_Stumps best(training_set.feature_space(), predicted);
    float best_acc = 0.0;
    int best_iter = 0;

    bool validate_is_train = false;
    if (validation_set.example_count() == 0
        || ((&validation_set == &training_set)
            && (training_ex_weights.size() == validate_ex_weights.size())
            && !(training_ex_weights != validate_ex_weights).any()))
        validate_is_train = true;

#if 0    
    cerr << "validate_is_train = " << validate_is_train << endl;

    cerr << "training_ex_weights.total() = " << training_ex_weights.total()
         << endl;
    cerr << "validate_ex_weights.total() = " << validate_ex_weights.total()
         << endl;
    
    if (&training_set == &validation_set)
        cerr << "training data is validate data" << endl;

    if (training_ex_weights.size() == validate_ex_weights.size())
        cerr << "tv dot product = "
             << (training_ex_weights * validate_ex_weights).total()
             << endl;

    size_t nz_train = std::count_if(training_ex_weights.begin(),
                                    training_ex_weights.end(),
                                    std::bind2nd(std::greater<float>(), 0.0));

    size_t nz_val = std::count_if(validate_ex_weights.begin(),
                                  validate_ex_weights.end(),
                                  std::bind2nd(std::greater<float>(), 0.0));
    
    cerr << nz_train << " non-zero training weights, "
         << nz_val << " non-zero validate weights" << endl;
#endif

    //cerr << "training_ex_weights = " << training_ex_weights << endl;
    //training_ex_weights = 1.0;
    //training_ex_weights.normalize();

    //cerr << "true_only = " << true_only << endl;
    //cerr << "trace = " << trace << endl;
    stumps.output = output_function;

    boost::multi_array<float, 2> training_output
        (boost::extents[training_set.example_count()][nl]);

    boost::multi_array<float, 2> validation_output
        (boost::extents[validation_set.example_count()][nl]);
    
    boost::multi_array<float, 2> weights
        = expand_weights(training_set, training_ex_weights, predicted);
    
    boost::scoped_ptr<boost::progress_display> progress;

    if (verbosity == 1 || verbosity == 2) {
        cerr << "training " << max_iter << " iterations..." << endl;
        progress.reset(new boost::progress_display(max_iter, cerr));
    }
    
    if (min_iter > max_iter)
        throw Exception("min_iter is greater than max_iter");

    if (verbosity > 2) {
        cerr << "  it";
        if (trace_training_acc) cerr << "   train";
        cerr << "     val    Z   ";
        if (nl == 2) {
            if (fs.info(predicted).categorical()) {
                for (unsigned i = 0;  i < 2;  ++i)
                    cerr << format("%6s",
                                   fs.info(predicted).categorical()->print(i).c_str());
                cerr << " miss";
            }
            else cerr << "false  true  miss";
        }
        else if (nl <= 5) {
            if (fs.info(predicted).categorical()) {
                for (unsigned i = 0;  i < std::min(nl, 5U);  ++i)
                    cerr << format("%6s",
                                   fs.info(predicted).categorical()->print(i).c_str());
            }
            else {
                for (unsigned i = 0;  i < std::min(nl, 5U);  ++i)
                    cerr << format("lbl%d  ", i);
            }
        }
        else if (verbosity > 3)
            for (unsigned i = 0;  i < nl;  ++i)
                cerr << format("lbl%-3d", i);
        cerr << "       arg feature" << endl;
    }
    
    double validate_acc = 0.0;
    double train_acc = 0.0;

    for (unsigned i = 0;  i < max_iter;  ++i) {

        if (progress) ++(*progress);
        //vector<ML::Feature> features;

        if (!fair) {

            Stump stump;
            Optimization_Info opt_info;

            if (validate_is_train || trace_training_acc) {
                stump
                    = train_iteration(context,
                                      training_set, weights, features, stumps,
                                      training_output, training_ex_weights,
                                      train_acc, opt_info);
            }
            else {
                stump
                    = train_iteration(context,
                                      training_set, weights, features, stumps,
                                      opt_info);
            }
            
            if (validate_is_train) validate_acc = train_acc;
            else {
                validate_acc
                    = update_accuracy(context,
                                      stump, opt_info, validation_set, features,
                                      validation_output, validate_ex_weights);
            }

            if (verbosity > 2) {
                cerr << format("%4d", i);
                if (trace_training_acc)
                    cerr << format(" %6.2f%% ", train_acc * 100.0);
                cerr << format("%6.2f%% ",
                               validate_acc * 100.0);
                cerr << stump.summary();
                cerr << endl;
            }
        }
        else {
            vector<Optimization_Info> opt_infos;
            vector<Stump> all_stumps
                = train_iteration_fair(context,
                                       training_set, weights, features, stumps,
                                       opt_infos);

            update_scores(training_output, training_set, all_stumps,
                          opt_infos,
                          context.group());
            update_scores(validation_output, validation_set, all_stumps,
                          opt_infos,
                          context.group());
            
            float train_acc
                = accuracy(training_output, training_set,
                           predicted, training_ex_weights);
            validate_acc
                = accuracy(validation_output, validation_set,
                           predicted, validate_ex_weights);
            
            if (verbosity > 2) {
                cerr << format("%4d %6.2f%% %6.2f%% ",
                               i, train_acc * 100.0, validate_acc * 100.0);

                for (unsigned i = 0;  i < all_stumps.size();  ++i) {
                    if (i != 0) cerr << "                     ";
                    const Stump & stump = all_stumps[i];
                    cerr << stump.summary() << endl;
                    if (i != all_stumps.size() - 1) cerr << ",";
                    cerr << endl;
                }
            }
        }
        
        if (validate_acc > best_acc && i >= min_iter) {
            best = stumps;
            best_acc = validate_acc;
            best_iter = i;
        }

        if (short_circuit_window > 0 && i >= min_iter
            && best_iter > 0 && i > best_iter + short_circuit_window) {
            cerr << "no improvement for " << short_circuit_window
                 << " iterations; short circuiting" << endl;
            break;
        }
    }
    
    if (profile)
        cerr << "training time: " << timer.elapsed() << "s" << endl;
    
    if (verbosity > 0) {
        cerr << format("best was %6.2f%% on iteration %d", best_acc * 100.0,
                       best_iter)
             << endl;
    }

    return best;
}

std::shared_ptr<Classifier_Impl>
Boosted_Stumps_Generator::
generate_and_update(Thread_Context & context,
                    const Training_Data & training_set,
                    boost::multi_array<float, 2> & weights,
                    const std::vector<Feature> & features_) const
{
    const Feature_Space & fs = *training_set.feature_space();

    vector<Feature> features = features_;

    boost::timer timer;

    unsigned nl = training_set.label_count(predicted);

    float best_acc = 0.0;
    int best_iter = 0;

    boost::multi_array<float, 2> training_output
        (boost::extents[training_set.example_count()][nl]);

    if (weights.shape()[0] != training_set.example_count()
        || weights.shape()[1] != nl)
        throw Exception("Boosting_Generator::generate_and_update(): "
                        "weights have the wrong shape");

    distribution<float> training_ex_weights(training_set.example_count(), 1.0);

    boost::scoped_ptr<boost::progress_display> progress;

    if (verbosity == 1 || verbosity == 2) {
        cerr << "training " << max_iter << " iterations..." << endl;
        progress.reset(new boost::progress_display(max_iter, cerr));
    }
    
    if (verbosity > 2) {
        cerr << "  it";
        cerr << "   train    Z   ";
        if (nl == 2) {
            if (fs.info(predicted).categorical()) {
                for (unsigned i = 0;  i < 2;  ++i)
                    cerr << format("%6s",
                                   fs.info(predicted).categorical()->print(i).c_str());
                cerr << " miss";
            }
            else cerr << "false  true  miss";
        }
        else if (nl <= 5) {
            if (fs.info(predicted).categorical()) {
                for (unsigned i = 0;  i < std::min(nl, 5U);  ++i)
                    cerr << format("%6s",
                                   fs.info(predicted).categorical()->print(i).c_str());
            }
            else {
                for (unsigned i = 0;  i < std::min(nl, 5U);  ++i)
                    cerr << format("lbl%d  ", i);
            }
        }
        else if (verbosity > 3)
            for (unsigned i = 0;  i < nl;  ++i)
                cerr << format("lbl%-3d", i);
        cerr << "       arg feature" << endl;
    }
    
    Boosted_Stumps stumps(feature_space, predicted);
    Boosted_Stumps best = stumps;

    double train_acc = 0.0;

    for (unsigned i = 0;  i < max_iter;  ++i) {

        if (progress) ++(*progress);

        Optimization_Info opt_info;

        Stump stump
            = train_iteration(context, training_set, weights, features, stumps,
                              opt_info);

        float Z = stump.Z;

        if (verbosity > 2) {
            cerr << format("%4d", i);
            cerr << format(" %6.2f%% ",
                           train_acc * 100.0);
            cerr << stump.summary() << endl;
        }
        
        if (train_acc > best_acc && i >= min_iter) {
            best_iter = i;
            best_acc = train_acc;
            best = stumps;
        }

        if (1.0 - Z <= 1e-5)
            break;  // too much capacity; we've learned non-zero weights
                    // perfectly
    }
    
    if (verbosity > 0) {
        cerr << format("best was %6.2f%% on iteration %d", best_acc * 100.0,
                       best_iter)
             << endl;
    }
    
    if (profile)
        cerr << "training time: " << timer.elapsed() << "s" << endl;
    
    std::shared_ptr<Boosted_Stumps>
        result(new Boosted_Stumps(stumps));
    
    return result;
}

std::vector<Stump>
Boosted_Stumps_Generator::
train_iteration_fair(Thread_Context & context,
                     const Training_Data & data,
                     boost::multi_array<float, 2> & weights,
                     std::vector<Feature> & features,
                     Boosted_Stumps & result,
                     vector<Optimization_Info> & opt_infos) const
{
    Feature predicted = model.predicted();

    bool bin_sym
        = convert_bin_sym(weights, data, predicted, features);

    vector<Stump> all_trained
        = weak_learner.train_all(context, data, weights, features);

    opt_infos.resize(all_trained.size());

    const Feature_Space & fs = *data.feature_space();

    if (fs.type() == DENSE) {
        for (unsigned i = 0;  i < opt_infos.size();  ++i) {
            opt_infos[i] = all_trained[i].optimize(fs.dense_features());
        }
    }


    /* Work out the weights.  This depends upon the 1/Z score. */
    distribution<float> cl_weights(all_trained.size());
    float total_z = 0.0;
    for (unsigned s = 0;  s < all_trained.size();  ++s) {
        float Z = all_trained[s].Z;
        if (Z < 1e-5) cl_weights[s] = 0.0;
        else { cl_weights[s] = 1.0 / Z;  total_z += 1.0 / Z; }
    }
    if (cl_weights.total() == 0.0)
        throw Exception("Boosted_Stumps_Generator::train_iteration_fair: "
                        "zero weight");
    
    /* Get the average Z score, which is needed by the logistic update. */
    //float avg_z = total_z / cl_weights.total();

    cl_weights.normalize();

    /* Insert it */
    result.insert(all_trained, cl_weights);

    update_weights(weights, all_trained, opt_infos, cl_weights, data,
                   cost_function, bin_sym, context.group());
    
    return all_trained;
}

Stump
Boosted_Stumps_Generator::
train_iteration(Thread_Context & context,
                const Training_Data & data,
                boost::multi_array<float, 2> & weights, vector<Feature> & features,
                Boosted_Stumps & result,
                Optimization_Info & opt_info) const
{
    //PROFILE_FUNCTION(t_train);
    Feature predicted = model.predicted();

    bool bin_sym
        = convert_bin_sym(weights, data, predicted, features);

    /* Make sure we have some features. */
    if (features.empty()) features = data.all_features();

    /* Find the best stump */
    Stump stump = weak_learner.train_weighted(context, data, weights, features);

    /* Insert it */
    result.insert(stump);

    /* Update the d distribution. */
    double total = 0.0;

    size_t nl = weights.shape()[1];

    static Worker_Task & task
        = Worker_Task::instance(num_threads() - 1);
    
    if (cost_function == CF_EXPONENTIAL) {
        typedef Boosting_Loss Loss;
        if (bin_sym) {
            typedef Binsym_Updater<Loss> Updater;
            typedef Update_Weights_Parallel<Updater> Update;
            Update update(task);
            
            update(stump,opt_info, 1.0, weights, data, total,
                   NO_JOB, context.group());
            task.run_until_finished(update.group);
        }
        else {
            typedef Normal_Updater<Loss> Updater;
            typedef Update_Weights_Parallel<Updater> Update;
            Updater updater(nl);
            Update update(task, updater);

            update(stump, opt_info, 1.0, weights, data, total,
                   NO_JOB, context.group());
            task.run_until_finished(update.group);
        }
    }
    else if (cost_function == CF_LOGISTIC) {
        typedef Logistic_Loss Loss;
        Loss loss(stump.Z);

        if (bin_sym) {
            typedef Binsym_Updater<Loss> Updater;
            typedef Update_Weights_Parallel<Updater> Update;
            Updater updater(loss);
            Update update(task, updater);

            update(stump, opt_info, 1.0, weights, data, total,
                   NO_JOB, context.group());
            task.run_until_finished(update.group);
        }
        else {
            //PROFILE_FUNCTION(t_update);
            typedef Normal_Updater<Loss> Updater;
            typedef Update_Weights_Parallel<Updater> Update;
            Updater updater(nl, loss);
            Update update(task, updater);

            update(stump, opt_info, 1.0, weights, data, total,
                   NO_JOB, context.group());
            task.run_until_finished(update.group);
        }
    }
    else throw Exception("Boosted_Stumps_Generator::train_iteration: "
                         "unknown cost function");

    //cerr << "Z = " << stump.Z << " total = " << total << endl;

    float * start = weights.data();
    float * finish = start + weights.num_elements();
    SIMD::vec_scale(start, 1.0 / total, start, finish - start);
    
    return stump;
}

/* This does the following: 
   - Trains a stump
   - Updates the weights
   - Updates the output weights
   - Calculates the accuracy over the training set

   all in one pass (which saves on execution time as less memory is used).
*/

Stump
Boosted_Stumps_Generator::
train_iteration(Thread_Context & context,
                const Training_Data & data,
                boost::multi_array<float, 2> & weights,
                vector<Feature> & features,
                Boosted_Stumps & result,
                boost::multi_array<float, 2> & output,
                const distribution<float> & ex_weights,
                double & training_accuracy,
                Optimization_Info & opt_info) const
{
    size_t ticks_before = ticks();

    bool bin_sym
        = convert_bin_sym(weights, data, predicted, features);

    convert_bin_sym(output, data, predicted, features);

    bin_sym_ticks += (ticks() - ticks_before);
    ticks_before = ticks();

    /* Find the best stump */
    Stump stump = weak_learner.train_weighted(context, data, weights, features);

    if (data.feature_space()->type() == DENSE)
        opt_info = stump.optimize(data.feature_space()->dense_features());

    /* Insert it */
    result.insert(stump);

    weak_learner_ticks += (ticks() - ticks_before);
    ticks_before = ticks();

    /* Update the d distribution. */
    double total = 0.0;
    double correct = 0.0;

    size_t nx = data.example_count();
    if (nx != output.shape()[0])
        throw Exception("update_scores: example counts don't match");

    typedef Normal_Updater<Boosting_Predict> Output_Updater;
    Output_Updater output_updater(nl);

    static Worker_Task & task
        = Worker_Task::instance(num_threads() - 1);

    if (cost_function == CF_EXPONENTIAL) {
        typedef Boosting_Loss Loss;
        if (bin_sym) {
            typedef Binsym_Updater<Loss> Weights_Updater;
            typedef Update_Weights_And_Scores_Parallel
                <Weights_Updater, Output_Updater, Binsym_Scorer>
                Update;
            Weights_Updater weights_updater;
            Update update(task, weights_updater, output_updater);
            
            update(stump, opt_info, 1.0, weights, output, data, ex_weights,
                   correct, total, NO_JOB, context.group());
            task.run_until_finished(update.group);
        }
        else {
            typedef Normal_Updater<Loss> Weights_Updater;
            typedef Update_Weights_And_Scores_Parallel
                <Weights_Updater, Output_Updater, Normal_Scorer>
                Update;
            Weights_Updater weights_updater(nl);
            Update update(task, weights_updater, output_updater);

            update(stump, opt_info, 1.0, weights, output, data, ex_weights,
                   correct, total, NO_JOB, context.group());
            task.run_until_finished(update.group);
        }
    }
    else if (cost_function == CF_LOGISTIC) {
        typedef Logistic_Loss Loss;
        Loss loss(stump.Z);

        if (bin_sym) {
            typedef Binsym_Updater<Loss> Weights_Updater;
            typedef Update_Weights_And_Scores_Parallel
                <Weights_Updater, Output_Updater, Binsym_Scorer>
                Update;
            Weights_Updater weights_updater(loss);
            Update update(task, weights_updater, output_updater);

            update(stump, opt_info, 1.0, weights, output, data, ex_weights,
                   correct, total, NO_JOB, context.group());
            task.run_until_finished(update.group);
        }
        else {
            typedef Normal_Updater<Loss> Weights_Updater;
            typedef Update_Weights_And_Scores_Parallel
                <Weights_Updater, Output_Updater, Normal_Scorer>
                Update;
            Weights_Updater weights_updater(nl, loss);
            Update update(task, weights_updater, output_updater);

            update(stump, opt_info, 1.0, weights, output, data, ex_weights,
                   correct, total, NO_JOB, context.group());
            task.run_until_finished(update.group);
        }
    }
    else throw Exception("Boosted_Stumps_Generator::train_iteration: "
                         "unknown cost function");

    training_accuracy = correct / ex_weights.total();

    //cerr << "total = " << total << " Z = " << stump.Z << endl;

    float * start = weights.data();
    float * finish = start + weights.num_elements();
    SIMD::vec_scale(start, 1.0 / total, start, finish - start);

    update_ticks += (ticks() - ticks_before);

    return stump;
}


/* This does the following: 
   - Trains a stump
   - Updates the weights
   - Updates the output weights
   - Calculates the accuracy over the training set

   all in one pass (which saves on execution time as less memory is used).
*/

double
Boosted_Stumps_Generator::
update_accuracy(Thread_Context & context,
                const Stump & stump,
                const Optimization_Info & opt_info,
                const Training_Data & data,
                const vector<Feature> & features,
                boost::multi_array<float, 2> & output,
                const distribution<float> & ex_weights) const
{
    size_t ticks_before = ticks();

    bool bin_sym
        = convert_bin_sym(output, data, predicted, features);

    bin_sym_ticks += (ticks() - ticks_before);
    ticks_before = ticks();

    double correct = 0.0;

    static Worker_Task & task
        = Worker_Task::instance(num_threads() - 1);

    if (bin_sym) {
        typedef Binsym_Updater<Boosting_Predict> Output_Updater;
        Output_Updater output_updater;
        
        typedef Update_Scores_Parallel<Output_Updater, Binsym_Scorer> Update;
        Update update(task, output_updater);
        
        update(stump, opt_info, 1.0, output, data, ex_weights, correct,
               NO_JOB, context.group());
        task.run_until_finished(update.group);
    }
    else {
        typedef Normal_Updater<Boosting_Predict> Output_Updater;
        Output_Updater output_updater(nl);
        
        typedef Update_Scores_Parallel<Output_Updater, Normal_Scorer> Update;
        Update update(task, output_updater);
        
        update(stump, opt_info, 1.0, output, data, ex_weights, correct,
               NO_JOB, context.group());
        task.run_until_finished(update.group);
    }
    
    update_ticks += (ticks() - ticks_before);

    return correct / ex_weights.total();
}


/*****************************************************************************/
/* REGISTRATION                                                              */
/*****************************************************************************/

namespace {

Register_Factory<Classifier_Generator, Boosted_Stumps_Generator>
    BOOSTED_STUMPS_REGISTER("boosted_stumps");

} // file scope

} // namespace ML

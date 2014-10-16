/* boosting_generator.cc
   Jeremy Barnes, 15 March 2006
   Copyright (c) 2006 Jeremy Barnes  All rights reserved.
   $Source$

   Generator for boosted stumps.
*/

#include "boosting_generator.h"
#include "registry.h"
#include <boost/timer.hpp>
#include <boost/progress.hpp>
#include "training_index.h"
#include "weighted_training.h"
#include "committee.h"
#include "jml/utils/sgi_numeric.h"
#include "boosting_training.h"
#include "jml/arch/simd_vector.h"
#include "boosting_core.h"
#include "boosting_core_parallel.h"
#include "config_impl.h"
#include "binary_symmetric.h"
#include "jml/utils/worker_task.h"
#include "jml/utils/guard.h"
#include <boost/scoped_ptr.hpp>


using namespace std;


namespace ML {

/*****************************************************************************/
/* BOOSTING_GENERATOR                                                        */
/*****************************************************************************/

Boosting_Generator::
Boosting_Generator()
{
    defaults();
}

Boosting_Generator::~Boosting_Generator()
{
}

void
Boosting_Generator::
configure(const Configuration & config)
{
    Early_Stopping_Generator::configure(config);

    config.find(max_iter,             "max_iter");
    config.find(min_iter,             "min_iter");
    config.find(cost_function,        "cost_function");
    config.find(short_circuit_window, "short_circuit_window");
    config.find(trace_training_acc,   "trace_training_acc");

    weak_learner = get_trainer("weak_learner", config);
}

void
Boosting_Generator::
defaults()
{
    Early_Stopping_Generator::defaults();
    max_iter = 500;
    min_iter = 10;
    cost_function = CF_EXPONENTIAL;
    short_circuit_window = 0;
    weak_learner.reset();
    trace_training_acc = false;
}

Config_Options
Boosting_Generator::
options() const
{
    Config_Options result = Early_Stopping_Generator::options();
    result
        .add("min_iter", min_iter, "1-max_iter",
             "minimum number of training iterations to run")
        .add("max_iter", max_iter, ">=min_iter",
             "maximum number of training iterations to run")
        .add("cost_function", cost_function,
             "select cost function for boosting weight update")
        .add("short_circuit_window", short_circuit_window, "0-",
             "short circuit (stop) training if no improvement for N iter "
             "(0 off)")
        .add("trace_training_acc", trace_training_acc,
             "trace the accuracy of the training set as well as validation")
        .subconfig("weak_leaner", weak_learner,
                   "weak learner that produces each bag");

    if (weak_learner) result.add(weak_learner->options());
    
    return result;
}

void
Boosting_Generator::
init(std::shared_ptr<const Feature_Space> fs, Feature predicted)
{
    Classifier_Generator::init(fs, predicted);
    weak_learner->init(fs, predicted);
}

std::shared_ptr<Classifier_Impl>
Boosting_Generator::
generate(Thread_Context & context,
         const Training_Data & training_set,
         const Training_Data & validation_set,
         const distribution<float> & training_ex_weights,
         const distribution<float> & validate_ex_weights,
         const std::vector<Feature> & features_, int) const
{
    vector<Feature> features = features_;

    boost::timer timer;

    unsigned nl = training_set.label_count(predicted);

    float best_acc = 0.0;
    int best_iter = 0;

    bool validate_is_train = false;
    if (validation_set.example_count() == 0
        || ((&validation_set == &training_set)
            && (training_ex_weights.size() == validate_ex_weights.size())
            && !(training_ex_weights != validate_ex_weights).any()))
        validate_is_train = true;

    boost::multi_array<float, 2> training_output
        (boost::extents[training_set.example_count()][nl]);

    boost::multi_array<float, 2> validation_output
        (boost::extents[validation_set.example_count()][nl]);
    
    boost::multi_array<float, 2> weights
        = expand_weights(training_set, training_ex_weights, predicted);

    boost::multi_array<float, 2> last_weights = weights;

    boost::scoped_ptr<boost::progress_display> progress;

    if (verbosity == 1 || verbosity == 2) {
        cerr << "training " << max_iter << " iterations..." << endl;
        progress.reset(new boost::progress_display(max_iter, cerr));
    }
    
    if (min_iter > max_iter)
        throw Exception("min_iter is greater than max_iter");

    if (verbosity > 2) {
        cerr << "  it   ";
        if (trace_training_acc) cerr << "train     ";
        cerr << "val   Z    Classifier" << endl;
    }
    
    vector<std::shared_ptr<Classifier_Impl> > classifiers;

    double validate_acc = 0.0;
    double train_acc = 0.0;

    for (unsigned i = 0;  i < max_iter;  ++i) {

        if (progress) ++(*progress);

        float Z;

        std::shared_ptr<Classifier_Impl> weak_classifier;
        Optimization_Info opt_info;

        if (validate_is_train || trace_training_acc)
            weak_classifier
                = train_iteration(context, training_set, weights, features,
                                  training_output, training_ex_weights,
                                  train_acc, Z, opt_info);
        else
            weak_classifier
                = train_iteration(context, training_set, weights, features,
                                  Z, opt_info);
        
        if (validate_is_train) validate_acc = train_acc;
        else
            validate_acc
                = update_accuracy(context, *weak_classifier, opt_info,
                                  validation_set, features,
                                  validation_output, validate_ex_weights);

#if 0
        float min_weight = 1.0, max_weight = 0.0, max_diff = 0.0;
        int where_max_diff = -1, where_max_weight = -1;

        int nx = training_set.example_count();

        double total = 0.0;

        for (unsigned x = 0;  x < nx;  ++x) {
            float old_w = last_weights[x][0];
            float new_w = weights[x][0];

            min_weight = std::min(min_weight, new_w);

            if (new_w > max_weight) {
                max_weight = new_w;
                where_max_weight = x;
            }

            float diff = fabs(old_w - new_w);
            if (diff > max_diff) {
                where_max_diff = x;
                max_diff = diff;
            }

            total += new_w;
        }

        cerr << "total = " << total << endl;

        cerr << "min_weight " << min_weight << " max_weight "
             << max_weight << " where " << where_max_weight
             << " old " << last_weights[where_max_weight][0]
             << " new " << weights[where_max_weight][0]
             << " pred " << weak_classifier->predict(0,
                                                     training_set[where_max_weight])
             << endl;

        cerr << "max_diff " << max_diff << " where " << where_max_diff
             << " old " << last_weights[where_max_diff][0]
             << " new " << weights[where_max_diff][0]
             << " pred " << weak_classifier->predict(0,
                                                     training_set[where_max_diff])
             << endl;

        cerr << "ratio = " << max_weight / min_weight << endl;

        last_weights.resize(boost::extents[weights.shape()[0]][weights.shape()[1]]);
        last_weights = weights;
#endif

        if (verbosity > 2) {
            cerr << format("%4d", i);
            if (trace_training_acc)
                cerr << format(" %6.2f%%", train_acc * 100.0);
            cerr << format(" %6.2f%% %5.3f ", validate_acc * 100.0, Z);
            cerr << weak_classifier->summary();
            cerr << endl;
        }

        classifiers.push_back(weak_classifier);

        if (validate_acc > best_acc && i >= min_iter) {
            best_iter = i;
            best_acc = validate_acc;
        }

        if (Z == 1.0f) {
            cerr << "Z = " << format("%12f", Z) << endl;
            cerr << "stopping due to perfect learning" << endl;
            break;  // too much capacity; we've learned non-zero weights
                    // perfectly
        }
    }

    if (verbosity > 0) {
        cerr << format("best was %6.2f%% on iteration %d", best_acc * 100.0,
                       best_iter)
             << endl;
    }

    if (profile)
        cerr << "training time: " << timer.elapsed() << "s" << endl;
    
    std::shared_ptr<Committee>
        result(new Committee(feature_space, predicted));
    
    for (int i = 0;  i <= best_iter;  ++i)
        result->add(classifiers[i]);
    
    return result;
}

std::shared_ptr<Classifier_Impl>
Boosting_Generator::
generate_and_update(Thread_Context & context,
                    const Training_Data & training_set,
                    boost::multi_array<float, 2> & weights,
                    const std::vector<Feature> & features_) const
{
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
    
    if (verbosity > 2)
        cerr << "  it   train     Z    Classifier" << endl;
    
    vector<std::shared_ptr<Classifier_Impl> > classifiers;

    double train_acc = 0.0;

    for (unsigned i = 0;  i < max_iter;  ++i) {

        if (progress) ++(*progress);

        float Z;

        Optimization_Info opt_info;

        std::shared_ptr<Classifier_Impl> weak_classifier
            = train_iteration(context, training_set, weights, features,
                              training_output, training_ex_weights,
                              train_acc, Z, opt_info);
        
        if (verbosity > 2) {
            cerr << format("%4d", i);
            cerr << format(" %6.2f%% %5.3f ", train_acc * 100.0, Z);
            cerr << weak_classifier->summary();
            cerr << endl;
        }
        
        classifiers.push_back(weak_classifier);

        if (train_acc > best_acc && i >= min_iter) {
            best_iter = i;
            best_acc = train_acc;
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
    
    std::shared_ptr<Committee>
        result(new Committee(feature_space, predicted));
    
    for (int i = 0;  i <= best_iter;  ++i)
        result->add(classifiers[i]);
    
    return result;
}

std::shared_ptr<Classifier_Impl>
Boosting_Generator::
train_iteration(Thread_Context & context,
                const Training_Data & data,
                boost::multi_array<float, 2> & weights,
                vector<Feature> & features,
                float & Z,
                Optimization_Info & opt_info) const
{
    bool bin_sym = convert_bin_sym(weights, data, predicted, features);

    /* Make sure we have some features. */
    if (features.empty()) features = data.all_features();

    Z = 0.0;
    
    /* Find the best stump */
    std::shared_ptr<Classifier_Impl> weak_classifier
        = weak_learner->generate(context, data, weights, features, Z);
    
    const Feature_Space & fs = *data.feature_space();

    if (fs.type() == DENSE)
        opt_info = weak_classifier->optimize(fs.dense_features());

    /* Update the d distribution. */
    double total = 0.0;
    
    size_t nl = weights.shape()[1];
    
    static Worker_Task & task = Worker_Task::instance(num_threads() - 1);
    
    if (cost_function == CF_EXPONENTIAL) {
        typedef Boosting_Loss Loss;
        if (bin_sym) {
            if (true /* use parallel */) {
                typedef Binsym_Updater<Loss> Updater;
                typedef Update_Weights_Parallel<Updater> Update;
                Update update(task);
                
                update(*weak_classifier, opt_info, 1.0, weights, data, total,
                       NO_JOB, context.group());
                task.run_until_finished(update.group);
            }
            else {
                typedef Binsym_Updater<Loss> Updater;
                typedef Update_Weights<Updater> Update;
                Update update;
                
                total = update(*weak_classifier, opt_info, 1.0, weights, data);
            }
        }
        else {
            typedef Normal_Updater<Loss> Updater;
            typedef Update_Weights_Parallel<Updater> Update;
            Updater updater(nl);
            Update update(task, updater);

            update(*weak_classifier, opt_info, 1.0, weights, data, total,
                   NO_JOB, context.group());
            task.run_until_finished(update.group);
        }
    }
    else if (cost_function == CF_LOGISTIC) {
        throw Exception("Boosting_Generator::train_iteration(): "
                        "what is Z for logistic loss?");
#if 0
        typedef Logistic_Loss Loss;
        Loss loss(Z);

        if (bin_sym) {
            typedef Binsym_Updater<Loss> Updater;
            typedef Update_Weights_Parallel<Updater> Update;
            Updater updater(loss);
            Update update(task, updater);

            update(*weak_classifier, opt_info, 1.0, weights, data, total,
                   NO_JOB, parent);
            task.run_until_finished(update.group);
        }
        else {
            //PROFILE_FUNCTION(t_update);
            typedef Normal_Updater<Loss> Updater;
            typedef Update_Weights_Parallel<Updater> Update;
            Updater updater(nl, loss);
            Update update(task, updater);

            update(*weak_classifier, opt_info, 1.0, weights, data, total,
                   NO_JOB, parent);
            task.run_until_finished(update.group);
        }
#endif
    }
    else throw Exception("Boosting_Generator::train_iteration: "
                         "unknown cost function");

    //cerr << "total = " << total << " Z = " << Z << endl;

    if (Z == 0.0) Z = total;

    if (Z > 1.0) {
        cerr << "weak learner returned Z of " << Z << endl;
        //cerr << weak_classifier->print() << endl;
    }
    
    float * start = weights.data();
    float * finish = start + weights.num_elements();
    SIMD::vec_scale(start, 1.0 / total, start, finish - start);
    
    return weak_classifier;
}

/* This does the following: 
   - Trains a stump
   - Updates the weights
   - Updates the output weights
   - Calculates the accuracy over the training set

   all in one pass (which saves on execution time as less memory is used).
*/

std::shared_ptr<Classifier_Impl>
Boosting_Generator::
train_iteration(Thread_Context & context,
                const Training_Data & data,
                boost::multi_array<float, 2> & weights,
                std::vector<Feature> & features,
                boost::multi_array<float, 2> & output,
                const distribution<float> & ex_weights,
                double & training_accuracy, float & Z,
                Optimization_Info & opt_info) const
{
    bool bin_sym
        = convert_bin_sym(weights, data, predicted, features);

    //cerr << "bin_sym = " << bin_sym << endl;

    convert_bin_sym(output, data, predicted, features);

    Z = 0.0;
    
    /* Find the best weak learner */
    std::shared_ptr<Classifier_Impl> weak_classifier
        = weak_learner->generate(context, data, weights, features, Z);

    const Feature_Space & fs = *data.feature_space();

    if (fs.type() == DENSE)
        opt_info = weak_classifier->optimize(fs.dense_features());

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
            
            update(*weak_classifier, opt_info,
                   1.0, weights, output, data, ex_weights,
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

            update(*weak_classifier, opt_info,
                   1.0, weights, output, data, ex_weights,
                   correct, total, NO_JOB, context.group());
            task.run_until_finished(update.group);
        }
    }
    else if (cost_function == CF_LOGISTIC) {
        throw Exception("Boosting_Generator::train_iteration(): "
                        "what is Z for logistic loss?");
#if 0
        typedef Logistic_Loss Loss;
        Loss loss(stump.Z);

        if (bin_sym) {
            typedef Binsym_Updater<Loss> Weights_Updater;
            typedef Update_Weights_And_Scores_Parallel
                <Weights_Updater, Output_Updater, Binsym_Scorer>
                Update;
            Weights_Updater weights_updater(loss);
            Update update(task, weights_updater, output_updater);

            update(stump, opt_info,
                   1.0, weights, output, data, ex_weights,
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

            update(stump, opt_info,
                   1.0, weights, output, data, ex_weights,
                   correct, total, NO_JOB, context.group());
            task.run_until_finished(update.group);
        }
#endif
    }
    else throw Exception("Boosted_Stumps_Generator::train_iteration: "
                         "unknown cost function");

    if (Z == 0.0) Z = total;

    //cerr << "total = " << total << " Z = " << Z << endl;

    training_accuracy = correct / ex_weights.total();

    float * start = weights.data();
    float * finish = start + weights.num_elements();
    SIMD::vec_scale(start, 1.0 / total, start, finish - start);
    
    return weak_classifier;
}


/* This does the following: 
   - Trains a stump
   - Updates the weights
   - Updates the output weights
   - Calculates the accuracy over the training set

   all in one pass (which saves on execution time as less memory is used).
*/

double
Boosting_Generator::
update_accuracy(Thread_Context & context,
                const Classifier_Impl & weak_classifier,
                const Optimization_Info & opt_info,
                const Training_Data & data,
                const vector<Feature> & features,
                boost::multi_array<float, 2> & output,
                const distribution<float> & ex_weights) const
{
    bool bin_sym
        = convert_bin_sym(output, data, predicted, features);

    double correct = 0.0;

    static Worker_Task & task
        = Worker_Task::instance(num_threads() - 1);

    if (bin_sym) {
        typedef Binsym_Updater<Boosting_Predict> Output_Updater;
        Output_Updater output_updater;
        
        typedef Update_Scores_Parallel<Output_Updater, Binsym_Scorer> Update;
        Update update(task, output_updater);
        
        update(weak_classifier, opt_info,
               1.0, output, data, ex_weights, correct,
               NO_JOB, context.group());
        task.run_until_finished(update.group);
    }
    else {
        typedef Normal_Updater<Boosting_Predict> Output_Updater;
        Output_Updater output_updater(nl);
        
        typedef Update_Scores_Parallel<Output_Updater, Normal_Scorer> Update;
        Update update(task, output_updater);
        
        update(weak_classifier, opt_info,
               1.0, output, data, ex_weights, correct,
               NO_JOB, context.group());

        task.run_until_finished(update.group);
    }
    
    return correct / ex_weights.total();
}


/*****************************************************************************/
/* REGISTRATION                                                              */
/*****************************************************************************/

namespace {

Register_Factory<Classifier_Generator, Boosting_Generator>
    BOOSTING_REGISTER("boosting");

} // file scope

} // namespace ML

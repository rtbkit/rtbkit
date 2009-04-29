/* perceptron_generator.cc
   Jeremy Barnes, 15 March 2006
   Copyright (c) 2006 Jeremy Barnes  All rights reserved.
   $Source$

   Generator for perceptrons.
*/

#include "perceptron_generator.h"
#include "registry.h"
#include <boost/timer.hpp>
#include <boost/progress.hpp>
#include "training_index.h"
#include "stats/distribution_simd.h"
#include "arch/simd_vector.h"
#include "algebra/lapack.h"
#include "algebra/matrix_ops.h"
#include "algebra/irls.h"
#include <iomanip>
#include "config_impl.h"
#include "utils/environment.h"
#include "utils/profile.h"
#include "worker_task.h"
#include "utils/guard.h"
#include "evaluation.h"
#include <boost/scoped_ptr.hpp>
#include <boost/bind.hpp>
#include "utils/smart_ptr_utils.h"

using namespace std;


namespace ML {


namespace {

Env_Option<bool> profile("PROFILE_PERCEPTRON", false);

double t_train = 0.0, t_decorrelate = 0.0;
double t_cholesky = 0.0, t_qr = 0.0, t_gs = 0.0, t_mean = 0.0, t_covar = 0.0;

struct Stats {
    ~Stats()
    {
        if (profile) {
            cerr << "perceptron profile: " << endl;
            cerr << "  decorrelate:    " << t_decorrelate << "s" << endl;
            cerr << "    qr:           " << t_qr          << "s" << endl;
            cerr << "    gram schmidt: " << t_gs          << "s" << endl;
            cerr << "    mean:         " << t_mean        << "s" << endl;
            cerr << "    covar         " << t_covar       << "s" << endl;
            cerr << "    cholesky:     " << t_cholesky    << "s" << endl;
            cerr << "  train:          " << t_train       << "s" << endl;
        }
    }
} stats;

} // file scope


/*****************************************************************************/
/* PERCEPTRON_GENERATOR                                                      */
/*****************************************************************************/

Perceptron_Generator::
Perceptron_Generator()
{
    defaults();
}

Perceptron_Generator::~Perceptron_Generator()
{
}

void
Perceptron_Generator::
configure(const Configuration & config)
{
    Early_Stopping_Generator::configure(config);
    
    config.find(max_iter, "max_iter");
    config.find(min_iter, "min_iter");
    config.find(learning_rate, "learning_rate");
    config.find(arch_str, "arch");
}

void
Perceptron_Generator::
defaults()
{
    Early_Stopping_Generator::defaults();
    max_iter = 100;
    min_iter = 10;
    learning_rate = 0.01;
    arch_str = "%i";
    activation = Perceptron::TANH;
}

Config_Options
Perceptron_Generator::
options() const
{
    Config_Options result = Early_Stopping_Generator::options();
    result
        .add("min_iter", min_iter, "1-max_iter",
             "minimum number of training iterations to run")
        .add("max_iter", max_iter, ">=min_iter",
             "maximum number of training iterations to run")
        .add("learning_rate", learning_rate, "0<=%n<=1",
             "rate of learning relative to dataset size")
        .add("arch", arch_str, "(see doc)",
             "hidden unit specification; %i=in vars, %o=out vars; eg 5_10")
        .add("activation", activation,
             "activation function for neurons");

    return result;
}

void
Perceptron_Generator::
init(boost::shared_ptr<const Feature_Space> fs, Feature predicted)
{
    Early_Stopping_Generator::init(fs, predicted);
    model = Perceptron(fs, predicted);
}

boost::shared_ptr<Classifier_Impl>
Perceptron_Generator::
generate(Thread_Context & context,
         const Training_Data & training_set,
         const Training_Data & validation_set,
         const distribution<float> & training_ex_weights,
         const distribution<float> & validate_ex_weights,
         const std::vector<Feature> & features, int) const
{
    boost::timer timer;

    Feature predicted = model.predicted();

    Perceptron current(model);
    Perceptron best(model);
    
    float best_acc = 0.0;
    int best_iter = 0;

    bool validate_is_train = false;
    if (validation_set.example_count() == 0
        || &validation_set == &training_set) 
        validate_is_train = true;

    cerr << "validate_is_train = " << validate_is_train << endl;

    boost::scoped_ptr<boost::progress_display> progress;

    log("perceptron_generator", 1)
        << "training " << max_iter << " iterations..." << endl;

    if (verbosity < 3)
        progress.reset(new boost::progress_display
                       (max_iter, log("perceptron_generator", 1)));
    
    vector<int> arch = Perceptron::parse_architecture(arch_str);
    
    boost::multi_array<float, 2> decorrelated
        = init(training_set, features, arch, current);

    boost::multi_array<float, 2> val_decorrelated
        (boost::extents[validation_set.example_count()][decorrelated.shape()[1]]);
    if (validate_is_train) val_decorrelated = decorrelated;
    else val_decorrelated = current.decorrelate(validation_set);

    size_t nx = decorrelated.shape()[0];
    size_t nf = decorrelated.shape()[1];

    log("perceptron_generator", 1)
        << current.parameters() << " parameters, "
        << nx << " examples, " << nx * nf << " training values" << endl;
    
    if (min_iter > max_iter)
        throw Exception("min_iter is greater than max_iter");

    log("perceptron_generator", 3)
        << "  it   train     val    best     diff" << endl;
    
    float validate_acc = 0.0;
    float train_acc = 0.0;

    const std::vector<Label> & labels
        = training_set.index().labels(predicted);
    const std::vector<Label> & val_labels
        = validation_set.index().labels(predicted);

    for (unsigned i = 0;  i < max_iter;  ++i) {

        if (progress) ++(*progress);

        if (validate_is_train) validate_acc = train_acc;
        else validate_acc
                 = current.accuracy(val_decorrelated, val_labels,
                                    validate_ex_weights);

        double last_best_acc = best_acc;

        if (validate_acc > best_acc || i == min_iter) {
            best = current;
            best_acc = validate_acc;
            best_iter = i;
        }
                
        train_acc = train_iteration(context, decorrelated, labels, training_ex_weights,
                                    current);
        
        log("perceptron_generator", 3)
            << format("%4d %6.2f%% %6.2f%% %6.2f%% %+7.3f%%",
                      i, train_acc * 100.0, validate_acc * 100.0,
                      best_acc * 100.0, (validate_acc - last_best_acc) * 100.0)
            << endl;
        
        log("perceptron_generator", 5) << current.print() << endl;
    }
    
    if (profile)
        log("perceptron_generator", 1)
            << "training time: " << timer.elapsed() << "s" << endl;
    
    log("perceptron_generator", 1)
        << format("best was %6.2f%% on iteration %d", best_acc * 100.0,
                  best_iter)
        << endl;

    log("perceptron_generator", 1)
        << "best accuracy: " << best.accuracy(training_set) << " train, "
        << best.accuracy(validation_set) << " validate" << endl;
    
    log("perceptron_generator", 4) << best.print() << endl;
    
    return make_sp(best.make_copy());
}

namespace {

boost::multi_array<float, 2>
cholesky(const boost::multi_array<double, 2> & A_)
{
    PROFILE_FUNCTION(t_cholesky);

    if (A_.shape()[0] != A_.shape()[1])
        throw Exception("cholesky: matrix isn't square");
    
    int n = A_.shape()[0];

    boost::multi_array<float, 2> A(boost::extents[n][n]);
    std::copy(A_.begin(), A_.end(), A.begin());
    
    int res = LAPack::potrf('U', n, A.data(), n);
    
    if (res < 0)
        throw Exception(format("cholesky: potrf: argument %d was illegal", -res));
    else if (res > 0)
        throw Exception(format("cholesky: potrf: leading minor %d not positive "
                               "definite", res));
    
    for (unsigned i = 0;  i < n;  ++i)
        std::fill(&A[i][0] + i + 1, &A[i][0] + n, 0.0);

    return A;

    //cerr << "residuals = " << endl << (A * transpose(A)) - A_ << endl;

    //boost::multi_array<float, 2> result(n, n);
    //std::copy(A.data_begin(), A.data_end(), result.data_begin());

    //return result;
}

boost::multi_array<float, 2>
lower_inverse(const boost::multi_array<float, 2> & A)
{
    if (A.shape()[0] != A.shape()[1])
        throw Exception("lower_inverse: matrix isn't square");
    
    int n = A.shape()[0];

    boost::multi_array<float, 2> L = A;
    
    for (int j = 0;  j < n;  ++j) {
        L[j][j] = 1.0 / L[j][j];

        for (int i = j + 1;  i < n;  ++i) {
            double sum = 0.0;
            for (unsigned k = j;  k < i;  ++k)
                sum -= L[i][k] * L[k][j];
            L[i][j] = sum / L[i][i];
        }
    }

    //cerr << "L * A = " << endl << L * A << endl;

    return L;
}

} // file scope

/** Decorrelates the training data, returning a dense decorrelated dataset. */
boost::multi_array<float, 2>
Perceptron_Generator::
decorrelate(const Training_Data & data,
            const std::vector<Feature> & possible_features,
            Perceptron & result) const
{
    PROFILE_FUNCTION(t_decorrelate);

    const Dataset_Index & index = data.index();

    vector<Feature> & features = result.features;
    features.clear();
    
    /* Figure out if we can keep each feature or not. */
    
    for (unsigned i = 0;  i < possible_features.size();  ++i) {
        const Feature & feature = possible_features[i];

        if (feature == result.predicted()) continue;  // don't use label as a feature!

        /* Find out information about the feature from the training data. */
        if (!index.dense(feature)) {
            cerr << "feature " << result.feature_space()->print(feature)
                 << " skipped due to missing values" << endl;
            continue;
        }
        else if (!index.exactly_one(feature)) {
            cerr << "feature " << result.feature_space()->print(feature)
                 << " skipped due to more than one value" << endl;
            continue;
        }
        else if (index.constant(feature)) {
            cerr << "feature " << result.feature_space()->print(feature)
                 << " skipped as it is constant over the dataset" << endl;
            continue;
        }
        else {
            float min, max;
            boost::tie(min, max) = index.range(feature);
            if (abs(min) > 1e10 || abs(max) > 1e10) {
                cerr << "feature " << result.feature_space()->print(feature)
                     << " skipped as its range is too large: "
                     << min << " to " << max << endl;
                continue;
            }
        }
        features.push_back(feature);
    }

    std::sort(features.begin(), features.end());
    
    size_t nx = data.example_count();
    size_t nf = features.size();

    /* Get a dense matrix of the features. */
    boost::multi_array<float, 2> input(boost::extents[nx][nf + 1]);
    
    for (unsigned x = 0;  x < nx;  ++x) {
        result.extract_features(data[x], &input[x][0]);
        input[x][nf] = 1.0;  // bias term
    }

    //cerr << "input.shape()[0] = " << input.shape()[0] << endl;
    //cerr << "input.shape()[1] = " << input.shape()[1] << endl;
    boost::multi_array<float, 2> inputt = transpose(input);

    //cerr << "inputt.shape()[0] = " << inputt.shape()[0] << endl;
    //cerr << "inputt.shape()[1] = " << inputt.shape()[1] << endl;
    vector<distribution<float> > dependent;
    vector<int> permutations(nf + 1, 0);
    
#if 0
    boost::multi_array<float, 2> input2 = inputt;

    /* Factorize the matrix with partial pivoting.  This allows us to find the
       largest number of linearly independent columns possible. */

    float tau[nf];
    vector<int> permutations(nf + 1, 0);
    permutations[nf] = 1;  // make bias be a leading column

    {
        PROFILE_FUNCTION(t_qr);
        int res = LAPack::geqp3(nx, nf + 1, inputt.data_begin(), inputt.shape()[1],
                                &permutations[0],
                                tau);
        
        if (res != 0)
            throw Exception(format("geqp3: error in parameter %d", -res));
    }
    
    cerr << "permutations = " << permutations << endl;
    for (unsigned i = 0;  i <= nf;  ++i)
        cerr << "r[" << i << "][" << i << "] = " << inputt[i][i] << endl;

    /* Check for linearly dependent columns. */
    inputt = input2;
#endif
    
    {
        PROFILE_FUNCTION(t_gs);
        permutations = remove_dependent_impl(inputt, dependent, 1e-4);
    }
    
    /* Find which features are left */
    vector<Feature> new_features;
    for (unsigned i = 0;  i < features.size();  ++i) {
        if (permutations[i] != -1)
            new_features.push_back(features[i]);
    }
    
    for (unsigned i = 0;  i < features.size();  ++i) {
        if (permutations[i] == -1) {
            cerr << "feature "
                 << result.feature_space()->print(features[i])
                 << " removed as it can be calculated as ";
            cerr << result.feature_space()->print(features[i])
                 << " = ";
            bool first = true;
            for (unsigned j = 0;  j < dependent[i].size();  ++j) {
                float v = dependent[i][j];
                if (abs(v) < 0.0001) continue;

                if (v < 0.0) {
                    cerr << " - ";
                }
                else {
                    if (first) ;
                    else cerr << " + ";
                }
                
                v = abs(v);
                if (abs(v - 1.0) < 0.0001) ;
                else cerr << setprecision(4) << v << " ";
                
                if (j == new_features.size()) ;
                else cerr << result.feature_space()->print(new_features[j]);
                
                first = false;
            }
            cerr << endl;
        }
    }

    /* Calculate the covariance matrix over those that are left.  Note that
       we could decorrelate them with the orthogonalization that we did above,
       but then we wouldn't be able to save it as a covariance matrix.
    */
    features.swap(new_features);
    nf = features.size();

    distribution<double> mean(nf, 0.0);
    {
        PROFILE_FUNCTION(t_mean);
        for (unsigned f = 0;  f < nf;  ++f) {
            mean[f] = SIMD::vec_sum_dp(&inputt[f][0], nx) / nx;
            for (unsigned x = 0;  x < nx;  ++x)
                inputt[f][x] -= mean[f];
        }
    }

    boost::multi_array<double, 2> covar(boost::extents[nf][nf]);
    {
        PROFILE_FUNCTION(t_covar);
        for (unsigned f = 0;  f < nf;  ++f) {
            for (unsigned f2 = 0;  f2 <= f;  ++f2)
                covar[f][f2] = covar[f2][f]
                    = SIMD::vec_dotprod_dp(&inputt[f][0], &inputt[f2][0], nx) / nx;
        }
    }

    /* Do the cholevsky stuff */
    boost::multi_array<float, 2> transform
        = transpose(lower_inverse(cholesky(covar)));

    /* Finally, we add a layer.  This will perform both the removal of the
       mean (via the biasing) and the decorrelation (via the application
       of the matrix).
    */

    /* y = (x - mean) * A;
         = (x * A) - (mean * A);
    */

    Perceptron::Layer layer;
    layer.weights.resize(boost::extents[transform.shape()[0]][transform.shape()[1]]);
    layer.weights = transform;
    layer.bias = distribution<float>(nf, 0.0);  // already have mean removed
    layer.activation = Perceptron::IDENTITY;
    
    boost::multi_array<float, 2> decorrelated(boost::extents[nx][nf]);
    float fv_in[nf];

    for (unsigned x = 0;  x < nx;  ++x) {
        for (unsigned f = 0;  f < nf;  ++f)
            fv_in[f] = inputt[f][x];

        layer.apply(&fv_in[0], &decorrelated[x][0]);
    }

    layer.bias = (transform * mean) * -1.0;  // now add the bias
    result.layers.clear();
    result.add_layer(layer);
    
    return decorrelated;
}

namespace {

struct Training_Job_Info {
    const boost::multi_array<float, 2> & decorrelated;
    const std::vector<Label> & labels;
    const distribution<float> & example_weights;
    const Perceptron & result;
    Lock lock;
    vector<boost::multi_array<float, 2> > & weight_updates;
    vector<distribution<float> > & bias_updates;
    double & correct;
    double & total;

    Training_Job_Info(const boost::multi_array<float, 2> & decorrelated,
                      const std::vector<Label> & labels,
                      const distribution<float> & example_weights,
                      const Perceptron & result,
                      vector<boost::multi_array<float, 2> > & weight_updates,
                      vector<distribution<float> > & bias_updates,
                      double & correct, double & total)
        : decorrelated(decorrelated), labels(labels),
          example_weights(example_weights), result(result),
          weight_updates(weight_updates), bias_updates(bias_updates),
          correct(correct), total(total)
    {
    }

    void train(int x_start, int x_end)
    {
        const vector<Perceptron::Layer> & layers = result.layers;
        size_t max_units = result.max_units;

        vector<distribution<float> > layer_outputs(layers.size());

        size_t nl = layers.size();

        // Accumulate the weight updates here.  They are applied later.
        vector<boost::multi_array<float, 2> > sub_weight_updates;
        vector<distribution<float> > sub_bias_updates(nl);
    
        for (unsigned l = 0;  l < nl;  ++l) {
            size_t no = layers[l].outputs();
            size_t ni = layers[l].inputs();
            layer_outputs[l].resize(no);
            sub_weight_updates
                .push_back(boost::multi_array<float, 2>(boost::extents[ni][no]));
            sub_bias_updates[l].resize(no);
        }
        
        double sub_correct = 0.0, sub_total = 0.0;

        //size_t nx = decorrelated.shape()[0];
        size_t nf = decorrelated.shape()[1];
        size_t no = layers.back().outputs(); // num outputs

        float learning_rate = 0.5;
        
        distribution<float> correct(no, 0.0);

        const float fire = 1.0, inhibit = -1.0;
        
        float errors[max_units], delta[max_units];
        
        for (unsigned x = x_start;  x < x_end;  ++x) {
            float w = example_weights[x];
            
            if (w == 0.0) continue;
            
            /* Compute weights going forward. */
            std::copy(&decorrelated[x][0], &decorrelated[x][0] + nf,
                      &layer_outputs[0][0]);
            
            for (unsigned l = 1;  l < nl;  ++l)
                layers[l].apply(layer_outputs[l - 1], layer_outputs[l]);
     
            /* Calculate the correctness. */
            Correctness c = correctness(layer_outputs.back().begin(),
                                        layer_outputs.back().end(),
                                        labels[x]);
            sub_correct += w * c.possible * c.correct;
            sub_total += w * c.possible;
       
            /* Calculate the error terms for each output unit. */
            /* TODO: regression */
            correct[labels[x]] = fire;
            
            for (unsigned i = 0;  i < no;  ++i)
                errors[i] = correct[i] - layer_outputs.back()[i];
            
            /* Backpropegate. */
            for (int l = nl - 1;  l >= 1;  --l) {
                
                const Perceptron::Layer & layer = layers[l];
                
                size_t no = layer.outputs();
                size_t ni = layer.inputs();
                
                /* Differentiate the output. */
                layer.deltas(&layer_outputs[l][0], &errors[0], delta);
                
                if (l > 1) {
                    /* Calculate new errors. */
                    for (unsigned i = 0;  i < ni;  ++i)
                        errors[i] = SIMD::vec_dotprod_dp(&delta[0],
                                                        &layer.weights[i][0], no);
                }
                
                /* Update the weights. */
                float k = w * learning_rate;
                for (unsigned i = 0;  i < ni;  ++i) {
                    float k2 = layer_outputs[l - 1][i] * k;
                    SIMD::vec_add(&sub_weight_updates[l][i][0], k2, &delta[0],
                                 &sub_weight_updates[l][i][0], no);
                }
            
                /* Update the bias terms.  The previous layer output (input) is
                   always 1. */
                SIMD::vec_add(&sub_bias_updates[l][0], k, &delta[0],
                             &sub_bias_updates[l][0], no);
            }
        
            correct[labels[x]] = inhibit;
        }

        Guard guard(lock);

        this->correct += sub_correct;
        this->total += sub_total;

        /* Finally, put the accumulated weights back. */
        for (unsigned l = 1;  l < nl;  ++l) {
            const Perceptron::Layer & layer = layers[l];
            
            size_t no = layer.outputs();
            size_t ni = layer.inputs();
            
            for (unsigned i = 0;  i < ni;  ++i)
                SIMD::vec_add(&weight_updates[l][i][0],
                             &sub_weight_updates[l][i][0],
                             &weight_updates[l][i][0], no);
            
            SIMD::vec_add(&bias_updates[l][0], &sub_bias_updates[l][0],
                         &bias_updates[l][0], no);
        }
    }
};

struct Training_Job {
    Training_Job(Training_Job_Info & info,
                 int x_start, int x_end)
        : info(info), x_start(x_start), x_end(x_end)
    {
    }

    Training_Job_Info & info;
    int x_start, x_end;
    
    void operator () () const
    {
        info.train(x_start, x_end);
    }
};

} // file scope

double
Perceptron_Generator::
train_iteration(Thread_Context & context,
                const boost::multi_array<float, 2> & decorrelated,
                const std::vector<Label> & labels,
                const distribution<float> & example_weights,
                Perceptron & result) const
{
    PROFILE_FUNCTION(t_train);

    size_t nx = decorrelated.shape()[0];

    if (!example_weights.empty() && nx != example_weights.size())
        throw Exception("Perceptron::train_iteration(): error propegating");

    vector<Perceptron::Layer> & layers = result.layers;
    size_t nl = layers.size();

    // Accumulate the weight updates here.  They are applied later.
    vector<boost::multi_array<float, 2> > weight_updates;
    vector<distribution<float> > bias_updates(nl);
    double correct = 0.0;
    double total = 0.0;
    
    for (unsigned l = 0;  l < nl;  ++l) {
        size_t no = layers[l].outputs();
        size_t ni = layers[l].inputs();
        weight_updates.push_back(boost::multi_array<float, 2>(boost::extents[ni][no]));
        bias_updates[l].resize(no);
    }

    Training_Job_Info info(decorrelated, labels, example_weights, result,
                           weight_updates, bias_updates, correct, total);

    static Worker_Task & worker = Worker_Task::instance(num_threads() - 1);
    
    int group;
    {
        group = worker.get_group(NO_JOB,
                                 format("Perceptron_Generator::train_iteration "
                                        "under %d", context.group()),
                                 context.group());
        Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                     boost::ref(worker),
                                     group));
        
        /* Do 4096 examples per job. */
        for (size_t x = 0;  x < nx;  x += 4096) {
            size_t end = std::min(x + 4096, nx);
            worker.add(Training_Job(info, x, end),
                       format("Perceptron_Generator::train_iteration(): %zd-%zd "
                              "under %d", x, end, group),
                       group);
        }
    }
    
    worker.run_until_finished(group);

    /* Finally, put the accumulated weights back. */
    for (unsigned l = 1;  l < nl;  ++l) {
        Perceptron::Layer & layer = layers[l];
        
        size_t no = layer.outputs();
        size_t ni = layer.inputs();
        
        for (unsigned i = 0;  i < ni;  ++i)
            SIMD::vec_add(&layer.weights[i][0], &weight_updates[l][i][0],
                         &layer.weights[i][0], no);
        
        SIMD::vec_add(&layer.bias[0], &bias_updates[l][0],
                     &layer.bias[0], no);
    }

    //cerr << "correct = " << correct << " total = " << total << endl;

    return correct / total;
}

boost::multi_array<float, 2>
Perceptron_Generator::
init(const Training_Data & data,
     const std::vector<Feature> & possible_features,
     const std::vector<int> & architecture,
     Perceptron & result) const
{
    result = model;

    /* Find out about the output that we need (in particular, how many
       values it can have). */
    boost::multi_array<float, 2> decorrelated
        = decorrelate(data, possible_features, result);

    Feature_Info pred_info = model.feature_space()->info(model.predicted());
    int nout = pred_info.value_count();
    if (nout == 0) nout = 1;  // regression problem; one output

    int nunits = result.features.size();

    /* Add hidden layers with the specified sizes */
    for (unsigned i = 0;  i < architecture.size();  ++i) {
        int units = architecture[i];
        if (units == -1) units = result.features.size();
        Perceptron::Layer layer(nunits, units, activation);
        result.add_layer(layer);
        nunits = units;
    }
    
    /* Add the output units. */
    Perceptron::Layer layer(nunits, nout, activation);
    result.add_layer(layer);

    return decorrelated;
}



/*****************************************************************************/
/* REGISTRATION                                                              */
/*****************************************************************************/

namespace {

Register_Factory<Classifier_Generator, Perceptron_Generator>
    PERCEPTRON_REGISTER("perceptron");

} // file scope

} // namespace ML

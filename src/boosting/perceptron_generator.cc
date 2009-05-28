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
#include "utils/vector_utils.h"
#include "backprop_cuda.h"

using namespace std;


namespace ML {


namespace {

Env_Option<bool> profile("PROFILE_PERCEPTRON", false);

double t_train = 0.0, t_decorrelate = 0.0;
double t_cholesky = 0.0, t_qr = 0.0, t_gs = 0.0, t_mean = 0.0, t_covar = 0.0;
double t_update = 0.0, t_fprop = 0.0, t_bprop = 0.0, t_zero = 0.0;
double t_setup = 0.0;

struct Stats {
    ~Stats()
    {
        if (profile) {
            cerr << "perceptron training profile: " << endl;
            cerr << "  decorrelate:    " << t_decorrelate << "s" << endl;
            cerr << "    qr:           " << t_qr          << "s" << endl;
            cerr << "    gram schmidt: " << t_gs          << "s" << endl;
            cerr << "    mean:         " << t_mean        << "s" << endl;
            cerr << "    covar         " << t_covar       << "s" << endl;
            cerr << "    cholesky:     " << t_cholesky    << "s" << endl;
            cerr << "  train:          " << t_train       << "s" << endl;
            cerr << "    setup:        " << t_setup       << "s" << endl;
            cerr << "    update:       " << t_update      << "s" << endl;
            cerr << "    fprop:        " << t_fprop       << "s" << endl;
            cerr << "    bprop:        " << t_bprop       << "s" << endl;
            cerr << "    zero:         " << t_zero        << "s" << endl;
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
    config.find(batch_size, "batch_size");
    config.find(activation, "activation");
    config.find(do_decorrelate, "decorrelate");
    config.find(use_cuda, "use_cuda");
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
    activation = ACT_TANH;
    do_decorrelate = true;
    batch_size = 1024;
    use_cuda = false;
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
             "activation function for neurons")
        .add("decorrelate", do_decorrelate,
             "decorrelate the features before training")
        .add("batch_size", batch_size, "0.0-1.0 or 1 - nvectors",
             "number of samples in each \"mini batch\" for stochastic")
        .add("use_cuda", use_cuda, "boolean", "use the CUDA optimized kernel");
    
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
        << "  it   train     rmse     val    best     diff" << endl;
    
    float validate_acc = 0.0;
    float train_acc = 0.0;
    float rms_error = 0.0;

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
                
        
        boost::tie(train_acc, rms_error)
            = train_iteration(context, decorrelated, labels,
                              training_ex_weights,
                              current);
        
        log("perceptron_generator", 3)
            << format("%4d %6.2f%% %8.6f %6.2f%% %6.2f%% %+7.3f%%",
                      i, train_acc * 100.0, rms_error, validate_acc * 100.0,
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

boost::multi_array<double, 2>
cholesky(const boost::multi_array<double, 2> & A_)
{
    PROFILE_FUNCTION(t_cholesky);

    if (A_.shape()[0] != A_.shape()[1])
        throw Exception("cholesky: matrix isn't square");
    
    int n = A_.shape()[0];

    boost::multi_array<double, 2> A(boost::extents[n][n]);
    std::copy(A_.begin(), A_.end(), A.begin());
    
    int res = LAPack::potrf('U', n, A.data(), n);
    
    if (res < 0)
        throw Exception(format("cholesky: potrf: argument %d was illegal", -res));
    else if (res > 0)
        throw Exception(format("cholesky: potrf: leading minor %d of %d "
                               "not positive definite", res, n));
    
    for (unsigned i = 0;  i < n;  ++i)
        std::fill(&A[i][0] + i + 1, &A[i][0] + n, 0.0);

    return A;

#if 0
    //cerr << "residuals = " << endl << (A * transpose(A)) - A_ << endl;

    boost::multi_array<float, 2> result(boost::extents[n][n]);
    std::copy(A.begin(), A.end(), result.begin());

    return result;
#endif
}

template<typename Float>
boost::multi_array<Float, 2>
lower_inverse(const boost::multi_array<Float, 2> & A)
{
    if (A.shape()[0] != A.shape()[1])
        throw Exception("lower_inverse: matrix isn't square");
    
    int n = A.shape()[0];

    boost::multi_array<Float, 2> L = A;
    
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

    cerr << "decorrelate: " << possible_features.size() << " features at input"
         << endl;
    
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
            cerr << "feature " << i << " ("
                 << result.feature_space()->print(feature)
                 << ") skipped due to more than one value" << endl;
            continue;
        }
        else if (index.constant(feature)) {
            cerr << "feature " << i << " ("
                 << result.feature_space()->print(feature)
                 << ") skipped as it is constant over the dataset" << endl;
            continue;
        }
        else {
            float min, max;
            boost::tie(min, max) = index.range(feature);
            if (abs(min) > 1e10 || abs(max) > 1e10) {
                cerr << "feature " << i << " ("
                     << result.feature_space()->print(feature)
                     << ") skipped as its range is too large: "
                     << min << " to " << max << endl;
                continue;
            }
        }
        features.push_back(feature);
    }

    cerr << "decorrelate: " << features.size()
         << " features after trivial elimination" << endl;
    
    std::sort(features.begin(), features.end());

    size_t nx = data.example_count();
    size_t nf = features.size();

    /* Get a dense matrix of the features. */
    boost::multi_array<double, 2> input(boost::extents[nx][nf + 1]);
    
    for (unsigned x = 0;  x < nx;  ++x) {
        result.extract_features(data[x], &input[x][0]);
        input[x][nf] = 1.0;  // bias term
    }

    //cerr << "input.shape()[0] = " << input.shape()[0] << endl;
    //cerr << "input.shape()[1] = " << input.shape()[1] << endl;
    boost::multi_array<double, 2> inputt = transpose(input);

    //cerr << "inputt.shape()[0] = " << inputt.shape()[0] << endl;
    //cerr << "inputt.shape()[1] = " << inputt.shape()[1] << endl;
    vector<distribution<double> > dependent;
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

    //cerr << "permutations = " << permutations << endl;
    
    /* Find which features are left */
    vector<Feature> new_features;
    for (unsigned i = 0;  i < features.size();  ++i) {
        if (permutations[i] != -1) {
            //cerr << "feature " << new_features.size() << " is old feature "
            //     << i << " (" << feature_space->print(features[i]) << ")"
            //     << endl;
            new_features.push_back(features[i]);
        }
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
                double v = dependent[i][j];
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

#if 0
    vector<double> cov2(&covar[121][0], (&covar[121][0]) + nf);

    cerr << "covariance[121][121] = " << covar[121][121] << endl;

    cerr << "covariance 121 = " << cov2 << endl;

    //cerr << "covariance = " << covar << endl;
#endif    


    /* Do the cholevsky stuff */
    boost::multi_array<double, 2> transform(boost::extents[nf][nf]);
    {
        PROFILE_FUNCTION(t_cholesky);
        transform = transpose(lower_inverse(cholesky(covar)));
    }

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
    layer.activation = ACT_IDENTITY;
    
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
    const vector<vector<float *> > & weight_updates;
    const vector<float *> & bias_updates;
    double & correct;
    double & total;
    double & total_rms_error;
    float learning_rate;

    Training_Job_Info(const boost::multi_array<float, 2> & decorrelated,
                      const std::vector<Label> & labels,
                      const distribution<float> & example_weights,
                      const Perceptron & result,
                      const vector<vector<float *> > & weight_updates,
                      const vector<float *> & bias_updates,
                      double & correct, double & total,
                      double & total_rms_error,
                      float learning_rate)
        : decorrelated(decorrelated), labels(labels),
          example_weights(example_weights), result(result),
          weight_updates(weight_updates), bias_updates(bias_updates),
          correct(correct), total(total), total_rms_error(total_rms_error),
          learning_rate(learning_rate)
    {
    }

    void train(int x_start, int x_end, bool one_thread)
    {
        //cerr << "training " << x_start << " to " << x_end << endl;

        const vector<Perceptron::Layer> & layers = result.layers;
        size_t max_units = result.max_units;

        vector<distribution<float> > layer_outputs(layers.size());

        size_t nl = layers.size();

        // Accumulate the weight updates here.  They are applied later.
        vector<boost::multi_array<float, 2> > sub_weight_updates_storage;
        vector<distribution<float> > sub_bias_updates_storage(nl);
        sub_weight_updates_storage.reserve(nl);

        vector<vector<float *> > sub_weight_updates_ptrs;
        vector<float *> sub_bias_updates_ptrs;
    
        for (unsigned l = 0;  l < nl;  ++l) {
            size_t no = layers[l].outputs();
            size_t ni = layers[l].inputs();
            layer_outputs[l].resize(no);

            if (!one_thread) {
                sub_weight_updates_storage
                    .push_back(boost::multi_array<float, 2>
                               (boost::extents[ni][no]));

                sub_weight_updates_ptrs.push_back(vector<float *>());

                for (unsigned i = 0;  i < ni;  ++i)
                    sub_weight_updates_ptrs.back()
                        .push_back(&sub_weight_updates_storage[l][i][0]);

                sub_bias_updates_storage[l].resize(no);
                sub_bias_updates_ptrs
                    .push_back(&sub_bias_updates_storage[l][0]);
            }
        }

        /* If there's only a single thread, then we don't gain anything by
           batching the update, we can just directly write it */
        const vector<vector<float *> > & sub_weight_updates
            = (one_thread ? weight_updates : sub_weight_updates_ptrs);
            
        const vector<float *> & sub_bias_updates
            = (one_thread ? bias_updates : sub_bias_updates_ptrs);

        double sub_correct = 0.0, sub_total = 0.0;

        //size_t nx = decorrelated.shape()[0];
        size_t nf = decorrelated.shape()[1];
        size_t no = layers.back().outputs(); // num outputs

        const float saturated = 0.8;
        const float fire = saturated, inhibit = -saturated;
        
        distribution<float> correct(no, inhibit);

        float errors[max_units], delta[max_units];

        double my_rms_error = 0.0;
        
        for (unsigned x = x_start;  x < x_end;  ++x) {
            float w = example_weights[x];
            
            if (w == 0.0) continue;
            
            /* Forward propagate */
            {
                PROFILE_FUNCTION(t_fprop);
                /* Compute weights going forward. */
                std::copy(&decorrelated[x][0], &decorrelated[x][0] + nf,
                          &layer_outputs[0][0]);
                
                for (unsigned l = 1;  l < nl;  ++l)
                    layers[l].apply(layer_outputs[l - 1], layer_outputs[l]);

                if (x == 0 && false) {
                    cerr << "fprop: " << endl;
                    for (unsigned l = 0;  l < nl;  ++l)
                        cerr << "layer " << (l-1) << ": " << layer_outputs[l]
                             << endl;
                }
                    
            }
            
            /* Calculate the correctness. */
            Correctness c = correctness(layer_outputs.back().begin(),
                                        layer_outputs.back().end(),
                                        labels[x]);
            sub_correct += w * c.possible * c.correct;
            sub_total += w * c.possible;
       
            /* Calculate the error terms for each output unit. */
            /* TODO: regression */
            correct[labels[x]] = fire;
            
            double example_rms_error = 0.0;

            PROFILE_FUNCTION(t_bprop);
            /* Original output errors.  Also update the RMS errors. */
            for (unsigned i = 0;  i < no;  ++i) {
                errors[i] = correct[i] - layer_outputs.back()[i];
                example_rms_error += 0.5 * errors[i] * errors[i] / no;
            }

            if (x == 0 && false) {
                cerr << "errors for layer " << 0 << ": "
                     << distribution<float>(errors, errors + no) << endl;
            }

            my_rms_error += example_rms_error * w;

            /* Backpropegate. */
            for (int l = nl - 1;  l >= 1;  --l) {
                
                const Perceptron::Layer & layer = layers[l];
                
                size_t no = layer.outputs();
                size_t ni = layer.inputs();
                
                /* Differentiate the output. */
                layer.deltas(&layer_outputs[l][0], &errors[0], delta);
                
                if (l > 1) {
                    /* Calculate new errors (for the next layer). */
                    for (unsigned i = 0;  i < ni;  ++i)
                        errors[i] = SIMD::vec_dotprod_dp(&delta[0],
                                                         &layer.weights[i][0],
                                                         no);
                    
                    if (x == 0 && false) {
                        cerr << "errors for layer " << l << ": "
                             << distribution<float>(errors, errors + ni)
                             << endl;
                    }
                    
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
        
            /* Turn back off this example for the next time. */
            correct[labels[x]] = inhibit;
        }

        if (one_thread) {
            /* Weight updates were already calculated in place */
            this->correct += sub_correct;
            this->total += sub_total;
            total_rms_error += my_rms_error;
        }
        else {
            Guard guard(lock);
            
            this->correct += sub_correct;
            this->total += sub_total;
            total_rms_error += my_rms_error;
            
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
    }
};

struct Training_Job {
    Training_Job(Training_Job_Info & info,
                 int x_start, int x_end, bool one_thread)
        : info(info), x_start(x_start), x_end(x_end), one_thread(one_thread)
    {
    }

    Training_Job_Info & info;
    int x_start, x_end;
    bool one_thread;
    
    void operator () () const
    {
        info.train(x_start, x_end, one_thread);
    }
};

} // file scope

std::pair<double, double>
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

    int our_batch_size = batch_size;
    if (batch_size == 0.0) our_batch_size = nx;
    else if (batch_size < 1.0) our_batch_size = nx * batch_size;

    //cerr << "batch size of " << our_batch_size << " examples" << endl;
    
    int done_ex = 0;

    // Accumulate the weight updates here.  They are applied later.  These
    // are not however used if we only present a single example at a time;
    // in that case, we apply to the output weights directly.
    vector<boost::multi_array<float, 2> > weight_updates;
    weight_updates.reserve(nl);
    vector<distribution<float> > bias_updates(nl);

    // Pointers to the weights that we update
    vector<vector<float *> > weight_updates_ptrs;
    vector<float *> bias_updates_ptrs;
    
    double correct = 0.0;
    double total = 0.0;
    double total_rms_error = 0.0;

    if (our_batch_size > 1) {
        PROFILE_FUNCTION(t_setup);

        for (unsigned l = 0;  l < nl;  ++l) {
            size_t no = layers[l].outputs();
            size_t ni = layers[l].inputs();
            weight_updates.push_back(boost::multi_array<float, 2>
                                     (boost::extents[ni][no]));
            bias_updates[l].resize(no);

            weight_updates_ptrs.push_back(vector<float *>());
            
            for (unsigned i = 0;  i < ni;  ++i)
                weight_updates_ptrs.back()
                    .push_back(&weight_updates[l][i][0]);

            bias_updates_ptrs
                .push_back(&bias_updates[l][0]);
        }
    }
    else {
        for (unsigned l = 0;  l < nl;  ++l) {
            size_t ni = layers[l].inputs();

            weight_updates_ptrs.push_back(vector<float *>());
            
            for (unsigned i = 0;  i < ni;  ++i)
                weight_updates_ptrs.back()
                    .push_back(&layers[l].weights[i][0]);

            bias_updates_ptrs
                .push_back(&layers[l].bias[0]);
        }
    }

    float biggest_update = 0.0, biggest_value = 0.0;

    for (; done_ex < nx;  done_ex += our_batch_size) {

        size_t last_ex = std::min<size_t>(done_ex + our_batch_size, nx);
        
        int num_in_batch = last_ex - done_ex;

        // Zero everything out
        if (our_batch_size > 1 || use_cuda) {
            PROFILE_FUNCTION(t_zero);
            
            for (unsigned l = 0;  l < nl;  ++l) {
                size_t no = layers[l].outputs();
                size_t ni = layers[l].inputs();
                
                float * to_empty = &weight_updates[l][0][0];
                std::fill(to_empty, to_empty + no * ni, 0.0f);
                
                std::fill(bias_updates[l].begin(), bias_updates[l].end(), 0.0);
            }
        }

        if (use_cuda) {
            using namespace CUDA;

            const vector<Perceptron::Layer> & layers = result.layers;

            // NOTE: we don't keep layer 0, as it's the decorrelating and
            // conditioning layer.  That's the reason for indexes starting
            // at one, etc.

            size_t num_active_layers = layers.size() - 1;

            vector<int> architecture_spec;

            architecture_spec.push_back(layers[1].inputs());
            for (unsigned l = 0;  l < num_active_layers;  ++l)
                architecture_spec.push_back(layers[l + 1].outputs());
            const int * architecture = &architecture_spec[0];
            
            //cerr << "architecture_spec = " << architecture_spec << endl;

            vector<const float *> weights_vec;
            for (unsigned l = 0;  l < num_active_layers;  ++l)
                weights_vec.push_back(&layers[l + 1].weights[0][0]);
            const float * const * weights = &weights_vec[0];
            
            vector<const float *> bias_vec;
            for (unsigned l = 0;  l < num_active_layers;  ++l)
                bias_vec.push_back(&layers[l + 1].bias[0]);
            const float * const * biases = &bias_vec[0];
            
            vector<int> w_strides_vec;
            for (unsigned l = 0;  l < num_active_layers;  ++l)
                w_strides_vec.push_back(layers[l + 1].weights.shape()[1]);

            //cerr << "w_strides_vec = " << w_strides_vec << endl;

            const int * w_strides = &w_strides_vec[0];

            Backprop backprop;
            
            const float saturated = 0.8;
            const float fire = saturated, inhibit = -saturated;

            boost::shared_ptr<Backprop::Plan>
                plan = backprop.plan(num_active_layers,
                                     architecture,
                                     weights,
                                     biases,
                                     w_strides,
                                     activation,
                                     fire,
                                     inhibit,
                                     learning_rate,
                                     false /* on_host */);


            for (unsigned i = 0;  i < 1;  ++i) {
                float correct = 0.0, total = 0.0, rms_error = 0.0;

                int x_start = done_ex;
                int x_end = last_ex;
                
                /* First, get everything into the shape needed for CUDA. */
                
                const float * feature_vectors = &decorrelated[x_start][0];
                int num_feature_vectors = x_end - x_start;
                const float * example_weights_ptr = &example_weights[x_start];
                const int * labels_ptr
                    = reinterpret_cast<const int *>(&labels[x_start]);
                
                vector<float *> weight_updates_vec;
                for (unsigned l = 0;  l < num_active_layers;  ++l)
                    weight_updates_vec.push_back(&weight_updates[l + 1][0][0]);
                float * const * weight_updates_ptrs = &weight_updates_vec[0];
                
                vector<float *> bias_updates_vec;
                for (unsigned l = 0;  l < num_active_layers;  ++l)
                    bias_updates_vec.push_back(&bias_updates[l + 1][0]);
                float * const * bias_updates_ptrs = &bias_updates_vec[0];
                
                boost::shared_ptr<Backprop::Context>
                    context = backprop.execute(*plan,
                                               feature_vectors,
                                               num_feature_vectors,
                                               example_weights_ptr,
                                               labels_ptr,
                                               weight_updates_ptrs,
                                               bias_updates_ptrs,
                                               correct,
                                               total,
                                               rms_error);


                backprop.synchronize(*context);
                
#if 0
                this->correct += sub_correct;
                this->total += sub_total;
                total_rms_error += my_rms_error;
#endif
            }
        }
        else {
        
            Training_Job_Info info(decorrelated, labels, example_weights,
                                   result,
                                   weight_updates_ptrs, bias_updates_ptrs,
                                   correct, total,
                                   total_rms_error, learning_rate);
            
            int ex_per_job = std::max(8,
                                      std::min(1024,
                                               num_in_batch / (4 * num_threads())));
            ex_per_job = std::min(ex_per_job, num_in_batch);
            
            //cerr << "num_in_batch = " << num_in_batch << " ex_per_job = " << ex_per_job << "our_batch_size = " << our_batch_size << endl;
            
            bool one_thread = ex_per_job == our_batch_size;
            
            if (one_thread) {
                Training_Job job(info, done_ex, last_ex, true /* one_thread */);
                job();
            }
            else {
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
                    
                    for (size_t x = done_ex;  x < last_ex;  x += ex_per_job) {
                        size_t end = std::min(x + ex_per_job, nx);
                        
                        //cerr << "x = " << x << " last_ex = " << last_ex
                        //     << " ex_per_job = " << ex_per_job << " nx = " << nx
                        //     << " end = " << end << endl;
                        
                        worker.add(Training_Job(info, x, end,
                                                false /* one_thread */),
                                   format("Perceptron_Generator::train_iteration(): %zd-%zd "
                                          "under %d", x, end, group),
                                   group);
                    }
                }
                
                worker.run_until_finished(group);
            }
        } // CUDA or not CUDA

        PROFILE_FUNCTION(t_update);

        biggest_value = 0.0;
        biggest_update = 0.0;

        /* Finally, put the accumulated weights back. */
        for (unsigned l = 1;  l < nl && (our_batch_size != 1 || use_cuda);
             ++l) {
            Perceptron::Layer & layer = layers[l];
            
            size_t no = layer.outputs();
            size_t ni = layer.inputs();
            
            for (unsigned i = 0;  i < ni;  ++i) {

                for (unsigned o = 0;  o < no;  ++o)
                    biggest_update = std::max(biggest_update,
                                              abs(weight_updates[l][i][o]));

                SIMD::vec_add(&layer.weights[i][0], &weight_updates[l][i][0],
                              &layer.weights[i][0], no);

                for (unsigned o = 0;  o < no;  ++o)
                    biggest_value = std::max(biggest_value,
                                             abs(layer.weights[i][o]));
            }

            for (unsigned o = 0;  o < no;  ++o)
                biggest_update = std::max(biggest_update,
                                          abs(bias_updates[l][o]));
            
            SIMD::vec_add(&layer.bias[0], &bias_updates[l][0],
                          &layer.bias[0], no);

            for (unsigned o = 0;  o < no;  ++o)
                biggest_value = std::max(biggest_value,
                                         abs(layer.bias[o]));
        }

    }

    //cerr << "biggest value: " << biggest_value
    //     << " biggest update: " << biggest_update << endl;
    
    //cerr << "correct = " << correct << " total = " << total << endl;

    return make_pair(correct / total, sqrt(total_rms_error));
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

    cerr << "adding decorrelating input layer with " << nunits
         << " linear units" << endl;

    /* Add hidden layers with the specified sizes */
    for (unsigned i = 0;  i < architecture.size();  ++i) {
        int units = architecture[i];
        if (units == -1) units = result.features.size();

        cerr << "adding hidden layer " << i + 1 << " with "
             << units << " units and activation function "
             << activation << endl;

        Perceptron::Layer layer(nunits, units, activation);
        result.add_layer(layer);
        nunits = units;
    }
    
    /* Add the output units. */
    Perceptron::Layer layer(nunits, nout, activation);
    result.add_layer(layer);

    cerr << "adding output layer with " << nout << " units and activation "
         << activation << endl;
    
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

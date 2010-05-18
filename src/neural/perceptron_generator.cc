/* perceptron_generator.cc
   Jeremy Barnes, 15 March 2006
   Copyright (c) 2006 Jeremy Barnes  All rights reserved.
   $Source$

   Generator for perceptrons.
*/

#include "perceptron_generator.h"
#include "jml/boosting/registry.h"
#include <boost/timer.hpp>
#include <boost/progress.hpp>
#include "jml/boosting/training_index.h"
#include "jml/stats/distribution_simd.h"
#include "jml/arch/simd_vector.h"
#include "jml/algebra/lapack.h"
#include "jml/algebra/matrix_ops.h"
#include "jml/algebra/irls.h"
#include <iomanip>
#include "jml/boosting/config_impl.h"
#include "jml/utils/environment.h"
#include "jml/utils/profile.h"
#include "jml/utils/worker_task.h"
#include "jml/utils/guard.h"
#include "jml/boosting/evaluation.h"
#include <boost/scoped_ptr.hpp>
#include <boost/bind.hpp>
#include "jml/utils/smart_ptr_utils.h"
#include "jml/utils/vector_utils.h"
#include "jml/utils/pair_utils.h"
#include "jml/neural/dense_layer.h"

#if (JML_USE_CUDA == 1)
#include "backprop_cuda.h"
#endif

using namespace std;


namespace ML {


namespace {

Env_Option<bool> profile("PROFILE_PERCEPTRON", false);

double t_train = 0.0, t_decorrelate = 0.0;
double t_cholesky = 0.0, t_qr = 0.0, t_gs = 0.0, t_mean = 0.0, t_covar = 0.0;
double t_update = 0.0, t_fprop = 0.0, t_bprop = 0.0, t_zero = 0.0;
double t_setup = 0.0, t_cuda = 0.0;

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
            cerr << "    cuda:         " << t_cuda        << "s" << endl;
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
    config.find(output_activation, "output_activation");
    config.find(do_decorrelate, "decorrelate");
    config.find(do_normalize, "normalize");
    config.find(use_cuda, "use_cuda");
    config.find(training_algo, "training_algo");
    config.find(training_mode, "training_mode");
    config.find(min_examples_per_job, "min_examples_per_job");
    config.find(max_examples_per_job, "max_examples_per_job");
    config.find(use_textures, "use_textures");
    config.find(target_value, "target_value");
    config.find(order_of_sensitivity, "order_of_sensitivity");
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
    activation = output_activation = TF_TANH;
    do_decorrelate = true;
    do_normalize = true;
    batch_size = 1024;
    use_cuda = false;
    training_algo = 0;
    training_mode = 1;
    min_examples_per_job = 8;
    max_examples_per_job = 1024;
    use_textures = false;
    target_value = 0.8;
    order_of_sensitivity = true;
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
        .add("learning_rate", learning_rate, "real",
             "positive: rate of learning relative to dataset size: negative for absolute")
        .add("arch", arch_str, "(see doc)",
             "hidden unit specification; %i=in vars, %o=out vars; eg 5_10")
        .add("activation", activation,
             "activation function for neurons")
        .add("output_activation", output_activation,
             "activation function for output layer of neurons")
        .add("decorrelate", do_decorrelate,
             "decorrelate the features before training")
        .add("normalize", do_normalize,
             "normalize to zero mean and unit std before training")
        .add("batch_size", batch_size, "0.0-1.0 or 1 - nvectors",
             "number of samples in each \"mini batch\" for stochastic")
        .add("use_cuda", use_cuda, "boolean", "use the CUDA optimized kernel")
        .add("training_algo", training_algo, "0-1", "train: 0 = std, 1 = ultrastochastic")
        .add("training_mode", training_mode, "0-3", "ultrastochastic mode: 0 = standard backprop; 1 = per neuron output; 2 = per neuron input; 3 = after each weight")
        .add("target_value", target_value, "0.0-1.0", "the output for a 1 that we ask the network to provide");
    
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

    bool regression = feature_space->info(predicted).type() == REAL;

    Perceptron current(model);
    Perceptron best(model);
    
    float best_acc = 0.0;
    float best_rmse = 0.0;
    int best_iter = 0;

    bool validate_is_train = false;
    //if (validation_set.example_count() == 0
    //    || &validation_set == &training_set) 
    //    validate_is_train = true;

    if (validate_ex_weights.total() == 0.0)
        throw Exception("no validate ex weights");

    boost::scoped_ptr<boost::progress_display> progress;

    //cerr << "training_ex_weights = " << training_ex_weights << endl;
    //cerr << "validate_ex_weights = " << validate_ex_weights << endl;

    log("perceptron_generator", 1)
        << "training " << max_iter << " iterations..." << endl;

    if (verbosity < 3)
        progress.reset(new boost::progress_display
                       (max_iter, log("perceptron_generator", 1)));
    
    vector<int> arch = Perceptron::parse_architecture(arch_str);
    
    boost::multi_array<float, 2> decorrelated
        = init(training_set, features, arch, current, context);

    size_t nx_validate
        = (validate_is_train ? training_set : validation_set)
        .example_count();

    boost::multi_array<float, 2> val_decorrelated
        (boost::extents[nx_validate][decorrelated.shape()[1]]);

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
        << "  it   train     rmse     val     rmse    best     diff" << endl;
    
    float validate_acc = 0.0, validate_rmse = 0.0;
    float train_acc = 0.0, train_rmse = 0.0;

    const std::vector<Label> & labels
        = training_set.index().labels(predicted);
    const std::vector<Label> & val_labels JML_UNUSED
        = validation_set.index().labels(predicted);

    double last_best_acc = 0.0;
    double last_best_rmse = 0.0;

    for (unsigned i = 0;  i < max_iter;  ++i) {

        boost::tie(train_acc, train_rmse)
            = train_iteration(context, decorrelated, labels,
                              training_ex_weights,
                              current);
        
        if (validate_is_train) {
            validate_acc = train_acc;
            validate_rmse = train_rmse;
        }
        else boost::tie(validate_acc, validate_rmse)
                 = current.accuracy(val_decorrelated, val_labels,
                                    validate_ex_weights);
        
        last_best_acc = best_acc;
        last_best_rmse = best_rmse;

        if (i == min_iter
            || (validate_acc > best_acc && !regression)
            || (validate_acc < best_acc && regression)) {
            best = current;
            best_acc = validate_acc;
            best_rmse = validate_rmse;
            best_iter = i;
        }

        log("perceptron_generator", 3)
            << format("%4d %6.2f%% %8.6f %6.2f%% %8.6f %6.2f%% %+7.3f%%",
                      i, train_acc * 100.0, train_rmse, validate_acc * 100.0,
                      validate_rmse,
                      best_acc * 100.0, (validate_acc - last_best_acc) * 100.0)
            << endl;

        if (progress) ++(*progress);

                
        log("perceptron_generator", 5) << current.print() << endl;
    }
    
    if (profile)
        log("perceptron_generator", 1)
            << "training time: " << timer.elapsed() << "s" << endl;
    
    log("perceptron_generator", 1)
        << format("best was %6.2f%% on iteration %d", best_acc * 100.0,
                  best_iter)
        << endl;

    float trn_acc, trn_rmse, val_acc, val_rmse;
    boost::tie(trn_acc, trn_rmse)
        = best.accuracy(training_set);
    boost::tie(val_acc, val_rmse)
        = best.accuracy(validation_set);

    log("perceptron_generator", 1)
        << "best accuracy: " << trn_acc << "/"
        << trn_rmse << " train, "
        << val_acc << "/" << val_rmse
        << " validation" << endl;
    
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
    distribution<double> stdev(nf, 0.0);
    boost::multi_array<double, 2> covar(boost::extents[nf][nf]);

    if (do_decorrelate && !do_normalize)
        throw Exception("normalization required if decorrelation is done");

    if (do_normalize || do_decorrelate) {
        {
            PROFILE_FUNCTION(t_mean);
            for (unsigned f = 0;  f < nf;  ++f) {
                mean[f] = SIMD::vec_sum_dp(&inputt[f][0], nx) / nx;
                for (unsigned x = 0;  x < nx;  ++x)
                    inputt[f][x] -= mean[f];
            }
        }

        cerr << "mean = " << mean << endl;

        {
            PROFILE_FUNCTION(t_covar);
            for (unsigned f = 0;  f < nf;  ++f) {
                for (unsigned f2 = 0;  f2 <= f;  ++f2)
                    covar[f][f2] = covar[f2][f]
                        = SIMD::vec_dotprod_dp(&inputt[f][0], &inputt[f2][0], nx) / nx;
            }
        
            for (unsigned f = 0;  f < nf;  ++f)
                stdev[f] = sqrt(covar[f][f]);
            
            cerr << "stdev = " << stdev << endl;
        }
    }
    
    boost::multi_array<double, 2> transform(boost::extents[nf][nf]);

    if (do_decorrelate) {
        /* Do the cholevsky stuff */
        PROFILE_FUNCTION(t_cholesky);
        transform = transpose(lower_inverse(cholesky(covar)));
    }
    else if (do_normalize) {
        /* Use a unit diagonal of 1/stdev for the transform; no
           decorrelation */
        std::fill(transform.origin(),
                  transform.origin() + transform.num_elements(),
                  0.0f);
        for (unsigned f = 0;  f < nf;  ++f)
            transform[f][f] = 1.0 / stdev[f];
    }
    else {
        /* Use the identity function */
        std::fill(transform.origin(),
                  transform.origin() + transform.num_elements(),
                  0.0f);
        for (unsigned f = 0;  f < nf;  ++f)
            transform[f][f] = 1.0;
    }
    
    /* Finally, we add a layer.  This will perform both the removal of the
       mean (via the biasing) and the decorrelation (via the application
       of the matrix).
    */
    
    /* y = (x - mean) * A;
         = (x * A) - (mean * A);
    */

    boost::shared_ptr<Dense_Layer<float> > layer
        (new Dense_Layer<float>("decorrelation", nf, nf, TF_IDENTITY,
                                MV_NONE));
    layer->weights.resize(boost::extents[transform.shape()[0]][transform.shape()[1]]);
    layer->weights = transform;
    layer->bias = distribution<float>(nf, 0.0);  // already have mean removed
    
    //cerr << "transform = " << transform << endl;

    boost::multi_array<float, 2> decorrelated(boost::extents[nx][nf]);
    float fv_in[nf];

    for (unsigned x = 0;  x < nx;  ++x) {
        for (unsigned f = 0;  f < nf;  ++f)
            fv_in[f] = inputt[f][x];

        //cerr << "fv_in = " << distribution<float>(fv_in, fv_in + nf) << endl;

        layer->apply(&fv_in[0], &decorrelated[x][0]);

        //cerr << "fv_out = "
        //     << distribution<float>(&decorrelated[x][0],
        //                            &decorrelated[x][0] + nf)
        //     << endl;
        
    }

    layer->bias = (transform * mean) * -1.0;  // now add the bias
    result.layers.clear();
    result.add_layer(layer);
    
    layer->validate();

    return decorrelated;
}

#if 0

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
    int algo;
    int mode;
    float inhibit;
    float fire;
    bool order_of_sensitivity;

    Training_Job_Info(const boost::multi_array<float, 2> & decorrelated,
                      const std::vector<Label> & labels,
                      const distribution<float> & example_weights,
                      const Perceptron & result,
                      const vector<vector<float *> > & weight_updates,
                      const vector<float *> & bias_updates,
                      double & correct, double & total,
                      double & total_rms_error,
                      float learning_rate,
                      int algo,
                      int mode,
                      float inhibit,
                      float fire,
                      bool order_of_sensitivity)
        : decorrelated(decorrelated), labels(labels),
          example_weights(example_weights), result(result),
          weight_updates(weight_updates), bias_updates(bias_updates),
          correct(correct), total(total), total_rms_error(total_rms_error),
          learning_rate(learning_rate), algo(algo), mode(mode),
          inhibit(inhibit), fire(fire),
          order_of_sensitivity(order_of_sensitivity)
    {
    }

    void fprop(int x,
               const Layer_Stack<Layer> & layers,
               vector<distribution<float> > & layer_outputs) const
    {
        size_t nf = decorrelated.shape()[1];
        size_t nl = layers.size();

        /* Forward propagate */
        {
            PROFILE_FUNCTION(t_fprop);
            /* Compute weights going forward. */
            std::copy(&decorrelated[x][0], &decorrelated[x][0] + nf,
                      &layer_outputs[0][0]);
            
#if 0
            cerr << "input for example " << x << ": " << layer_outputs[0]
                 << endl;
#endif
            
            for (unsigned l = 1;  l < nl;  ++l) {
                layers[l]->apply(layer_outputs[l - 1], layer_outputs[l]);
#if 0
                cerr << "  output of layer " << l << ": " << layer_outputs[l]
                     << endl;
#endif
            }
        }
    }

    void train(int x_start, int x_end, bool one_thread)
    {
        switch (algo) {
        case 0:
            train_normal(x_start, x_end, one_thread);
            break;

        case 1:
            train_ultrastochastic(x_start, x_end, one_thread);
            break;

        default:
            throw Exception("unknown training algo");
        }
    }

    void train_normal(int x_start, int x_end, bool one_thread)
    {
        //cerr << "training " << x_start << " to " << x_end << endl;

        const Layer_Stack<Layer> & layers = result.layers;
        size_t max_units = layers.max_width();

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

        size_t no = layers.outputs(); // num outputs

        distribution<float> correct(no, inhibit);

        float errors[max_units], delta[max_units];

        double my_rms_error = 0.0;
        
        bool regression_problem
            = result.feature_space()->info(result.predicted()).type() == REAL;

        for (unsigned x = x_start;  x < x_end;  ++x) {
            float w = example_weights[x];
            
            if (w == 0.0) continue;
            
            fprop(x, layers, layer_outputs);
            
            if (regression_problem) {
                correct[0] = labels[x].value();
                double error = fabs(correct[0] - layer_outputs.back()[0]);
                sub_correct += w * error;
                my_rms_error+= w * error * error;
                sub_total += w;
            }
            else {
                /* Calculate the correctness. */
                Correctness c = correctness(layer_outputs.back().begin(),
                                            layer_outputs.back().end(),
                                            labels[x]);
                sub_correct += w * c.possible * c.correct;
                my_rms_error += w * c.possible * c.margin * c.margin;
                sub_total += w * c.possible;
                correct[labels[x]] = fire;
            }
       


            //cerr << "layer_output = " << layer_outputs.back()
            //     << " correct " << correct << " label " << labels[x].value()
            //     << endl;

            //cerr << "correct = " << correct << endl;
            
            double example_rms_error = 0.0;

            PROFILE_FUNCTION(t_bprop);
            /* Original output errors.  Also update the RMS errors. */
            for (unsigned i = 0;  i < no;  ++i) {
                errors[i] = correct[i] - layer_outputs.back()[i];
                example_rms_error += 0.5 * errors[i] * errors[i] / no;
            }

            if ((x == 0 && false)) {
                cerr << "errors for layer " << 0 << ": "
                     << distribution<float>(errors, errors + no) << endl;
            }

            //my_rms_error += example_rms_error * w;

            /* Backpropegate.s */
            for (int l = nl - 1;  l >= 1;  --l) {
                
                const Dense_Layer<float> & layer
                    = dynamic_cast<const Dense_Layer<float> &>(*layers[l]);
                
                size_t no = layer.outputs();
                size_t ni = layer.inputs();

#if 0
                cerr << "layer " << l << " errors: "
                     << distribution<float>(errors, errors + no)
                     << endl;
#endif
                
                
                /* Differentiate the output. */
                layer.deltas(&layer_outputs[l][0], &errors[0], delta);

                
                if (l > 1) {
                    /* Calculate new errors (for the next layer). */
                    for (unsigned i = 0;  i < ni;  ++i)
                        errors[i] = SIMD::vec_dotprod_dp(&delta[0],
                                                         &layer.weights[i][0],
                                                         no);
#if 0
                    cerr << "errors for layer " << l - 1 << ": "
                         << distribution<float>(errors, errors + ni)
                         << endl;
#endif
                }
                
                /* Update the weights. */
                float k = w * learning_rate;
                for (unsigned i = 0;  i < ni;  ++i) {

                    float k2 = layer_outputs[l - 1][i] * k;

#if 0
                    for (unsigned o = 0;  o < no;  ++o)
                        cerr << "update parameter (" << i << "," << o
                             << ") : before " << layer.weights[i][o]
                             << " + " << k << " * " << layer_outputs[l-1][i]
                             << " * " << delta[o] << " = "
                             << k2 * delta[o] << endl;
#endif
                    
                    SIMD::vec_add(&sub_weight_updates[l][i][0], k2, &delta[0],
                                  &sub_weight_updates[l][i][0], no);
                }
            
                /* Update the bias terms.  The previous layer output (input) is
                   always 1. */
                SIMD::vec_add(&sub_bias_updates[l][0], k, &delta[0],
                              &sub_bias_updates[l][0], no);
            }
        
            /* Turn back off this example for the next time. */
            if (!regression_problem)
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
                const Dense_Layer<float> & layer
                    = dynamic_cast<const Dense_Layer<float> &>(*layers[l]);
                
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

    /** output_num_in_isolation == -1: calculate deltas for all outputs.
        Otherwise, just for the given output */
    double bprop(int x,
                 const vector<boost::shared_ptr<Layer> > & layers,
                 const vector<distribution<float> > & layer_outputs,
                 vector<distribution<float> > & errors,
                 vector<distribution<float> > & deltas,
                 int output_num_in_isolation = -1) const
    {
        double example_rms_error = 0.0;
        size_t nl = layers.size();
        int no = layers[nl - 1]->outputs();
        
        PROFILE_FUNCTION(t_bprop);

        /* Original output errors.  Also update the RMS errors. */
        for (unsigned o = 0;  o < no;  ++o) {
            float correct = (labels[x] == o ? fire : inhibit);
            float error = correct - layer_outputs[nl - 1][o];

            if (isnan(error)) {
                cerr << "example " << x << " of " << example_weights.size()
                     << endl;
                throw Exception("bprop: error is NaN");
            }

            example_rms_error += 0.5 * error * error / no;

            if (o == output_num_in_isolation || output_num_in_isolation == -1) {
                /* If we just want to calculate deltas for one of the outputs,
                   then we set the others to zero. */
                errors[nl - 1][o] = error;
            }
            else errors[nl - 1][o] = 0.0;
        }

        //cerr << "errors " << nl - 1 << " = " << errors[nl - 1] << endl;
        
        /* Backpropegate. */
        for (int l = nl - 1;  l >= 1;  --l) {
            
            const Dense_Layer<float> & layer
                = dynamic_cast<const Dense_Layer<float> &>(*layers[l]);
            
            size_t no = layer.outputs();
            size_t ni = layer.inputs();
            
            /* Differentiate the output. */
            layer.deltas(&layer_outputs[l][0], &errors[l][0], &deltas[l][0]);

            //cerr << "deltas " << l << " = " << deltas[l] << endl;
            
            if (l > 1) {
                /* Calculate new errors (for the next layer). */
                for (unsigned i = 0;  i < ni;  ++i)
                    errors[l - 1][i]
                        = SIMD::vec_dotprod_dp(&deltas[l][0],
                                               &layer.weights[i][0],
                                               no);

                //cerr << "errors " << l - 1 << " = " << errors[l - 1] << endl;
            }
        }

        return example_rms_error;
    }

    static vector<int> sort_by_abs(const distribution<float> & values)
    {
        vector<pair<float, int> > to_sort;
        to_sort.reserve(values.size());

        for (unsigned i = 0;  i < values.size();  ++i) {
            if (isnan(values[i]))
                throw Exception("sort_by_abs: value is NaN");
            if (!isfinite(values[i]))
                throw Exception("sort_by_abs: value is not finite");
            to_sort.push_back(make_pair(abs(values[i]), i));
        }

        std::sort(to_sort.begin(), to_sort.end());

        return vector<int>(second_extractor(to_sort.begin()),
                           second_extractor(to_sort.end()));
    }

    void update_with_deltas
        (const vector<boost::shared_ptr<Layer> > & original,
         const vector<boost::shared_ptr<Layer> > & updated)
    {
        size_t nl = updated.size();

        /* Finally, put the accumulated weights back.  We calculate the
           updates by taking the difference between the current weights
           and the original weights. */
        for (unsigned l = 1;  l < nl;  ++l) {

            const Dense_Layer<float> & layer0
                = dynamic_cast<const Dense_Layer<float> &>(*original[l]);
            const Dense_Layer<float> & layer
                = dynamic_cast<const Dense_Layer<float> &>(*updated[l]);

            size_t no = layer.outputs();
            size_t ni = layer.inputs();
            
            for (unsigned i = 0;  i < ni;  ++i)
                for (unsigned o = 0;  o < no;  ++o)
                    weight_updates[l][i][o]
                        += (layer.weights[i][o] - layer0.weights[i][o]);
            
            for (unsigned o = 0;  o < no;  ++o)
                bias_updates[l][o]
                    += (layer.bias[o] - layer0.bias[o]);
        }
    }

    void train_ultrastochastic(int x_start, int x_end, bool one_thread)
    {
        /* Train using a more stochastic algorithm.  This algorithm will
           update the output weights from one hidden unit at a time, and
           then re-do the fprop and backprop so that multiple hidden units
           that have similar behaviour will tend to diverge from each other.
        */

        //cerr << "training ultrastochastic " << x_start << " to " << x_end
        //     << " mode = " << mode << endl;

        // If we have a single thread, we can simply update the values in place
        // straight away.  Otherwise, we need to accumulate updates.

        // Make a deep copy of the old perceptron that we can update as we go
        // and know what the combined updates were

        size_t nl = result.layers.size();

        vector<boost::shared_ptr<Layer> > layers;

        // Starting version of weights to compute updates
        vector<boost::shared_ptr<Layer> > original;

        if (one_thread) {
            layers = result.layers;
            // original not needed
        }
        else {
            // Layers is a copy
            for (unsigned i = 0;  i < layers.size();  ++i)
                layers.push_back(make_sp(result.layers[i]->make_copy()));
            original = result.layers;
        }

        vector<distribution<float> > layer_outputs(layers.size());
        
        vector<distribution<float> > errors(nl), deltas(nl);
        vector<distribution<float> > errors2(nl), deltas2(nl);

        for (unsigned l = 0;  l < nl;  ++l) {
            size_t no = layers[l]->outputs();
            errors[l].resize(no);
            deltas[l].resize(no);
            errors2[l].resize(no);
            deltas2[l].resize(no);
            layer_outputs[l].resize(no);
        }

        /* Modify the parameters straight away */

        double sub_correct = 0.0, sub_total = 0.0;
        double my_rms_error = 0.0;
        
        for (unsigned x = x_start;  x < x_end;  ++x) {
            float w = example_weights[x];
            
            if (w == 0.0) continue;

            double example_rms_error;
            
            // Calculate outputs
            fprop(x, layers, layer_outputs);

            // Propagate the error gradient
            example_rms_error = bprop(x, layers, layer_outputs, errors, deltas);

            /* Update the weights. */
            float k = w * learning_rate;

            if (mode == 4 || mode == 5) {
                /* Implement the fraka4 mode.  In this mode, we optimize
                   separately for each of the output units. */
                int no_final = layers[nl - 1]->outputs();

                vector<int> output_indexes = sort_by_abs(deltas[nl - 1]);

                for (unsigned oi = 0;  oi < no_final;  ++oi) {
                    unsigned o
                        = (order_of_sensitivity
                           ? output_indexes[oi]
                           : oi);

                    fprop(x, layers, layer_outputs);
                    example_rms_error = bprop(x, layers, layer_outputs,
                                              errors, deltas, o);

                    if (mode == 4)
                        example_rms_error = bprop(x, layers, layer_outputs,
                                                  errors2, deltas2);
                    
                    for (int l = nl - 1;  l >= 1;  --l) {

                        Dense_Layer<float> & layer
                            = dynamic_cast<Dense_Layer<float> &>
                            (*layers[l]);
                        
                        size_t no = layer.outputs();
                        size_t ni = layer.inputs();
                        
                        const distribution<float> & delta
                            = (l == 1 || mode == 5 ? deltas[l] : deltas2[l]);
                        
                        /* Update the bias terms.  The previous layer output
                           (input) is always 1. */
                        SIMD::vec_add(&layer.bias[0], k, &delta[0],
                                      &layer.bias[0], no);

                        //cerr << "w updates for layer " << l << endl;
                        
                        for (unsigned i = 0;  i < ni;  ++i) {
                            float k2 = layer_outputs[l - 1][i] * k;

#if 0
                            if (i < 3 || i >= (ni - 3)) {
                                for (unsigned o = 0;  o < no;  ++o) {
                                    if (o < 3 || o >= (no - 3))
                                        cerr << "update[" << i << "]["
                                             << o << "] = "
                                             << k2 * delta[o] << endl;
                                }
                            }
#endif

                            SIMD::vec_add(&layer.weights[i][0], k2, &delta[0],
                                          &layer.weights[i][0], no);
                        }

#if 0
                        for (unsigned o = 0;  o < no;  ++o) {
                            if (o < 3 || o >= (no - 3))
                                cerr << "biasup[" << o << "] = "
                                     << k * delta[o] << endl;
                        }
#endif
                    }
                }

#if 0
                cerr << "AFTER PRESENTATION OF ONE EXAMPLE" << endl;

                for (int l = nl - 1;  l >= 1;  --l) {
                    
                    Layer & layer = *layers[l];
                    
                    size_t no = layer.outputs();
                    size_t ni = layer.inputs();

                    for (unsigned i = 0;  i < ni;  ++i) {
                        if (i < 3 || i >= (ni - 3)) {
                            for (unsigned o = 0;  o < no;  ++o) {
                                if (o < 3 || o >= (no - 3))
                                    cerr << "weights[" << l << "][" << i
                                         << "][" << o << "] = "
                                         << layer.weights[i][o]
                                         << endl;
                            }
                        }
                    }

                    for (unsigned o = 0;  o < no;  ++o) {
                        if (o < 3 || o >= (no - 3))
                            cerr << "biases[" << l << "][" << o << "] = "
                                 << layer.bias[o] << endl;
                    }
                }
#endif
            }


            // Now for the updates.  We go from the lower layer to the higher
            // layer.  Once all updates have been made on a particular unit,
            // we re-propagate.

            for (unsigned l = 1;  l < nl && mode < 4;  ++l) {

                Dense_Layer<float> & layer
                    = dynamic_cast<Dense_Layer<float> &>(*layers[l]);
                
                size_t no = layer.outputs();
                size_t ni = layer.inputs();

                vector<int> input_indexes = sort_by_abs(deltas[l - 1]);
                vector<int> output_indexes = sort_by_abs(deltas[l]);

                const distribution<float> & delta = deltas[l];
                
                if (mode == 0 || mode == 1) {
                    /* Update the bias terms.  The previous layer output
                       (input) is always 1. */
                    SIMD::vec_add(&layer.bias[0], k, &delta[0],
                                  &layer.bias[0], no);
                    
#if 0
                    cerr << "x " << x << " layer " << l << " k " << k
                         << " output " << layer_outputs[l]
                         << " error " << errors[l] << " bias upd " << delta << endl;
#endif
                    
                    if (mode == 1) {
                        fprop(x, layers, layer_outputs);
                        example_rms_error
                            = bprop(x, layers, layer_outputs, errors, deltas);
                    }
                    
                    for (unsigned ii = 0;  ii < ni;  ++ii) {
                        unsigned i = (order_of_sensitivity ? input_indexes[ii] : ii);
                        
                        float k2 = layer_outputs[l - 1][i] * k;
                        SIMD::vec_add(&layer.weights[i][0], k2, &delta[0],
                                      &layer.weights[i][0], no);
                        
                        if (mode == 1) {
                            // re-propagate; we've modified some parameters already
                            fprop(x, layers, layer_outputs);
                            example_rms_error
                                = bprop(x, layers, layer_outputs, errors, deltas);
                        }
                    }
                }
                else if (mode == 2) {
                    for (unsigned oi = 0;  oi < no;  ++oi) {
                        unsigned o = (order_of_sensitivity ? output_indexes[oi] : oi);
                        
                        layer.bias[o] += k * delta[o];
                        
                        fprop(x, layers, layer_outputs);
                        example_rms_error
                            = bprop(x, layers, layer_outputs, errors, deltas);
                        
                        for (unsigned i = 0;  i < ni;  ++i) {
                            float k2 = layer_outputs[l - 1][i] * k;
                            layer.weights[i][o] += k2 * delta[o];
                        }

                        fprop(x, layers, layer_outputs);
                        example_rms_error
                            = bprop(x, layers, layer_outputs, errors, deltas);
                    }
                }
                else if (mode == 3) {
                    for (unsigned o = 0;  o < no;  ++o) {
                        layer.bias[o] += k * delta[o];
                        fprop(x, layers, layer_outputs);
                        example_rms_error
                            = bprop(x, layers, layer_outputs, errors, deltas);
                    }
                    
                    for (unsigned i = 0;  i < ni;  ++i) {
                        float k2 = layer_outputs[l - 1][i] * k;

                        for (unsigned o = 0;  o < no;  ++o) {
                            layer.weights[i][o] += k2 * delta[o];
                            fprop(x, layers, layer_outputs);
                            example_rms_error
                                = bprop(x, layers, layer_outputs, errors, deltas);
                        }
                    }
                }
            }

            /* Calculate the correctness. */
            Correctness c = correctness(layer_outputs.back().begin(),
                                        layer_outputs.back().end(),
                                        labels[x]);
            sub_correct += w * c.possible * c.correct;
            sub_total += w * c.possible;
            my_rms_error += example_rms_error * w;
        }


        Guard guard(lock);
            
        this->correct += sub_correct;
        this->total += sub_total;
        total_rms_error += my_rms_error;

        if (one_thread) return;  // layer updates were done in place

        update_with_deltas(original, layers);
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

#endif

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

    vector<boost::shared_ptr<Dense_Layer<float> > > layers;
    //for (unsigned i = 0;  i < result.layers.size();  ++i)
    //    layers.push_back(boost::dynamic_pointer_cast<Dense_Layer<float> >
    //                     (result.layers[i]));
    size_t nl = layers.size();

    float inhibit, fire;
    boost::tie(inhibit, fire)
        = layers[0]->targets(target_value);

    int our_batch_size = batch_size;
    if (batch_size == 0.0) our_batch_size = nx;
    else if (batch_size < 1.0) our_batch_size = nx * batch_size;

    //cerr << "batch size of " << our_batch_size << " examples" << endl;
    
    int done_ex = 0;

    /* If we specified a negative learning rate, that means that we wanted it
       to be absolute (and so not depend upon the size of the dataset).  In
       order to get this behaviour, we need to multipy by the number of
       examples in the dataset, since it is implicitly multiplied by the
       example weight (on average 1/num examples in training set) as part of
       the training, and we want to counteract this effect).
    */
    float learning_rate = this->learning_rate;
    if (learning_rate < 0.0) {
        learning_rate *= -1.0 * example_weights.size() / example_weights.total();
        //cerr << "learning_rate: " << learning_rate << endl;
    }

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

    if (our_batch_size > 1 || use_cuda) {
        PROFILE_FUNCTION(t_setup);

        for (unsigned l = 0;  l < nl;  ++l) {
            size_t no = layers[l]->outputs();
            size_t ni = layers[l]->inputs();
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
            size_t ni = layers[l]->inputs();

            weight_updates_ptrs.push_back(vector<float *>());
            
            for (unsigned i = 0;  i < ni;  ++i)
                weight_updates_ptrs.back()
                    .push_back(&layers[l]->weights[i][0]);

            bias_updates_ptrs
                .push_back(&layers[l]->bias[0]);
        }
    }

    float biggest_update = 0.0, biggest_value = 0.0;

    for (; done_ex < nx;  done_ex += our_batch_size) {

        //size_t last_ex = std::min<size_t>(done_ex + our_batch_size, nx);
        
        //int num_in_batch = last_ex - done_ex;

        // Zero everything out
        if (our_batch_size > 1 || use_cuda) {
            PROFILE_FUNCTION(t_zero);
            
            for (unsigned l = 0;  l < nl;  ++l) {
                size_t no = layers[l]->outputs();
                size_t ni = layers[l]->inputs();
                
                float * to_empty = &weight_updates[l][0][0];
                std::fill(to_empty, to_empty + no * ni, 0.0f);
                
                std::fill(bias_updates[l].begin(), bias_updates[l].end(), 0.0);
            }
        }

        if (use_cuda) {
#if (JML_USE_CUDA == 1)
            PROFILE_FUNCTION(t_cuda);

            using namespace CUDA;

            const vector<boost::shared_ptr<Layer> > & layers
                = result.layers;

            // NOTE: we don't keep layer 0, as it's the decorrelating and
            // conditioning layer.  That's the reason for indexes starting
            // at one, etc.

            size_t num_active_layers = layers.size() - 1;

            vector<int> architecture_spec;

            architecture_spec.push_back(layers[1]->inputs());
            for (unsigned l = 0;  l < num_active_layers;  ++l)
                architecture_spec.push_back(layers[l + 1]->outputs());
            const int * architecture = &architecture_spec[0];
            
            //cerr << "architecture_spec = " << architecture_spec << endl;

            vector<const float *> weights_vec;
            for (unsigned l = 0;  l < num_active_layers;  ++l)
                weights_vec.push_back(&layers[l + 1]->weights[0][0]);
            const float * const * weights = &weights_vec[0];
            
            vector<const float *> bias_vec;
            for (unsigned l = 0;  l < num_active_layers;  ++l)
                bias_vec.push_back(&layers[l + 1]->bias[0]);
            const float * const * biases = &bias_vec[0];
            
            vector<int> w_strides_vec;
            for (unsigned l = 0;  l < num_active_layers;  ++l)
                w_strides_vec.push_back(layers[l + 1]->weights.shape()[1]);

            //cerr << "w_strides_vec = " << w_strides_vec << endl;

            const int * w_strides = &w_strides_vec[0];

            Backprop backprop;
            
            boost::shared_ptr<Backprop::Plan>
                plan = backprop.plan(num_active_layers,
                                     architecture,
                                     weights,
                                     biases,
                                     w_strides,
                                     activation,
                                     inhibit,
                                     fire,
                                     learning_rate,
                                     false /* on_host */,
                                     use_textures);


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
#else
            throw Exception("CUDA not configured but was asked for");
#endif
        }
        else {

#if 0        
            Training_Job_Info info(decorrelated, labels, example_weights,
                                   result,
                                   weight_updates_ptrs, bias_updates_ptrs,
                                   correct, total,
                                   total_rms_error, learning_rate,
                                   training_algo, training_mode,
                                   inhibit, fire,
                                   order_of_sensitivity);
            
            int ex_per_job = std::max(min_examples_per_job,
                                      std::min(max_examples_per_job,
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
#endif

        } // CUDA or not CUDA

        PROFILE_FUNCTION(t_update);

        biggest_value = 0.0;
        biggest_update = 0.0;

        /* Finally, put the accumulated weights back. */
        for (unsigned l = 1;  l < nl && (our_batch_size != 1 || use_cuda);
             ++l) {
            Dense_Layer<float> & layer
                = dynamic_cast<Dense_Layer<float> &>(*layers[l]);
            
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

        //cerr << "biggest value: " << biggest_value
        //     << " biggest update: " << biggest_update << endl;

    }

    
    //cerr << "correct = " << correct << " total = " << total << endl;

    return make_pair(correct / total, sqrt(total_rms_error / total));
}

boost::multi_array<float, 2>
Perceptron_Generator::
init(const Training_Data & data,
     const std::vector<Feature> & possible_features,
     const std::vector<int> & architecture,
     Perceptron & result,
     Thread_Context & context) const
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

        boost::shared_ptr<Layer>
            layer(new Dense_Layer<float>(format("hidden%d", i),
                                         nunits, units, activation,
                                         MV_NONE, context));
        result.add_layer(layer);
        nunits = units;
    }
    
    /* Add the output units. */
    boost::shared_ptr<Layer> layer
        (new Dense_Layer<float>("output", nunits, nout, output_activation,
                                MV_NONE, context));
    result.add_layer(layer);

    cerr << "adding output layer with " << nout << " units and activation "
         << output_activation << endl;
    
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

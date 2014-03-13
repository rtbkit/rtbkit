/* perceptron_generator.cc
   Jeremy Barnes, 15 March 2006
   Copyright (c) 2006 Jeremy Barnes  All rights reserved.

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
#include "discriminative_trainer.h"

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
    config.find(output_activation, "output_activation");
    config.find(do_decorrelate, "decorrelate");
    config.find(do_normalize, "normalize");
    config.find(target_value, "target_value");
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
    target_value = 0.8;
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
        .add("target_value", target_value, "0.0-1.0", "the output for a 1 that we ask the network to provide");
    
    return result;
}

void
Perceptron_Generator::
init(std::shared_ptr<const Feature_Space> fs, Feature predicted)
{
    Early_Stopping_Generator::init(fs, predicted);
    model = Perceptron(fs, predicted);
}

std::shared_ptr<Classifier_Impl>
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
    size_t nxv = val_decorrelated.shape()[0];
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
    double last_best_rmse JML_UNUSED = 0.0;

    int our_batch_size = batch_size;
    if (batch_size == 0.0) our_batch_size = nx;
    else if (batch_size < 1.0) our_batch_size = nx * batch_size;

    /* If we specified a negative learning rate, that means that we wanted it
       to be absolute (and so not depend upon the size of the dataset).  In
       order to get this behaviour, we need to multipy by the number of
       examples in the dataset, since it is implicitly multiplied by the
       example weight (on average 1/num examples in training set) as part of
       the training, and we want to counteract this effect).
    */
    float learning_rate = this->learning_rate;
    if (learning_rate < 0.0) {
        learning_rate
            *= -1.0 * training_ex_weights.size() / training_ex_weights.total();
    }

    // Create a layer stack without the decorrelation layer to be trained
    Layer_Stack<Layer> train_stack;
    for (unsigned i = 1;  i < current.layers.size();  ++i)
        train_stack.add(current.layers.share(i));
    
    Discriminative_Trainer trainer;
    trainer.layer = &train_stack;

    bool randomize = false;
    float sample_proportion = 1.0;

    vector<const float *> examples(nx);
    for (unsigned i = 0;  i < nx;  ++i)
        examples[i] = &decorrelated[i][0];

    vector<const float *> val_examples(nxv);
    for (unsigned i = 0;  i < nxv;  ++i)
        val_examples[i] = &val_decorrelated[i][0];
    
    Output_Encoder & output_encoder = current.output;
    output_encoder.configure(model.feature_space()->info(model.predicted()),
                             train_stack, target_value);


    for (unsigned i = 0;  i < max_iter;  ++i) {

        //cerr << "params = " << Parameters_Copy<float>(train_stack.parameters()).values << endl;

        //cerr << "mode = " << output_encoder.mode << " value_true = " << output_encoder.value_true
        //     << " value_false = " << output_encoder.value_false << " num_inputs = "
        //     << output_encoder.num_inputs << " num_outputs = " << output_encoder.num_outputs
        //     << endl;

        {
            PROFILE_FUNCTION(t_train);
            
            boost::tie(train_acc, train_rmse)
                = trainer.train_iter(examples, labels, training_ex_weights,
                                     output_encoder,
                                     context, our_batch_size, learning_rate,
                                     verbosity, sample_proportion, randomize);
        }

        if (validate_is_train) {
            validate_acc = train_acc;
            validate_rmse = train_rmse;
        }
        else {
            boost::tie(validate_acc, validate_rmse)
                = trainer.test(val_examples, val_labels, validate_ex_weights,
                               output_encoder, context, verbosity);
        }

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

    std::shared_ptr<Dense_Layer<float> > layer
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

        std::shared_ptr<Layer>
            layer(new Dense_Layer<float>(format("hidden%d", i),
                                         nunits, units, activation,
                                         MV_NONE, context));
        result.add_layer(layer);
        nunits = units;
    }
    
    /* Add the output units. */
    std::shared_ptr<Layer> layer
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

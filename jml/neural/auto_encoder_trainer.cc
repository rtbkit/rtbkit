/* auto_encoder_trainer.cc
   Jeremy Barnes, 11 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Trainer for an auto-encoder.
*/

#include "auto_encoder_trainer.h"
#include "jml/utils/configuration.h"
#include "jml/arch/threads.h"

#include <boost/progress.hpp>
#include "jml/utils/worker_task.h"
#include <boost/tuple/tuple.hpp>
#include "jml/utils/guard.h"
#include "jml/utils/configuration.h"
#include "jml/arch/timers.h"
#include <boost/bind.hpp>
#include "auto_encoder_stack.h"
#include "jml/utils/check_not_nan.h"
#include "jml/stats/distribution_ops.h"


using namespace std;

namespace ML {


/*****************************************************************************/
/* AUTO_ENCODER_TRAINER                                                      */
/*****************************************************************************/

Auto_Encoder_Trainer::
Auto_Encoder_Trainer()
{
    defaults();
}

void
Auto_Encoder_Trainer::
defaults()
{
    learning_rate = 0.75;
    minibatch_size = 512;
    niter = 50;
    prob_cleared = 0.10;
    verbosity = 2;
    randomize_order = true;
    sample_proportion = 0.8;
    test_every = 1;
    prob_any_noise = 0.5;
    stack_backprop_iter = 0;
    individual_learning_rates = false;
    weight_decay_l1 = 0.0;
    weight_decay_l2 = 0.0;
    dump_testing_output = 0;
}

void
Auto_Encoder_Trainer::
configure(const std::string & name, const Configuration & config)
{
    config.get(prob_cleared, "prob_cleared");
    config.get(learning_rate, "learning_rate");
    config.get(minibatch_size, "minibatch_size");
    config.get(niter, "niter");
    config.get(verbosity, "verbosity");
    config.get(randomize_order, "randomize_order");
    config.get(sample_proportion, "sample_proportion");
    config.get(test_every, "test_every");
    config.get(prob_any_noise, "prob_any_noise");
    config.get(stack_backprop_iter, "stack_backprop_iter");
    config.get(individual_learning_rates, "individual_learning_rates");
    config.get(weight_decay_l1, "weight_decay_l1");
    config.get(weight_decay_l2, "weight_decay_l2");
    config.get(dump_testing_output, "dump_testing_output");
}

template<typename Float>
distribution<Float>
Auto_Encoder_Trainer::
add_noise(const distribution<Float> & inputs,
          Thread_Context & context,
          bool force_noise) const
{
    distribution<Float> result = inputs;

    if (!force_noise
        && (prob_any_noise == 0.0
            || context.random01() > prob_any_noise)) return result;

    for (unsigned i = 0;  i < inputs.size();  ++i)
        if (context.random01() < prob_cleared)
            result[i] = std::numeric_limits<float>::quiet_NaN();
    
    return result;
}

std::pair<double, double>
Auto_Encoder_Trainer::
train_example(const Auto_Encoder & encoder,
              const distribution<float> & inputs_,
              Parameters & updates,
              Thread_Context & context) const
{
    // What precision do we do the calculations in?
    typedef double Float;

    distribution<Float> inputs(inputs_);

    distribution<Float> noisy_inputs
        = add_noise(inputs, context, false /* force_noise */);

    size_t temp_space_size = encoder.rfprop_temporary_space_required();

    Float temp_space[temp_space_size];

    distribution<Float> reconstruction(inputs.size());

    // Forward propagate (calculate the reconstruction from the noisy
    // input)

    encoder.rfprop(&noisy_inputs[0], temp_space, temp_space_size,
                   &reconstruction[0]);

    // Calculate the error (difference between the reconstruction and the
    // input) and the error gradient

    distribution<Float> error = inputs - reconstruction;
    distribution<Float> derror = -2.0 * error;

    // Backpropagate the error gradient through the parameters

    encoder.rbprop(&noisy_inputs[0], &reconstruction[0],
                   temp_space, temp_space_size,
                   &derror[0], 0 /* input_errors_out */, updates, 1.0);

    // Calculate the exact error as well
    distribution<Float> exact_error;

    if (!equivalent(noisy_inputs, inputs)) {
        distribution<Float> exact_reconstruction
            = encoder.reconstruct(inputs);
        
        exact_error = inputs - exact_reconstruction;
    }
    else exact_error = error;

    return make_pair(exact_error.dotprod(exact_error), error.dotprod(error));
}

namespace {

struct Train_Examples_Job {

    const Auto_Encoder_Trainer & trainer;
    const Auto_Encoder & layer;
    const vector<distribution<float> > & data;
    int first;
    int last;
    const vector<int> & examples;
    const Thread_Context & context;
    int random_seed;
    Parameters_Copy<double> & updates;
    Lock & update_lock;
    double & error_exact;
    double & error_noisy;
    boost::progress_display * progress;

    Train_Examples_Job(const Auto_Encoder_Trainer & trainer,
                       const Auto_Encoder & layer,
                       const vector<distribution<float> > & data,
                       int first, int last,
                       const vector<int> & examples,
                       const Thread_Context & context,
                       int random_seed,
                       Parameters_Copy<double> & updates,
                       Lock & update_lock,
                       double & error_exact,
                       double & error_noisy,
                       boost::progress_display * progress)
        : trainer(trainer),
          layer(layer), data(data), first(first), last(last),
          examples(examples),
          context(context), random_seed(random_seed), updates(updates),
          update_lock(update_lock),
          error_exact(error_exact), error_noisy(error_noisy),
          progress(progress)
    {
    }

    void operator () ()
    {
        Thread_Context thread_context(context);
        thread_context.seed(random_seed);
        
        double total_error_exact = 0.0, total_error_noisy = 0.0;

        Parameters_Copy<double> local_updates(layer, 0.0);

        for (unsigned x = first;  x < last;  ++x) {

            double eex, eno;
            boost::tie(eex, eno)
                = trainer.train_example(layer, data[x],
                                        local_updates,
                                        thread_context);

            total_error_exact += eex;
            total_error_noisy += eno;
        }

        Guard guard(update_lock);

        //cerr << "applying local updates" << endl;
        updates.values += local_updates.values;

        error_exact += total_error_exact;
        error_noisy += total_error_noisy;
        
        if (progress)
            (*progress) += (last - first);
    }
};

} // file scope

std::pair<double, double>
Auto_Encoder_Trainer::
train_iter(Auto_Encoder & encoder,
           const std::vector<distribution<float> > & data,
           Thread_Context & thread_context,
           double learning_rate) const
{
    Worker_Task & worker = thread_context.worker();

    int nx = data.size();
    int ni JML_UNUSED = encoder.inputs();
    int no JML_UNUSED = encoder.outputs();

    int microbatch_size = minibatch_size / (num_threads() * 4);
            
    Lock update_lock;

    double total_mse_exact = 0.0, total_mse_noisy = 0.0;
    
    vector<int> examples;
    for (unsigned x = 0;  x < nx;  ++x) {
        // Randomly exclude some samples
        if (thread_context.random01() >= sample_proportion)
            continue;
        examples.push_back(x);
    }
    
    if (randomize_order) {
        Thread_Context::RNG_Type rng = thread_context.rng();
        std::random_shuffle(examples.begin(), examples.end(), rng);
    }
    
    int nx2 = examples.size();

    std::auto_ptr<boost::progress_display> progress;
    if (verbosity >= 3) progress.reset(new boost::progress_display(nx2, cerr));

    for (unsigned x = 0;  x < nx2;  x += minibatch_size) {
                
        Parameters_Copy<double> updates(encoder.parameters(), 0.0);
                
        // Now, submit it as jobs to the worker task to be done
        // multithreaded
        int group;
        {
            int parent = -1;  // no parent group
            group = worker.get_group(NO_JOB, "dump user results task",
                                     parent);
                    
            // Make sure the group gets unlocked once we've populated
            // everything
            Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                         boost::ref(worker),
                                         group));
                    
                    
            for (unsigned x2 = x;  x2 < nx2 && x2 < x + minibatch_size;
                 x2 += microbatch_size) {
                        
                Train_Examples_Job job(*this,
                                       encoder,
                                       data,
                                       x2,
                                       min<int>(nx2,
                                                min(x + minibatch_size,
                                                    x2 + microbatch_size)),
                                       examples,
                                       thread_context,
                                       thread_context.random(),
                                       updates,
                                       update_lock,
                                       total_mse_exact,
                                       total_mse_noisy,
                                       progress.get());

                // Send it to a thread to be processed
                worker.add(job, "blend job", group);
            }
        }
                
        worker.run_until_finished(group);

        // If we have weight decay, include that in the parameter updates
        // * l1 weight decay: we move each parameter towards zero by the
        //   same amount;
        // * l2 weight decay: we move each parameter towards zero by an
        //   amount proportional to the value of the parameter
        // ...

        //cerr << "applying minibatch updates" << endl;
        
        encoder.parameters().update(updates, -learning_rate);
    }

    return make_pair(sqrt(total_mse_exact / nx2), sqrt(total_mse_noisy / nx2));
}

std::pair<double, double>
Auto_Encoder_Trainer::
train_iter(Auto_Encoder & encoder,
           const std::vector<distribution<float> > & data,
           Thread_Context & thread_context,
           const Parameters_Copy<float> & learning_rates) const
{
    Worker_Task & worker = thread_context.worker();

    int nx = data.size();
    int ni JML_UNUSED = encoder.inputs();
    int no JML_UNUSED = encoder.outputs();

    int microbatch_size = minibatch_size / (num_threads() * 4);
            
    Lock update_lock;

    double total_mse_exact = 0.0, total_mse_noisy = 0.0;
    
    vector<int> examples;
    for (unsigned x = 0;  x < nx;  ++x) {
        // Randomly exclude some samples
        if (thread_context.random01() >= sample_proportion)
            continue;
        examples.push_back(x);
    }
    
    if (randomize_order) {
        Thread_Context::RNG_Type rng = thread_context.rng();
        std::random_shuffle(examples.begin(), examples.end(), rng);
    }
    
    int nx2 = examples.size();

    std::auto_ptr<boost::progress_display> progress;
    if (verbosity >= 3) progress.reset(new boost::progress_display(nx2, cerr));

    for (unsigned x = 0;  x < nx2;  x += minibatch_size) {
                
        Parameters_Copy<double> updates(encoder.parameters(), 0.0);
                
        // Now, submit it as jobs to the worker task to be done
        // multithreaded
        int group;
        {
            int parent = -1;  // no parent group
            group = worker.get_group(NO_JOB, "dump user results task",
                                     parent);
                    
            // Make sure the group gets unlocked once we've populated
            // everything
            Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                         boost::ref(worker),
                                         group));
                    
                    
            for (unsigned x2 = x;  x2 < nx2 && x2 < x + minibatch_size;
                 x2 += microbatch_size) {
                        
                Train_Examples_Job job(*this,
                                       encoder,
                                       data,
                                       x2,
                                       min<int>(nx2,
                                                min(x + minibatch_size,
                                                    x2 + microbatch_size)),
                                       examples,
                                       thread_context,
                                       thread_context.random(),
                                       updates,
                                       update_lock,
                                       total_mse_exact,
                                       total_mse_noisy,
                                       progress.get());

                // Send it to a thread to be processed
                worker.add(job, "blend job", group);
            }
        }
                
        worker.run_until_finished(group);

        //cerr << "applying minibatch updates" << endl;
        
        updates.values *= -1.0 / minibatch_size;

        encoder.parameters().update(updates, learning_rates);
    }

    return make_pair(sqrt(total_mse_exact / nx2), sqrt(total_mse_noisy / nx2));
}

void
Auto_Encoder_Trainer::
train(Auto_Encoder & encoder,
      const std::vector<distribution<float> > & training_data,
      const std::vector<distribution<float> > & testing_data,
      Thread_Context & thread_context,
      int niter) const
{
    if (niter == -1) niter = this->niter;

    double learning_rate = this->learning_rate;
    
    int nx = training_data.size();
    int nxt = testing_data.size();

    if (verbosity == 2)
        cerr << "iter      lr  -- inst train--  ---- train ----  ---- test -----\n"
             << "----  ------    exact   noisy    exact   noisy    exact   noisy\n";
    
    Parameters_Copy<float> learning_rates;

    for (unsigned iter = 0;  iter < niter;  ++iter) {

        if (verbosity >= 2)
            cerr << format("%4d", iter) << flush;

        if (iter % 5 == 0 && !individual_learning_rates) {
#if 0
            learning_rate
                = calc_learning_rate(encoder, training_data, thread_context);
            //cerr << "optimal learning rate calculated was " << learning_rate
            //     << endl;

            learning_rate /= 3.0 * nx * sample_proportion;

            cerr << "calculated learning rate: " << learning_rate
                 << " heuristic learning rate: "
                 << (0.75 / (nx * sample_proportion))
                 << " ratio "
                 << (learning_rate / (0.75 / (nx * sample_proportion)))
                 << endl;
#endif

            learning_rate = this->learning_rate / (nx * sample_proportion);

            //cerr << "learning_rate = " << learning_rate << " nx = " << nx
            //     << endl;
        }
        else if (iter % 5 == 0 && individual_learning_rates) {
            learning_rates
                = calc_learning_rates(encoder, training_data, thread_context);
        }

        if (verbosity >= 3)
            cerr << "iter " << iter << " training on " << nx << " examples"
                 << endl;
        else if (verbosity >= 2)
            cerr << format("  %6.4f", learning_rate) << flush;
        Timer timer;
        
        double train_error_exact, train_error_noisy;

        if (!individual_learning_rates)
            boost::tie(train_error_exact, train_error_noisy)
                = train_iter(encoder, training_data, thread_context,
                             learning_rate);
        else 
            boost::tie(train_error_exact, train_error_noisy)
                = train_iter(encoder, training_data, thread_context,
                             learning_rates);
        
        if (verbosity >= 3) {
            cerr << "rmse of iteration: exact " << train_error_exact
                 << " noisy " << train_error_noisy << endl;
            if (verbosity >= 3) cerr << timer.elapsed() << endl;
        }
        else if (verbosity == 2)
            cerr << format("  %7.5f %7.5f",
                           train_error_exact, train_error_noisy)
                 << flush;
        
        if (iter % test_every == (test_every - 1)
            || iter == niter - 1) {
            timer.restart();

            double train_error_exact = 0.0, train_error_noisy = 0.0;
            
            if (verbosity >= 3)
                cerr << "testing on " << nxt << " examples"
                     << endl;
            
            boost::tie(train_error_exact, train_error_noisy)
                = test(encoder, training_data, thread_context);
            
            if (verbosity >= 3) {
                cerr << "training rmse of iteration: exact "
                     << train_error_exact << " noisy " << train_error_noisy
                     << endl;
                cerr << timer.elapsed() << endl;
            }
            else if (verbosity == 2)
                cerr << format("  %7.5f %7.5f",
                               train_error_exact, train_error_noisy);

            double test_error_exact = 0.0, test_error_noisy = 0.0;
            
            if (verbosity >= 3)
                cerr << "testing on " << nxt << " examples"
                     << endl;
            
            boost::tie(test_error_exact, test_error_noisy)
                = test(encoder, testing_data, thread_context);
            
            if (verbosity >= 3) {
                cerr << "testing rmse of iteration: exact "
                     << test_error_exact << " noisy " << test_error_noisy
                     << endl;
                cerr << timer.elapsed() << endl;
            }
            else if (verbosity == 2)
                cerr << format("  %7.5f %7.5f",
                               test_error_exact, test_error_noisy);
        }
        
        if (verbosity == 2) cerr << endl;
    }
}

double
Auto_Encoder_Trainer::
calc_learning_rate(const Auto_Encoder & layer,
                   const std::vector<distribution<float> > & training_data,
                   Thread_Context & thread_context) const
{
    // See http://videolectures.net/eml07_lecun_wia/ and the slides at
    // http://carbon.videolectures.net/2007/pascal/eml07_whistler/lecun_yann/eml07_lecun_wia_01.pdf especially slide 48.

    // 1.  Pick an initial eigenvector estimate at random
    
    Parameters_Copy<double> eig(layer);
    for (unsigned i = 0;  i < eig.values.size();  ++i)
        eig.values[i] = 1 / sqrt(eig.values.size());

    // Get a mutable version so that we can modify its parameters
    auto_ptr<Auto_Encoder> modified(layer.deep_copy());
    
    // Current set of parameters
    Parameters_Copy<double> params(*modified);

    // Temporary space for the fprop/bprop
    size_t temp_space_size = layer.rfprop_temporary_space_required();
    float temp_space[temp_space_size];

    double alpha = 0.0001;
    double gamma = 0.05;

    for (unsigned i = 0;  i < 500;  ++i) {
        if (i == 100) gamma = 0.01;
        if (i == 300) gamma = 0.005;

        //cerr << "i = " << i << " two_norm = " << eig.values.two_norm()
        //     << endl;

        int example_num = min<int>(training_data.size() - 1,
                                   thread_context.random01()
                                   * training_data.size());

        const distribution<float> & exact_input = training_data[example_num];
        distribution<float> noisy_input
            = add_noise(exact_input, thread_context, false /* force_noise */);
        
        distribution<float> reconstructed(layer.inputs());

        Parameters_Copy<double> gradient1(eig, 0.0);

        layer.rfprop(&noisy_input[0], temp_space, temp_space_size,
                     &reconstructed[0]);

        distribution<float> errors = -2.0 * (exact_input - reconstructed);
        
        layer.rbprop(&noisy_input[0], &reconstructed[0],
                     temp_space, temp_space_size, &errors[0], 0,
                     gradient1, 1.0);

        Parameters_Copy<double> new_params = params;
        new_params.values += alpha / eig.values.two_norm() * eig.values;

        modified->parameters().set(new_params);

        modified->rfprop(&noisy_input[0], temp_space, temp_space_size,
                         &reconstructed[0]);
        
        errors = -2.0 * (exact_input - reconstructed);

        Parameters_Copy<double> gradient2(eig, 0.0);
        
        modified->rbprop(&noisy_input[0], &reconstructed[0],
                         temp_space, temp_space_size, &errors[0], 0,
                         gradient2, 1.0);
        
        distribution<double> dgradient = gradient2.values - gradient1.values;
        dgradient *= (gamma / alpha);

        eig.values = (1.0 - gamma) * eig.values + gamma * dgradient;

        //distribution<float> values(eig.values.begin(), eig.values.begin() + 20);
        //cerr << "values = " << values << endl;
    }

    //distribution<float> values(eig.values.begin(), eig.values.begin() + 20);
    //cerr << "values = " << values << endl;

    return 1.0 / eig.values.two_norm();
}

Parameters_Copy<float>
Auto_Encoder_Trainer::
calc_learning_rates(const Auto_Encoder & layer,
                    const std::vector<distribution<float> > & training_data,
                    Thread_Context & thread_context) const
{
    // Where we store the average hessian
    Parameters_Copy<double> avg_hessian_diag(layer, 0.0);

    // 1.  Pick an initial eigenvector estimate at random
    
    Parameters_Copy<double> eig(layer);
    for (unsigned i = 0;  i < eig.values.size();  ++i)
        eig.values[i] = 1 / sqrt(eig.values.size());

    // Get a mutable version so that we can modify its parameters
    auto_ptr<Auto_Encoder> modified(layer.deep_copy());
    
    // Current set of parameters
    Parameters_Copy<double> params(*modified);

    // Temporary space for the fprop/bprop
    size_t temp_space_size = layer.rfprop_temporary_space_required();
    float temp_space[temp_space_size];

    double alpha = 0.0001;
    double gamma = 0.05;

    for (unsigned i = 0;  i < 500;  ++i) {
        if (i == 100) gamma = 0.01;
        if (i == 300) gamma = 0.005;

        //cerr << "i = " << i << " two_norm = " << eig.values.two_norm()
        //     << endl;

        int example_num = min<int>(training_data.size() - 1,
                                   thread_context.random01()
                                   * training_data.size());

        const distribution<float> & exact_input = training_data[example_num];
        distribution<float> noisy_input
            = add_noise(exact_input, thread_context, false /* force_noise */);
        
        distribution<float> reconstructed(layer.inputs());

        Parameters_Copy<double> gradient1(eig, 0.0);

        layer.rfprop(&noisy_input[0], temp_space, temp_space_size,
                     &reconstructed[0]);

        distribution<float> errors = 2.0 * (reconstructed - exact_input);
        
        distribution<float> derrors(layer.inputs(), 2.0);
        
        layer.rbbprop(&noisy_input[0], &reconstructed[0],
                      temp_space, temp_space_size, &errors[0],
                      &derrors[0], 0, 0,
                      gradient1, &avg_hessian_diag, 1.0);
        
        Parameters_Copy<double> new_params = params;
        new_params.values += alpha / eig.values.two_norm() * eig.values;

        modified->parameters().set(new_params);

        modified->rfprop(&noisy_input[0], temp_space, temp_space_size,
                         &reconstructed[0]);
        
        errors = -2.0 * (exact_input - reconstructed);

        Parameters_Copy<double> gradient2(eig, 0.0);
        
        modified->rbprop(&noisy_input[0], &reconstructed[0],
                         temp_space, temp_space_size, &errors[0], 0,
                         gradient2, 1.0);
        
        distribution<double> dgradient = gradient2.values - gradient1.values;
        dgradient *= (gamma / alpha);

        eig.values = (1.0 - gamma) * eig.values + dgradient;

        //distribution<float> values(eig.values.begin(), eig.values.begin() + 20);
        //cerr << "values = " << values << endl;
    }

    //distribution<float> values(eig.values.begin(), eig.values.begin() + 20);
    //cerr << "values = " << values << endl;

    distribution<float> ahd_values(avg_hessian_diag.values.begin(),
                                   avg_hessian_diag.values.begin() + 20);
    ahd_values /= 500;
    cerr << "avg_hessian_diag = " << ahd_values << endl;

    double base_rate = 1.0 / eig.values.two_norm();
    Parameters_Copy<float> result(params, 0.0);

    // Limit our updates to 10 times faster than the base rate
    double mu = 0.1;
    
    result.values = base_rate / ((abs(avg_hessian_diag.values) / 500) + mu);

    double avg_result = result.values.mean();

    cerr << "avg_result = " << avg_result << endl;

    cerr << "base_rate = " << base_rate << endl;

    cerr << "result.values = "
         << distribution<float>(result.values.begin(),
                                result.values.begin() + 20)
         << endl;

    result.values *= 2.0 * base_rate / avg_result;

    //result.values.fill(base_rate);

    return result;
}

void
Auto_Encoder_Trainer::
train_stack(Auto_Encoder_Stack & stack,
            const std::vector<distribution<float> > & training_data,
            const std::vector<distribution<float> > & testing_data,
            Thread_Context & thread_context) const
{
    int nx = training_data.size();
    int nxt = testing_data.size();

    if (nx == 0)
        throw Exception("can't train on no data");

    int nlayers = stack.size();

    vector<distribution<float> > layer_train = training_data;
    vector<distribution<float> > layer_test = testing_data;

    // Learning rate is per-example
    double learning_rate = this->learning_rate / nx;

    // Compensate for the example proportion
    learning_rate /= sample_proportion;

    Auto_Encoder_Stack test_stack("test");

    for (unsigned layer_num = 0;  layer_num < nlayers;  ++layer_num) {
        cerr << endl << endl << endl << "--------- LAYER " << layer_num
             << " ---------" << endl << endl;

        Auto_Encoder & layer = stack[layer_num];

        vector<distribution<float> > next_layer_train, next_layer_test;

        int ni = layer.inputs();

        if (ni != layer_train[0].size())
            throw Exception("ni is wrong");

        train(layer, layer_train, layer_test, thread_context, niter);

        next_layer_train.resize(nx);
        next_layer_test.resize(nxt);

        // Add it to the testing stack so that we can test up to here
        test_stack.add(make_unowned_sp(layer));

        if (stack_backprop_iter > 0 && layer_num != 0) {

            // Test the layer stack
            if (verbosity >= 3)
                cerr << "calculating whole stack testing performance on "
                     << nxt << " examples" << endl;
            
            if (verbosity >= 1) {
                double test_error_exact = 0.0, test_error_noisy = 0.0;
                boost::tie(test_error_exact, test_error_noisy)
                    = test(test_stack, testing_data, thread_context);
                
                cerr << "testing rmse of stack: exact "
                     << test_error_exact << " noisy " << test_error_noisy
                     << endl;
                
                cerr << endl << endl << "training whole stack backprop" << endl;
                train(test_stack, training_data, testing_data, thread_context,
                      stack_backprop_iter);
            }
        }

        // Calculate the inputs to the next layer
        
        if (verbosity >= 3)
            cerr << "calculating next layer training inputs on "
                 << nx << " examples" << endl;
        double train_error_exact = 0.0, train_error_noisy = 0.0;
        boost::tie(train_error_exact, train_error_noisy)
            = test_and_update(layer, layer_train, next_layer_train,
                              thread_context);
        
        if (verbosity >= 2)
            cerr << "training rmse of layer: exact "
                 << train_error_exact << " noisy " << train_error_noisy
                 << endl;
        
        if (verbosity >= 3)
            cerr << "calculating next layer testing inputs on "
                 << nxt << " examples" << endl;
        double test_error_exact = 0.0, test_error_noisy = 0.0;
        boost::tie(test_error_exact, test_error_noisy)
            = test_and_update(layer, layer_test, next_layer_test,
                              thread_context);
        
        if (verbosity >= 2)
            cerr << "testing rmse of layer: exact "
                 << test_error_exact << " noisy " << test_error_noisy
                 << endl;

        layer_train.swap(next_layer_train);
        layer_test.swap(next_layer_test);

        // Test the layer stack
        if (verbosity >= 3)
            cerr << "calculating whole stack testing performance on "
                 << nxt << " examples" << endl;
        
        if (verbosity >= 1) {
            boost::tie(test_error_exact, test_error_noisy)
                = test(test_stack, testing_data, thread_context);

            cerr << "testing rmse of stack: exact "
                 << test_error_exact << " noisy " << test_error_noisy
                 << endl;
        }
    }
}


namespace {

std::string print_dist(const distribution<float> & dist)
{
    string result;
    for (unsigned i = 0;  i < min<int>(dist.size(), 12);  ++i)
        result += format("%6.3f ", dist[i]);
    if (dist.size() > 20) result += "...";
    return result;
}

struct Test_Examples_Job {

    const Auto_Encoder_Trainer & trainer;
    const Auto_Encoder & layer;
    const vector<distribution<float> > & data_in;
    vector<distribution<float> > & data_out;
    int first;
    int last;
    const Thread_Context & context;
    int random_seed;
    Lock & update_lock;
    double & error_exact;
    double & error_noisy;
    boost::progress_display * progress;

    Test_Examples_Job(const Auto_Encoder_Trainer & trainer,
                      const Auto_Encoder & layer,
                      const vector<distribution<float> > & data_in,
                      vector<distribution<float> > & data_out,
                      int first, int last,
                      const Thread_Context & context,
                      int random_seed,
                      Lock & update_lock,
                      double & error_exact,
                      double & error_noisy,
                      boost::progress_display * progress)
        : trainer(trainer), layer(layer), data_in(data_in), data_out(data_out),
          first(first), last(last),
          context(context), random_seed(random_seed),
          update_lock(update_lock),
          error_exact(error_exact), error_noisy(error_noisy),
          progress(progress)
    {
    }

    void operator () ()
    {
        Thread_Context thread_context(context);
        thread_context.seed(random_seed);

        double test_error_exact = 0.0, test_error_noisy = 0.0;

        for (unsigned x = first;  x < last;  ++x) {
            int ni JML_UNUSED = layer.inputs();
            int no JML_UNUSED = layer.outputs();

            // Present this input
            distribution<float> model_input(data_in[x]);
            
            distribution<bool> was_cleared;

            // Add noise
            distribution<float> noisy_input
                = trainer.add_noise(model_input, thread_context,
                                    true /* force_noise */);
            
            // Apply the layer
            distribution<float> hidden_rep
                = layer.apply(noisy_input);
            
            // Reconstruct the input
            distribution<float> denoised_input
                = layer.iapply(hidden_rep);
            
            // Error signal
            distribution<float> diff
                = model_input - denoised_input;
    
            // Overall error
            double error = pow(diff.two_norm(), 2);

            test_error_noisy += error;


            // Apply the layer
            distribution<float> hidden_rep2
                = layer.apply(model_input);

            if (!data_out.empty())
                data_out.at(x) = hidden_rep2.cast<float>();
            
            // Reconstruct the input
            distribution<float> reconstructed_input
                = layer.iapply(hidden_rep2);
            
            // Error signal
            distribution<float> diff2
                = model_input - reconstructed_input;
    
            // Overall error
            double error2 = pow(diff2.two_norm(), 2);

            if (x < trainer.dump_testing_output) {
                cerr << endl
                     << "example " << x << ": error exact " << error2
                     << " noisy " << error << endl
                     << "  input: " << print_dist(model_input) << endl
                     << "  output:" << print_dist(reconstructed_input) << endl
                     << "  diff:  " << print_dist(diff2) << endl;
                if (isnan(noisy_input).any()) {
                    cerr << "  noisy: " << print_dist(noisy_input) << endl
                         << "  dnoisd:" << print_dist(denoised_input) << endl
                         << "  diff:  " << print_dist(diff)
                         << endl;
                }
            }
    
            test_error_exact += error2;
        }

        Guard guard(update_lock);
        error_exact += test_error_exact;
        error_noisy += test_error_noisy;
        if (progress)
            (*progress) += (last - first);
    }
};

} // file scope

std::pair<double, double>
Auto_Encoder_Trainer::
test_and_update(const Auto_Encoder & encoder,
                const std::vector<distribution<float> > & data_in,
                std::vector<distribution<float> > & data_out,
                Thread_Context & thread_context) const
{
    Lock update_lock;
    double error_exact = 0.0;
    double error_noisy = 0.0;

    int nx = data_in.size();

    std::auto_ptr<boost::progress_display> progress;
    if (verbosity >= 3) progress.reset(new boost::progress_display(nx, cerr));

    Worker_Task & worker = thread_context.worker();
            
    // Now, submit it as jobs to the worker task to be done
    // multithreaded
    int group;
    {
        int parent = -1;  // no parent group
        group = worker.get_group(NO_JOB, "dump user results task",
                                 parent);
        
        // Make sure the group gets unlocked once we've populated
        // everything
        Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                     boost::ref(worker),
                                     group));
        
        // 20 jobs per CPU
        int batch_size = nx / (num_threads() * 20);
        
        for (unsigned x = 0; x < nx;  x += batch_size) {
            
            Test_Examples_Job job(*this, encoder, data_in, data_out,
                                  x, min<int>(x + batch_size, nx),
                                  thread_context,
                                  thread_context.random(),
                                  update_lock,
                                  error_exact, error_noisy,
                                  progress.get());
            
            // Send it to a thread to be processed
            worker.add(job, "blend job", group);
        }
    }

    worker.run_until_finished(group);

    return make_pair(sqrt(error_exact / nx),
                     sqrt(error_noisy / nx));
}
    
} // namespace ML


#if 0

            cerr << "weights: " << endl;
            for (unsigned i = 0;  i < 10;  ++i) {
                for (unsigned j = 0;  j < 10;  ++j) {
                    cerr << format("%7.4f", layer.weights[i][j]);
                }
                cerr << endl;
            }
            
            double max_abs_weight = 0.0;
            double total_abs_weight = 0.0;
            double total_weight_sqr = 0.0;
            for (unsigned i = 0;  i < ni;  ++i) {
                for (unsigned j = 0;  j < nh;  ++j) {
                    double abs_weight = abs(layer.weights[i][j]);
                    max_abs_weight = std::max(max_abs_weight, abs_weight);
                    total_abs_weight += abs_weight;
                    total_weight_sqr += abs_weight * abs_weight;
                }
            }

            double avg_abs_weight = total_abs_weight / (ni * nh);
            double rms_avg_weight = sqrt(total_weight_sqr / (ni * nh));

            cerr << "max = " << max_abs_weight << " avg = "
                 << avg_abs_weight << " rms avg = " << rms_avg_weight
                 << endl;

            //cerr << "iscales: " << layer.iscales << endl;
            //cerr << "hscales: " << layer.hscales << endl;
            //cerr << "bias: " << layer.bias << endl;
            //cerr << "ibias: " << layer.ibias << endl;

            distribution<LFloat> svalues(min(ni, nh));
            boost::multi_array<LFloat, 2> layer2 = layer.weights;
            int nvalues = std::min(ni, nh);
        
            boost::multi_array<LFloat, 2> rvectors(boost::extents[ni][nvalues]);
            boost::multi_array<LFloat, 2> lvectorsT(boost::extents[nvalues][nh]);

            int result = LAPack::gesdd("S", nh, ni,
                                       layer2.data(), nh,
                                       &svalues[0],
                                       &lvectorsT[0][0], nh,
                                       &rvectors[0][0], nvalues);
            if (result != 0)
                throw Exception("gesdd returned non-zero");
        

            if (false) {
                boost::multi_array<LFloat, 2> weights2
                    = rvectors * diag(svalues) * lvectorsT;
                
                cerr << "weights2: " << endl;
                for (unsigned i = 0;  i < 10;  ++i) {
                    for (unsigned j = 0;  j < 10;  ++j) {
                        cerr << format("%7.4f", weights2[i][j]);
                    }
                    cerr << endl;
                }
            }

            //if (iter == 0) layer.weights = rvectors * lvectorsT;

            //if (iter == 0) layer.weights = rvectors * lvectorsT;

            //cerr << "svalues = " << svalues << endl;

        if (init_with_svd) {
            // Initialize with a SVD
            SVD_Decomposition init;
            init.train(layer_train, nh);
            
            for (unsigned i = 0;  i < ni;  ++i) {
                distribution<float> init_i(&init.lvectors[i][0],
                                            &init.lvectors[i][0] + nh);
                //init_i /= sqrt(init.singular_values_order);
                //init_i *= init.singular_values_order / init.singular_values_order[0];
                
                std::copy(init_i.begin(), init_i.end(),
                          &layer.weights[i][0]);
                layer.bias.fill(0.0);
                layer.ibias.fill(0.0);
                layer.iscales.fill(1.0);
                layer.hscales.fill(1.0);
                //layer.hscales = init.singular_values_order;
            }
        }


#endif

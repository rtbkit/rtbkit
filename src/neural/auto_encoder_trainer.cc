/* auto_encoder_trainer.cc
   Jeremy Barnes, 11 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Trainer for an auto-encoder.
*/

#include "auto_encoder_trainer.h"
#include "utils/configuration.h"
#include "arch/threads.h"

#include <boost/progress.hpp>
#include "boosting/worker_task.h"
#include <boost/tuple/tuple.hpp>
#include "utils/guard.h"
#include "utils/configuration.h"
#include "arch/timers.h"
#include <boost/bind.hpp>
#include "auto_encoder_stack.h"
#include "utils/check_not_nan.h"


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
}

template<typename Float>
distribution<Float>
Auto_Encoder_Trainer::
add_noise(const distribution<Float> & inputs,
          Thread_Context & context) const
{
    distribution<Float> result = inputs;

    for (unsigned i = 0;  i < inputs.size();  ++i)
        if (context.random01() < prob_cleared)
            result[i] = std::numeric_limits<float>::quiet_NaN();
    
    return result;
}

std::pair<double, double>
Auto_Encoder_Trainer::
train_example(const Auto_Encoder & encoder,
              const distribution<float> & inputs,
              Parameters & updates,
              Thread_Context & context) const
{
    distribution<float> noisy_inputs = add_noise(inputs, context);

    size_t temp_space_size = encoder.rfprop_temporary_space_required();

    float temp_space[temp_space_size];

    distribution<float> reconstruction(inputs.size());

    // Forward propagate (calculate the reconstruction from the noisy
    // input)

    encoder.rfprop(&noisy_inputs[0], temp_space, temp_space_size,
                   &reconstruction[0]);

    // Calculate the error (difference between the reconstruction and the
    // input) and the error gradient

    distribution<float> error = inputs - reconstruction;
    distribution<float> derror = -2.0 * error;

    // Backpropagate the error gradient through the parameters

    encoder.rbprop(&inputs[0], &reconstruction[0],
                   temp_space, temp_space_size,
                   &derror[0], 0 /* input_errors_out */, updates, 1.0);

    // Calculate the exact error as well
    distribution<float> exact_error;

    if (!equivalent(noisy_inputs, inputs)) {
        distribution<float> exact_reconstruction
            = encoder.reconstruct(inputs);
        
        exact_error = inputs - exact_reconstruction;
    }
    else exact_error = error;

    return make_pair(error.dotprod(error), exact_error.dotprod(exact_error));
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
           Thread_Context & thread_context)
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
        
        encoder.parameters().update(updates, learning_rate);
    }

    return make_pair(sqrt(total_mse_exact / nx2), sqrt(total_mse_noisy / nx2));
}

void
Auto_Encoder_Trainer::
train(Auto_Encoder & encoder,
      const std::vector<distribution<float> > & training_data,
      const std::vector<distribution<float> > & testing_data,
      Thread_Context & thread_context)
{
    int nx = training_data.size();
    int nxt = testing_data.size();

    if (verbosity == 2)
        cerr << "iter  ---- train ----  ---- test -----\n"
             << "        exact   noisy    exact   noisy\n";
    
    for (unsigned iter = 0;  iter < niter;  ++iter) {
        if (verbosity >= 3)
            cerr << "iter " << iter << " training on " << nx << " examples"
                 << endl;
        else if (verbosity >= 2)
            cerr << format("%4d", iter) << flush;
        Timer timer;
        
        double train_error_exact, train_error_noisy;
        boost::tie(train_error_exact, train_error_noisy)
            = train_iter(encoder, training_data, thread_context);
        
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

void
Auto_Encoder_Trainer::
train_stack(Auto_Encoder_Stack & stack,
            const std::vector<distribution<float> > & training_data,
            const std::vector<distribution<float> > & testing_data,
            Thread_Context & thread_context)
{
    int nx = training_data.size();
    int nxt = testing_data.size();

    if (nx == 0)
        throw Exception("can't train on no data");

    int nlayers = stack.size();

    vector<distribution<float> > layer_train = training_data;
    vector<distribution<float> > layer_test = testing_data;

    // Learning rate is per-example
    learning_rate /= nx;

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

        train(layer, layer_train, layer_test, thread_context);

        next_layer_train.resize(nx);
        next_layer_test.resize(nxt);

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

        // Add it so that we can test up to here
        test_stack.add(make_unowned_sp(layer));

        // Test the layer stack
        if (verbosity >= 3)
            cerr << "calculating whole stack testing performance on "
                 << nxt << " examples" << endl;
        
        if (verbosity >= 1) {
            boost::tie(test_error_exact, test_error_noisy)
                = test(stack, testing_data, thread_context);

            cerr << "testing rmse of stack: exact "
                 << test_error_exact << " noisy " << test_error_noisy
                 << endl;
        }
    }
}


namespace {

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
                = trainer.add_noise(model_input, thread_context);
            
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
test(const Auto_Encoder & encoder,
     const std::vector<distribution<float> > & data,
     Thread_Context & thread_context)
{
    Lock update_lock;
    double error_exact = 0.0;
    double error_noisy = 0.0;

    int nx = data.size();

    std::auto_ptr<boost::progress_display> progress;
    if (verbosity >= 3) progress.reset(new boost::progress_display(nx, cerr));

    Worker_Task & worker = thread_context.worker();

    // If this is empty, then no data is written out
    vector<distribution<float> > dummy_data_out;
            
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

            Test_Examples_Job job(*this, encoder, data, dummy_data_out,
                                  x, min<int>(x + batch_size, nx),
                                  thread_context,
                                  thread_context.random(),
                                  update_lock,
                                  error_exact, error_noisy,
                                  progress.get());
            
            // Send it to a thread to be processed
            worker.add(job, "test examples job", group);
        }
    }
    
    worker.run_until_finished(group);
    
    return make_pair(sqrt(error_exact / nx),
                     sqrt(error_noisy / nx));
}

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

    config.get(prob_cleared, "prob_cleared");
    config.get(learning_rate, "learning_rate");
    config.get(minibatch_size, "minibatch_size");
    config.get(niter, "niter");
    config.get(verbosity, "verbosity");
    config.get(transfer_function, "transfer_function");
    config.get(init_with_svd, "init_with_svd");
    config.get(use_dense_missing, "use_dense_missing");
    config.get(layer_sizes, "layer_sizes");
    config.get(randomize_order, "randomize_order");
    config.get(sample_proportion, "sample_proportion");
    config.get(test_every, "test_every");


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
#endif

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

        //layer.zero_fill();
        //layer.bias.fill(0.01);
        //layer.weights[0][0] = -0.3;

        Twoway_Layer layer(use_dense_missing, ni, nh, transfer_function,
                           thread_context);

        if (ni == nh && false) {
            //layer.zero_fill();
            for (unsigned i = 0;  i < ni;  ++i) {
                layer.weights[i][i] += 1.0;
            }
        }

        // Calculate the inputs to the next layer
        
        if (verbosity >= 3)
            cerr << "calculating next layer training inputs on "
                 << nx << " examples" << endl;
        double train_error_exact = 0.0, train_error_noisy = 0.0;
        boost::tie(train_error_exact, train_error_noisy)
            = layer.test_and_update(layer_train, next_layer_train,
                                    prob_cleared, thread_context,
                                    verbosity);

        if (verbosity >= 2)
            cerr << "training rmse of layer: exact "
                 << train_error_exact << " noisy " << train_error_noisy
                 << endl;
        
        if (verbosity >= 3)
            cerr << "calculating next layer testing inputs on "
                 << nxt << " examples" << endl;
        double test_error_exact = 0.0, test_error_noisy = 0.0;
        boost::tie(test_error_exact, test_error_noisy)
            = layer.test_and_update(layer_test, next_layer_test,
                                    prob_cleared, thread_context,
                                    verbosity);

#if 0
        push_back(layer);

        // Test the layer stack
        if (verbosity >= 3)
            cerr << "calculating whole stack testing performance on "
                 << nxt << " examples" << endl;
        boost::tie(test_error_exact, test_error_noisy)
            = test(testing_data, prob_cleared, thread_context, verbosity);
        
        if (verbosity >= 2)
            cerr << "testing rmse of stack: exact "
                 << test_error_exact << " noisy " << test_error_noisy
                 << endl;
        
        if (verbosity >= 2)
            cerr << "testing rmse of layer: exact "
                 << test_error_exact << " noisy " << test_error_noisy
                 << endl;

        pop_back();
#endif


#endif

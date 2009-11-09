/* discriminative_trainer.cc
   Jeremy Barnes, 9 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Discriminative training for neural network architectures via backprop.
*/

#include "discriminative_trainer.h"
#include "boosting/worker_task.h"
#include "utils/guard.h"
#include <boost/progress.hpp>
#include <boost/tuple/tuple.hpp>
#include "arch/threads.h"
#include <boost/bind.hpp>
#include "arch/timers.h"
#include "stats/auc.h"

using namespace std;


namespace ML {


/*****************************************************************************/
/* DISCRIMINATIVE_TRAINER                                                    */
/*****************************************************************************/

pair<double, double>
Discriminative_Trainer::
train_example(const distribution<float> & data,
              float label,
              Parameters_Copy<double> & updates) const
{
    distribution<float> label2(1, label);
    return train_example(data, label2, updates);
}

pair<double, double>
Discriminative_Trainer::
train_example(const distribution<float> & data,
              const distribution<float> & label,
              Parameters_Copy<double> & updates) const
{
    /* fprop */
    
    size_t temp_space_required
        = layer->fprop_temporary_space_required();

    float temp_space[temp_space_required];

    distribution<float> outputs
        = layer->fprop(data, temp_space, temp_space_required);

    /* error */

    distribution<float> errors = (label - outputs);
    double error = errors.dotprod(errors);

    // TODO: get the loss function to do this...
    distribution<float> derrors = -2.0 * errors;

    distribution<float> input_derrors;

    /* bprop */
    layer->bprop(derrors,
                 temp_space, temp_space_required,
                 updates,
                 input_derrors, 1.0, false /* calculate_input_errors */);

    return make_pair(sqrt(error), outputs[0]);
}

namespace {

struct Train_Examples_Job {

    const Discriminative_Trainer & trainer;
    const std::vector<distribution<float> > & data;
    const std::vector<float> & labels;
    Thread_Context & thread_context;
    const vector<int> & examples;
    int first;
    int last;
    Parameters_Copy<double> & updates;
    vector<float> & outputs;
    double & total_rmse;
    Lock & updates_lock;
    boost::progress_display * progress;
    int verbosity;

    Train_Examples_Job(const Discriminative_Trainer & trainer,
                       const std::vector<distribution<float> > & data,
                       const std::vector<float> & labels,
                       Thread_Context & thread_context,
                       const vector<int> & examples,
                       int first, int last,
                       Parameters_Copy<double> & updates,
                       vector<float> & outputs,
                       double & total_rmse,
                       Lock & updates_lock,
                       boost::progress_display * progress,
                       int verbosity)
        : trainer(trainer), data(data), labels(labels),
          thread_context(thread_context), examples(examples),
          first(first), last(last), updates(updates), outputs(outputs),
          total_rmse(total_rmse), updates_lock(updates_lock),
          progress(progress), verbosity(verbosity)
    {
    }

    void operator () ()
    {
        Parameters_Copy<double> local_updates(*trainer.layer);

        double total_rmse_local = 0.0;

        for (unsigned ix = first; ix < last;  ++ix) {
            int x = examples[ix];

            double rmse_contribution;
            double output;

            boost::tie(rmse_contribution, output)
                = trainer.train_example(data[x],
                                      labels[x],
                                      local_updates);
            
            outputs[ix] = output;
            total_rmse_local += rmse_contribution;
        }

        Guard guard(updates_lock);
        total_rmse += total_rmse_local;
        updates.values += local_updates.values;
        if (progress) progress += (last - first);
    }
};

} // file scope

std::pair<double, double>
Discriminative_Trainer::
train_iter(const std::vector<distribution<float> > & data,
           const std::vector<float> & labels,
           Thread_Context & thread_context,
           int minibatch_size, float learning_rate,
           int verbosity,
           float sample_proportion,
           bool randomize_order) const
{
    Worker_Task & worker = thread_context.worker();

    int nx = data.size();

    int microbatch_size = minibatch_size / (num_threads() * 4);
            
    Lock update_lock;

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

    double total_mse = 0.0;
    distribution<float> outputs;
    outputs.resize(nx2);    

    std::auto_ptr<boost::progress_display> progress;
    if (verbosity >= 3) progress.reset(new boost::progress_display(nx2, cerr));

    for (unsigned x = 0;  x < nx2;  x += minibatch_size) {
                
        Parameters_Copy<double> updates(*layer);
                
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
                        
                Train_Examples_Job
                    job(*this,
                        data,
                        labels,
                        thread_context,
                        examples,
                        x2,
                        min<int>(nx2,
                                 min(x + minibatch_size,
                                     x2 + microbatch_size)),
                        updates,
                        outputs,
                        total_mse,
                        update_lock,
                        progress.get(),
                        verbosity);

                // Send it to a thread to be processed
                worker.add(job, "backprop job", group);
            }
        }
        
        worker.run_until_finished(group);

        //cerr << "applying minibatch updates" << endl;
        
        layer->parameters().update(updates, learning_rate);
    }

    // TODO: calculate AUC score
    distribution<float> test_labels;
    for (unsigned i = 0;  i < nx2;  ++i)
        test_labels.push_back(labels[examples[i]]);

    double auc = calc_auc(outputs, test_labels);

    return make_pair(sqrt(total_mse / nx2), auc);
}

std::pair<double, double>
Discriminative_Trainer::
train(const std::vector<distribution<float> > & training_data,
      const std::vector<float> & training_labels,
      const std::vector<distribution<float> > & testing_data,
      const std::vector<float> & testing_labels,
      const Configuration & config,
      ML::Thread_Context & thread_context) const
{
    double learning_rate = 0.75;
    int minibatch_size = 512;
    int niter = 50;
    int verbosity = 2;

    Transfer_Function_Type transfer_function = TF_TANH;

    bool randomize_order = true;
    float sample_proportion = 0.8;
    int test_every = 1;

    config.get(learning_rate, "learning_rate");
    config.get(minibatch_size, "minibatch_size");
    config.get(niter, "niter");
    config.get(verbosity, "verbosity");
    config.get(transfer_function, "transfer_function");
    config.get(randomize_order, "randomize_order");
    config.get(sample_proportion, "sample_proportion");
    config.get(test_every, "test_every");

    int nx = training_data.size();

    if (training_data.size() != training_labels.size())
        throw Exception("label and example sizes don't match");


    int nxt = testing_data.size();

    if (nx == 0)
        throw Exception("can't train on no data");

    // Learning rate is per-example
    learning_rate /= nx;

    // Compensate for the example proportion
    learning_rate /= sample_proportion;

    if (verbosity == 2)
        cerr << "iter  ---- train ----  ---- test -----\n"
             << "         rmse     auc     rmse     auc\n";
    
    for (unsigned iter = 0;  iter < niter;  ++iter) {
        if (verbosity >= 3)
            cerr << "iter " << iter << " training on " << nx << " examples"
                 << endl;
        else if (verbosity >= 2)
            cerr << format("%4d", iter) << flush;
        Timer timer;

        double train_error_rmse, train_error_auc;
        boost::tie(train_error_rmse, train_error_auc)
            = train_iter(training_data, training_labels,
                         thread_context,
                         minibatch_size, learning_rate,
                         verbosity, sample_proportion,
                         randomize_order);
        
        if (verbosity >= 3) {
            cerr << "error of iteration: rmse " << train_error_rmse
                 << " noisy " << train_error_auc << endl;
            if (verbosity >= 3) cerr << timer.elapsed() << endl;
        }
        else if (verbosity == 2)
            cerr << format("  %7.5f %7.5f",
                           train_error_rmse, train_error_auc)
                 << flush;
        
        if (iter % test_every == (test_every - 1)
            || iter == niter - 1) {
            timer.restart();
            double test_error_rmse = 0.0, test_error_auc = 0.0;
                
            if (verbosity >= 3)
                cerr << "testing on " << nxt << " examples"
                     << endl;

            boost::tie(test_error_rmse, test_error_auc)
                = test(testing_data, testing_labels,
                       thread_context, verbosity);
            
            if (verbosity >= 3) {
                cerr << "testing error of iteration: rmse "
                     << test_error_rmse << " auc " << test_error_auc
                     << endl;
                cerr << timer.elapsed() << endl;
            }
            else if (verbosity == 2)
                cerr << format("  %7.5f %7.5f",
                               test_error_rmse, test_error_auc);
        }
        
        if (verbosity == 2) cerr << endl;
    }

    // TODO: return something
    return make_pair(0.0, 0.0);
}

namespace {

struct Test_Examples_Job {

    const Discriminative_Trainer & trainer;
    const vector<distribution<float> > & data;
    const vector<float> & labels;
    int first;
    int last;
    const Thread_Context & context;
    Lock & update_lock;
    double & error_rmse;
    vector<float> & outputs;
    boost::progress_display * progress;
    int verbosity;

    Test_Examples_Job(const Discriminative_Trainer & trainer,
                      const vector<distribution<float> > & data,
                      const vector<float> & labels,
                      int first, int last,
                      const Thread_Context & context,
                      Lock & update_lock,
                      double & error_rmse,
                      vector<float> & outputs,
                      boost::progress_display * progress,
                      int verbosity)
        : trainer(trainer), data(data), labels(labels),
          first(first), last(last),
          context(context),
          update_lock(update_lock),
          error_rmse(error_rmse), outputs(outputs),
          progress(progress), verbosity(verbosity)
    {
    }

    void operator () ()
    {
        double local_error_rmse = 0.0;

        for (unsigned x = first;  x < last;  ++x) {
            distribution<float> output
                = trainer.layer->apply(data[x]);

            outputs[x] = output[0];
            local_error_rmse += pow(labels[x] - output[0], 2);
        }

        Guard guard(update_lock);
        error_rmse += local_error_rmse;
        if (progress && verbosity >= 3) (*progress) += (last - first);
    }
};

} // file scope

pair<double, double>
Discriminative_Trainer::
test(const std::vector<distribution<float> > & data,
     const std::vector<float> & labels,
     ML::Thread_Context & thread_context,
     int verbosity) const
{
    Lock update_lock;
    double mse_total = 0.0;

    int nx = data.size();

    std::auto_ptr<boost::progress_display> progress;
    if (verbosity >= 3) progress.reset(new boost::progress_display(nx, cerr));

    Worker_Task & worker = thread_context.worker();

    distribution<float> outputs;
    outputs.resize(nx);
            
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
            
            Test_Examples_Job job(*this, data, labels,
                                  x, min<int>(x + batch_size, nx),
                                  thread_context,
                                  update_lock,
                                  mse_total, outputs,
                                  progress.get(),
                                  verbosity);
            
            // Send it to a thread to be processed
            worker.add(job, "test discrim job", group);
        }
    }
    
    worker.run_until_finished(group);
    
    return make_pair(sqrt(mse_total / nx),
                     calc_auc(outputs, labels));
}

} // namespace ML

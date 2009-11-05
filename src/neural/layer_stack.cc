/* layer_stack.cc
   Jeremy Barnes, 4 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Stack of layers.
*/

#include "layer_stack.h"
#include "dense_layer.h"
#include "twoway_layer.h"
#include "layer_stack_impl.h"

namespace ML {

template class Layer_Stack<Layer>;
template class Layer_Stack<Dense_Layer<float> >;
template class Layer_Stack<Dense_Layer<double> >;
template class Layer_Stack<Twoway_Layer>;


#if 0

/*****************************************************************************/
/* DNAE_STACK_UPDATES                                                        */
/*****************************************************************************/

struct DNAE_Stack;

struct DNAE_Stack_Updates : public std::vector<Twoway_Layer_Updates> {

    DNAE_Stack_Updates();

    DNAE_Stack_Updates(const DNAE_Stack & stack);

    void init(const DNAE_Stack & stack);

    DNAE_Stack_Updates & operator += (const DNAE_Stack_Updates & updates);
};


/*****************************************************************************/
/* DNAE_STACK                                                                */
/*****************************************************************************/

/** A stack of denoising autoencoder layers, to create a deep encoder. */

struct DNAE_Stack : public std::vector<Twoway_Layer> {

    distribution<float> apply(const distribution<float> & input) const;
    distribution<double> apply(const distribution<double> & input) const;

    distribution<float> iapply(const distribution<float> & output) const;
    distribution<double> iapply(const distribution<double> & output) const;

    void serialize(ML::DB::Store_Writer & store) const;
    void reconstitute(ML::DB::Store_Reader & store);

    /** Train a single example.  Returns the RMSE in the first and the
        output value (which can be used to calculate the AUC) in the
        second.
    */
    std::pair<double, double>
    train_discrim_example(const distribution<float> & data,
                          float label,
                          DNAE_Stack_Updates & udpates) const;

    /** Trains a single iteration on the given data with the selected
        parameters.  Returns a moving estimate of the RMSE on the
        training set. */
    std::pair<double, double>
    train_discrim_iter(const std::vector<distribution<float> > & data,
                       const std::vector<float> & labels,
                       Thread_Context & thread_context,
                       int minibatch_size, float learning_rate,
                       int verbosity,
                       float sample_proportion,
                       bool randomize_order);

    /** Perform backpropagation given an error gradient.  Note that doing
        so will adversely affect the performance of the autoencoder, as
        the reverse weights aren't modified when performing this training.

        Returns the best training and testing error.
    */
    std::pair<double, double>
    train_discrim(const std::vector<distribution<float> > & training_data,
                  const std::vector<float> & training_labels,
                  const std::vector<distribution<float> > & testing_data,
                  const std::vector<float> & testing_labels,
                  const Configuration & config,
                  ML::Thread_Context & thread_context);

    /** Update given the learning rate and the gradient. */
    void update(const DNAE_Stack_Updates & updates, double learning_rate);
    
    /** Test the discriminative power of the network.  Returns the RMS error
        or AUC depending upon whether it's a regression or classification
        task.
    */
    std::pair<double, double>
    test_discrim(const std::vector<distribution<float> > & data,
                 const std::vector<float> & labels,
                 ML::Thread_Context & thread_context,
                 int verbosity);
    

    /** Train (unsupervised) as a stack of denoising autoencoders. */
    void train_dnae(const std::vector<distribution<float> > & training_data,
                    const std::vector<distribution<float> > & testing_data,
                    const Configuration & config,
                    ML::Thread_Context & thread_context);

    /** Tests on both pristine and noisy inputs.  The first returned is the
        error on pristine inputs.  The second is the error on noisy inputs.
        the prob_cleared parameter describes the probability that noise will
        be added to any given input.
    */
    std::pair<double, double>
    test_dnae(const std::vector<distribution<float> > & data,
              float prob_cleared,
              ML::Thread_Context & thread_context,
              int verbosity) const;
    
    bool operator == (const DNAE_Stack & other) const;
};


IMPL_SERIALIZE_RECONSTITUTE(DNAE_Stack);


/*****************************************************************************/
/* DNAE_STACK_UPDATES                                                        */
/*****************************************************************************/

DNAE_Stack_Updates::
DNAE_Stack_Updates()
{
}

DNAE_Stack_Updates::
DNAE_Stack_Updates(const DNAE_Stack & stack)
{
    init(stack);
}

void
DNAE_Stack_Updates::
init(const DNAE_Stack & stack)
{
    resize(stack.size());
    for (unsigned i = 0;  i < size();  ++i)
        (*this)[i].init(false /* train_generative */, stack[i]);
}

DNAE_Stack_Updates &
DNAE_Stack_Updates::
operator += (const DNAE_Stack_Updates & updates)
{
    if (updates.size() != size())
        throw Exception("DNAE_Stack_Updates: out of sync");
    
    for (unsigned i = 0;  i < size();  ++i)
        (*this)[i] += updates[i];
    
    return *this;
}


/*****************************************************************************/
/* DNAE_STACK                                                                */
/*****************************************************************************/

distribution<float>
DNAE_Stack::
apply(const distribution<float> & input) const
{
    distribution<float> output = input;
    
    // Go down the stack
    for (unsigned l = 0;  l < size();  ++l)
        output = (*this)[l].apply(output);
    
    return output;
}

distribution<double>
DNAE_Stack::
apply(const distribution<double> & input) const
{
    distribution<double> output = input;
    
    // Go down the stack
    for (unsigned l = 0;  l < size();  ++l)
        output = (*this)[l].apply(output);
    
    return output;
}

distribution<float>
DNAE_Stack::
iapply(const distribution<float> & output) const
{
    distribution<float> input = output;
    
    // Go down the stack
    for (int l = size() - 1;  l >= 0;  --l)
        input = (*this)[l].iapply(input);
    
    return input;
}

distribution<double>
DNAE_Stack::
iapply(const distribution<double> & output) const
{
    distribution<double> input = output;
    
    // Go down the stack
    for (int l = size() - 1;  l >= 0;  --l)
        input = (*this)[l].iapply(input);
    
    return input;
}

void
DNAE_Stack::
serialize(ML::DB::Store_Writer & store) const
{
    store << (char)1; // version
    store << compact_size_t(size());
    for (unsigned i = 0;  i < size();  ++i)
        (*this)[i].serialize(store);
}

void
DNAE_Stack::
reconstitute(ML::DB::Store_Reader & store)
{
    char version;
    store >> version;
    if (version != 1) {
        cerr << "version = " << (int)version << endl;
        throw Exception("DNAE_Stack::reconstitute(): invalid version");
    }
    compact_size_t sz(store);
    resize(sz);

    for (unsigned i = 0;  i < sz;  ++i)
        (*this)[i].reconstitute(store);
}

pair<double, double>
DNAE_Stack::
train_discrim_example(const distribution<float> & data,
                      float label,
                      DNAE_Stack_Updates & updates) const
{
    /* fprop */

    vector<distribution<double> > outputs(size() + 1);
    outputs[0] = data;

    for (unsigned i = 0;  i < size();  ++i)
        outputs[i + 1] = (*this)[i].apply(outputs[i]);

    /* errors */
    distribution<double> errors = (label - outputs.back());
    double error = errors.dotprod(errors);
    distribution<double> derrors = -2.0 * errors;
    distribution<double> new_derrors;

    /* bprop */
    for (int i = size() - 1;  i >= 0;  --i) {
        (*this)[i].backprop_example(outputs[i + 1],
                                    derrors,
                                    outputs[i],
                                    new_derrors,
                                    updates[i]);
        derrors.swap(new_derrors);
    }

    return make_pair(sqrt(error), outputs.back()[0]);
}

struct Train_Discrim_Examples_Job {

    const DNAE_Stack & stack;
    const std::vector<distribution<float> > & data;
    const std::vector<float> & labels;
    Thread_Context & thread_context;
    const vector<int> & examples;
    int first;
    int last;
    DNAE_Stack_Updates & updates;
    vector<float> & outputs;
    double & total_rmse;
    Lock & updates_lock;
    boost::progress_display * progress;
    int verbosity;

    Train_Discrim_Examples_Job(const DNAE_Stack & stack,
                               const std::vector<distribution<float> > & data,
                               const std::vector<float> & labels,
                               Thread_Context & thread_context,
                               const vector<int> & examples,
                               int first, int last,
                               DNAE_Stack_Updates & updates,
                               vector<float> & outputs,
                               double & total_rmse,
                               Lock & updates_lock,
                               boost::progress_display * progress,
                               int verbosity)
        : stack(stack), data(data), labels(labels),
          thread_context(thread_context), examples(examples),
          first(first), last(last), updates(updates), outputs(outputs),
          total_rmse(total_rmse), updates_lock(updates_lock),
          progress(progress), verbosity(verbosity)
    {
    }

    void operator () ()
    {
        DNAE_Stack_Updates local_updates(stack);

        double total_rmse_local = 0.0;

        for (unsigned ix = first; ix < last;  ++ix) {
            int x = examples[ix];

            double rmse_contribution;
            double output;

            boost::tie(rmse_contribution, output)
                = stack.train_discrim_example(data[x],
                                              labels[x],
                                              local_updates);

            outputs[ix] = output;
            total_rmse_local += rmse_contribution;
        }

        Guard guard(updates_lock);
        total_rmse += total_rmse_local;
        updates += local_updates;
        if (progress) progress += (last - first);
    }
};

std::pair<double, double>
DNAE_Stack::
train_discrim_iter(const std::vector<distribution<float> > & data,
                   const std::vector<float> & labels,
                   Thread_Context & thread_context,
                   int minibatch_size, float learning_rate,
                   int verbosity,
                   float sample_proportion,
                   bool randomize_order)
{
    Worker_Task & worker = thread_context.worker();

    int nx = data.size();

    int microbatch_size = minibatch_size / (num_cpus() * 4);
            
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
    Model_Output outputs;
    outputs.resize(nx2);    

    std::auto_ptr<boost::progress_display> progress;
    if (verbosity >= 3) progress.reset(new boost::progress_display(nx2, cerr));

    for (unsigned x = 0;  x < nx2;  x += minibatch_size) {
                
        DNAE_Stack_Updates updates(*this);
                
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
                        
                Train_Discrim_Examples_Job
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
        
        update(updates, learning_rate);
    }

    // TODO: calculate AUC score
    distribution<float> test_labels;
    for (unsigned i = 0;  i < nx2;  ++i)
        test_labels.push_back(labels[examples[i]]);

    double auc = outputs.calc_auc(test_labels);

    return make_pair(sqrt(total_mse / nx2), auc);
}

std::pair<double, double>
DNAE_Stack::
train_discrim(const std::vector<distribution<float> > & training_data,
              const std::vector<float> & training_labels,
              const std::vector<distribution<float> > & testing_data,
              const std::vector<float> & testing_labels,
              const Configuration & config,
              ML::Thread_Context & thread_context)
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
            = train_discrim_iter(training_data, training_labels,
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
                = test_discrim(testing_data, testing_labels,
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

void
DNAE_Stack::
update(const DNAE_Stack_Updates & updates, double learning_rate)
{
    if (updates.size() != size())
        throw Exception("DNAE_Stack::update(): updates have the wrong size");

    for (unsigned i = 0;  i < size();  ++i)
        (*this)[i].update(updates[i], learning_rate);
}

struct Test_Discrim_Job {

    const DNAE_Stack & stack;
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

    Test_Discrim_Job(const DNAE_Stack & stack,
                     const vector<distribution<float> > & data,
                     const vector<float> & labels,
                     int first, int last,
                     const Thread_Context & context,
                     Lock & update_lock,
                     double & error_rmse,
                     vector<float> & outputs,
                     boost::progress_display * progress,
                     int verbosity)
        : stack(stack), data(data), labels(labels),
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
                = stack.apply(data[x]);

            outputs[x] = output[0];
            local_error_rmse += pow(labels[x] - output[0], 2);
        }

        Guard guard(update_lock);
        error_rmse += local_error_rmse;
        if (progress && verbosity >= 3) (*progress) += (last - first);
    }
};

pair<double, double>
DNAE_Stack::
test_discrim(const std::vector<distribution<float> > & data,
             const std::vector<float> & labels,
             ML::Thread_Context & thread_context,
             int verbosity)
{
    Lock update_lock;
    double mse_total = 0.0;

    int nx = data.size();

    std::auto_ptr<boost::progress_display> progress;
    if (verbosity >= 3) progress.reset(new boost::progress_display(nx, cerr));

    Worker_Task & worker = thread_context.worker();

    Model_Output outputs;
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
        int batch_size = nx / (num_cpus() * 20);
        
        for (unsigned x = 0; x < nx;  x += batch_size) {
            
            Test_Discrim_Job job(*this, data, labels,
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
                     outputs.calc_auc
                     (distribution<float>(labels.begin(), labels.end())));
}


#endif

} // namespace ML


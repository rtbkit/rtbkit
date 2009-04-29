/* classifier.cc
   Jeremy Barnes, 6 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Implementation of basic classifier methods.
*/

#include "classifier.h"
#include "classifier_persist_impl.h"
#include "arch/threads.h"
#include "utils/file_functions.h"
#include "ace/OS.h"
#include "evaluation.h"
#include "config_impl.h"
#include "worker_task.h"
#include "utils/guard.h"
#include <boost/bind.hpp>
#include <boost/thread/tss.hpp>


using namespace std;
using namespace DB;



namespace ML {


BYTE_PERSISTENT_ENUM_IMPL(Output_Encoding);


/*****************************************************************************/
/* CLASSIFIER_IMPL                                                           */
/*****************************************************************************/

Classifier_Impl::Classifier_Impl()
    : label_count_(0)
{
}

namespace {

size_t get_label_count(const Feature_Space & fs,
                       const Feature & predicted)
{
    Feature_Info info = fs.info(predicted);
    return info.value_count();
}

size_t check_label_count(const Feature_Space & fs,
                         const Feature & predicted,
                         size_t label_count)
{
    /* Don't try to check if we didn't put a feature in. */
    if (predicted == MISSING_FEATURE) return label_count;

    Feature_Info info = fs.info(predicted);

    /* For reals, we assume the number of labels are known. */
    if (info.type() != Feature_Info::REAL
        && info.value_count() != label_count) {
        

        throw Exception("Classifier_Impl: check_label_count(): feature (" 
                        + ostream_format(info.value_count())
                        + ") and label ("
                        + ostream_format(label_count)
                        + ") counts don't match; feature = "
                        + fs.print(predicted));
    }

    return label_count;
}

} // file scope

Classifier_Impl::
Classifier_Impl(const boost::shared_ptr<const Feature_Space> & feature_space,
                const Feature & predicted)
    : feature_space_(feature_space), predicted_(predicted),
      label_count_(get_label_count(*feature_space, predicted))
{
}

Classifier_Impl::
Classifier_Impl(const boost::shared_ptr<const Feature_Space> & feature_space,
                const Feature & predicted,
                size_t label_count)
    : feature_space_(feature_space), predicted_(predicted),
      label_count_(check_label_count(*feature_space, predicted, label_count))
{
}

void Classifier_Impl::
init(const boost::shared_ptr<const Feature_Space> & feature_space,
     const Feature & predicted)
{
    feature_space_ = feature_space;
    predicted_ = predicted;
    label_count_ = get_label_count(*feature_space, predicted);
}

void Classifier_Impl::
init(const boost::shared_ptr<const Feature_Space> & feature_space,
     const Feature & predicted,
     size_t label_count)
{
    feature_space_ = feature_space;
    predicted_ = predicted;
    label_count_ = check_label_count(*feature_space, predicted, label_count);
}

Classifier_Impl::~Classifier_Impl()
{
}

float Classifier_Impl::predict(int label, const Feature_Set & features) const
{
    if (label < 0 || label >= label_count())
        throw Exception(format("Attempt to predict non-existent label: "
                               "label = %d, label_count = %zd", label,
                               label_count()));
    return predict(features)[label];
}

float Classifier_Impl::predict_highest(const Feature_Set & features) const
{
    distribution<float> prediction = predict(features);
    return std::max_element(prediction.begin(), prediction.end())
        - prediction.begin();
}

namespace {

struct Accuracy_Job_Info {
    const Training_Data & data;
    const distribution<float> & example_weights;
    const Classifier_Impl & classifier;
    Lock lock;
    double & correct;
    double & total;

    Accuracy_Job_Info(const Training_Data & data,
                      const distribution<float> & example_weights,
                      const Classifier_Impl & classifier,
                      double & correct, double & total)
        : data(data), example_weights(example_weights),
          classifier(classifier),
          correct(correct), total(total)
    {
    }

    void calc(int x_start, int x_end)
    {
        //cerr << "calculating from " << x_start << " to " << x_end << endl;

        double sub_total = 0.0, sub_correct = 0.0;

        for (unsigned x = x_start;  x < x_end;  ++x) {
            double w = (example_weights.empty() ? 1.0 : example_weights[x]);
            if (w == 0.0) continue;

            //cerr << "x = " << x << " w = " << w << endl;
            
            distribution<float> result = classifier.predict(data[x]);
            Correctness c = correctness(result, classifier.predicted(), data[x]);
            sub_correct += w * c.possible * c.correct;
            sub_total += w * c.possible;
        }

        Guard guard(lock);
        correct += sub_correct;
        total += sub_total;
    }
};

struct Accuracy_Job {
    Accuracy_Job(Accuracy_Job_Info & info,
                 int x_start, int x_end)
        : info(info), x_start(x_start), x_end(x_end)
    {
    }

    Accuracy_Job_Info & info;
    int x_start, x_end;
    
    void operator () () const
    {
        info.calc(x_start, x_end);
    }
};

} // file scope

float Classifier_Impl::
accuracy(const Training_Data & data,
         const distribution<float> & example_weights) const
{
    double correct = 0.0;
    double total = 0.0;

    if (!example_weights.empty()
        && example_weights.size() != data.example_count())
        throw Exception("Classifier_Impl::accuracy(): dataset and weight "
                        "vector sizes don't match");

    unsigned nx = data.example_count();

    Accuracy_Job_Info info(data, example_weights, *this, correct, total);
    static Worker_Task & worker = Worker_Task::instance(num_threads() - 1);
    
    int group;
    {
        int parent = -1;  // no parent group
        group = worker.get_group(NO_JOB,
                                 format("accuracy group under %d", parent),
                                 parent);
        Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                     boost::ref(worker),
                                     group));
        
        /* Do 1024 examples per job. */
        for (unsigned x = 0;  x < data.example_count();  x += 1024)
            worker.add(Accuracy_Job(info, x, std::min(x + 1024, nx)),
                       format("accuracy example %d to %d under %d",
                              x, x + 1024, group),
                       group);
    }

    worker.run_until_finished(group);
    
    return correct / total;
}

namespace {

struct Predict_Job {

    int x_start, x_end;
    const Classifier_Impl & classifier;
    const Training_Data & data;

    Predict_Job(int x_start, int x_end,
                const Classifier_Impl & classifier,
                const Training_Data & data)
        : x_start(x_start), x_end(x_end), classifier(classifier),
          data(data)
    {
        
    }

    typedef void result_type;

    void operator () (Classifier_Impl::Predict_All_Output_Func output)
    {
        for (unsigned x = x_start;  x < x_end;  ++x) {
            Label_Dist prediction = classifier.predict(data[x]);
            output(x, &prediction[0]);
        }
    }

    void operator () (int label,
                      Classifier_Impl::Predict_One_Output_Func output)
    {
        for (unsigned x = x_start;  x < x_end;  ++x) {
            float prediction = classifier.predict(label, data[x]);
            output(x, prediction);
        }
    }
};

} // file scope

void
Classifier_Impl::
predict(const Training_Data & data,
        Predict_All_Output_Func output) const
{
    unsigned nx = data.example_count();

    static Worker_Task & worker = Worker_Task::instance(num_threads() - 1);
    
    int group;
    {
        int parent = -1;  // no parent group
        group = worker.get_group(NO_JOB,
                                 format("predict group under %d", parent),
                                 parent);
        Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                     boost::ref(worker),
                                     group));
        
        /* Do 1024 examples per job. */
        for (unsigned x = 0;  x < data.example_count();  x += 1024)
            worker.add(boost::bind(Predict_Job(x,
                                               std::min(x + 1024, nx),
                                               *this,
                                               data),
                                   output),
                       "predict job",
                       group);
    }

    worker.run_until_finished(group);
}

void
Classifier_Impl::
predict(const Training_Data & data,
        int label,
        Predict_One_Output_Func output) const
{
    unsigned nx = data.example_count();

    static Worker_Task & worker = Worker_Task::instance(num_threads() - 1);
    
    int group;
    {
        int parent = -1;  // no parent group
        group = worker.get_group(NO_JOB,
                                 format("predict group under %d", parent),
                                 parent);
        Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                     boost::ref(worker),
                                     group));
        
        /* Do 1024 examples per job. */
        for (unsigned x = 0;  x < data.example_count();  x += 1024)
            worker.add(boost::bind(Predict_Job(x,
                                               std::min(x + 1024, nx),
                                               *this,
                                               data),
                                   label,
                                   output),
                       "predict job",
                       group);
    }

    worker.run_until_finished(group);
}

boost::shared_ptr<Classifier_Impl>
Classifier_Impl::
poly_reconstitute(DB::Store_Reader & store,
                  const boost::shared_ptr<const Feature_Space> & fs)
{
    compact_size_t fs_flag(store);
    //cerr << "fs_flag = " << fs_flag << endl;
    if (fs_flag) {
        boost::shared_ptr<Feature_Space> fs2;
        store >> fs2;  // ignore this one
        return Registry<Classifier_Impl>::singleton().reconstitute(store, fs);
    }
    else 
        return Registry<Classifier_Impl>::singleton().reconstitute(store, fs);
}

boost::shared_ptr<Classifier_Impl>
Classifier_Impl::poly_reconstitute(DB::Store_Reader & store)
{
    //cerr << __PRETTY_FUNCTION__ << endl;
    compact_size_t fs_flag(store);

    //cerr << "poly_reconstitute: fs_flag = " << fs_flag << endl;

    boost::shared_ptr<const Feature_Space> fs;
    boost::shared_ptr<Feature_Space> fs_mutable;
    if (fs_flag) {
        store >> fs_mutable;
        fs = fs_mutable;
    }
    else fs = FS_Context::inner();

    //cerr << "reconstituting with feature space" << endl;

    boost::shared_ptr<Classifier_Impl> result
        = Registry<Classifier_Impl>::singleton().reconstitute(store, fs);

    //cerr << "result->predicted() = " << result->predicted() << endl;

    //cerr << "poly_reconstitute: feature_space = " << result->feature_space()
    //     << endl;

    if (fs_mutable) {
        //cerr << "freezing mutable" << endl;
        fs_mutable->freeze();
    }

    return result;
}


void Classifier_Impl::
poly_serialize(DB::Store_Writer & store, bool write_fs) const
{
    if (write_fs) {
        store << compact_size_t(1);
        store << feature_space();
    }
    else store << compact_size_t(0);

    Registry<Classifier_Impl>::singleton().serialize(store, this);
}

bool Classifier_Impl::merge_into(const Classifier_Impl & other, float)
{
    return false;
}

Classifier_Impl *
Classifier_Impl::merge(const Classifier_Impl & other, float) const
{
    return 0;
}

std::string
Classifier_Impl::
summary() const
{
    return class_id();
}


/*****************************************************************************/
/* POLYMORPHIC STUFF                                                         */
/*****************************************************************************/

namespace {

/* Put one of these objects per thread. */
boost::thread_specific_ptr<vector<boost::shared_ptr<const Feature_Space> > >
    fs_stack;

} // file scope

FS_Context::
FS_Context(const boost::shared_ptr<const Feature_Space> & feature_space)
{
    if (!fs_stack.get())
        fs_stack.reset(new vector<boost::shared_ptr<const Feature_Space> >());
    fs_stack->push_back(feature_space);
}

FS_Context::~FS_Context()
{
    assert(fs_stack.get());
    assert(!fs_stack->empty());
    if (fs_stack->empty())
        throw Exception("FS stack was empty in destructor; bad problem");
    fs_stack->pop_back();
}

const boost::shared_ptr<const Feature_Space> & FS_Context::inner()
{
    assert(fs_stack.get());
    if (fs_stack->empty()) throw Exception("feature space stack is empty");
    return fs_stack->back();
}

DB::Store_Writer &
operator << (DB::Store_Writer & store,
             const boost::shared_ptr<const Classifier_Impl> & classifier)
{
    classifier->poly_serialize(store);
    return store;
}

DB::Store_Reader &
operator >> (DB::Store_Reader & store,
             boost::shared_ptr<Classifier_Impl> & classifier)
{
    Classifier_Impl::poly_reconstitute(store, FS_Context::inner());
    return store;
}


/*****************************************************************************/
/* CLASSIFIER                                                                */
/*****************************************************************************/

Classifier::Classifier()
{
}

Classifier::Classifier(const boost::shared_ptr<Classifier_Impl> & impl)
    : impl(impl)
{
}

Classifier::Classifier(Classifier_Impl * impl, bool take_copy)
{
    if (take_copy) this->impl.reset(impl->make_copy());
    else this->impl.reset(impl);
}

Classifier::Classifier(const Classifier_Impl & impl)
    : impl(impl.make_copy())
{
}

Classifier::
Classifier(const std::string & name,
           const boost::shared_ptr<const Feature_Space> & feature_space)
{
    throw Exception("Classifier::Classifier(string, fs): not implemented");
    //impl = Registry<Classifier_Impl>::singleton().create(name, feature_space);
}

Classifier::
Classifier(const boost::shared_ptr<const Feature_Space> & feature_space,
           DB::Store_Reader & store)
{
    impl = Classifier_Impl::poly_reconstitute(store, feature_space);
}

Classifier::Classifier(const Classifier & other)
    : impl(other.impl ? other.impl->make_copy() : 0)
{
}

Classifier::Classifier &
Classifier::operator = (const Classifier & other)
{
    if (other.impl) impl.reset(other.impl->make_copy());
    else impl.reset();
    return *this;
}

DB::Store_Writer &
operator << (DB::Store_Writer & store, const Classifier & classifier)
{
    classifier.serialize(store);
    return store;
}

DB::Store_Reader &
operator >> (DB::Store_Reader & store, Classifier & classifier)
{
    classifier.reconstitute(store, FS_Context::inner());
    return store;
}

void Classifier::load(const std::string & filename)
{
    Store_Reader store(filename);
    reconstitute(store);
}

void Classifier::
load(const std::string & filename, boost::shared_ptr<const Feature_Space> fs)
{
    Store_Reader store(filename);
    reconstitute(store, fs);
}

void Classifier::save(const std::string & filename, bool write_fs) const
{
    Store_Writer store(filename);
    serialize(store, write_fs);
}


/*****************************************************************************/
/* FACTORY_BASE<CLASSIFIER_IMPL>                                             */
/*****************************************************************************/

Factory_Base<Classifier_Impl>::~Factory_Base()
{
}

} // namespace ML



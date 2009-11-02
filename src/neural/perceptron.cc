/* perceptron.cc
   Jeremy Barnes, 16 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Wrapper around the MLP class to go with the rest of the classifier
   framework.  Implementation.
*/

#include "perceptron.h"
#include "boosting/classifier_persist_impl.h"
#include "utils/profile.h"
#include "utils/environment.h"
#include "algebra/matrix_ops.h"
#include "boosting/training_index.h"
#include "algebra/irls.h"
#include "utils/vector_utils.h"
#include <iomanip>
#include <boost/random/lagged_fibonacci.hpp>
#include <boost/random/uniform_01.hpp>
#include "utils/parse_context.h"
#include "stats/distribution_simd.h"
#include "stats/distribution_ops.h"
#include "boosting/evaluation.h"
#include "arch/simd_vector.h"
#include "algebra/lapack.h"
#include "boosting/config_impl.h"
#include "boosting/worker_task.h"
#include "utils/guard.h"
#include <boost/bind.hpp>
#include "arch/backtrace.h"

using namespace std;
using namespace DB;

namespace ML {

namespace {

Env_Option<bool> profile("PROFILE_PERCEPTRON", false);

double t_predict = 0.0, t_accuracy = 0.0, t_decorrelate = 0.0;

struct Stats {
    ~Stats()
    {
        if (profile) {
            cerr << "perceptron runtime profile: " << endl;
            cerr << "  decorrelate:    " << t_decorrelate << "s" << endl;
            cerr << "  predict:        " << t_predict     << "s" << endl;
            cerr << "  accuracy:       " << t_accuracy    << "s" << endl;
        }
    }
} stats;

} // file scope


/*****************************************************************************/
/* PERCEPTRON                                                                */
/*****************************************************************************/

Perceptron::Perceptron()
    : max_units(0)
{
}

Perceptron::
Perceptron(const boost::shared_ptr<const Feature_Space> & feature_space,
               const Feature & predicted)
    : Classifier_Impl(feature_space, predicted), max_units(0)
{
}

Perceptron::
Perceptron(DB::Store_Reader & reader,
           const boost::shared_ptr<const Feature_Space> & feature_space)
{
    this->reconstitute(reader, feature_space);
}

Perceptron::
Perceptron(const boost::shared_ptr<const Feature_Space> & feature_space,
           const Feature & predicted,
           size_t label_count)
    : Classifier_Impl(feature_space, predicted, label_count),
      max_units(0)
{
}

float
Perceptron::
predict(int label, const Feature_Set & features) const
{
    return predict(features).at(label);
}

distribution<float>
Perceptron::
predict(const Feature_Set & fs) const
{
    PROFILE_FUNCTION(t_predict);

    float scratch1[max_units], scratch2[max_units];
    float * input = scratch1;
    float * output = scratch2;
    extract_features(fs, input);
    layers[0]->apply(input, output);
    
    for (unsigned l = 1;  l < layers.size();  ++l) {
        layers[l]->apply(output, input);
        std::swap(input, output);
    }
    
    return distribution<float>(output, output + layers.back()->outputs());
}

namespace {

struct Accuracy_Job_Info {
    const boost::multi_array<float, 2> & decorrelated;
    const std::vector<Label> & labels;
    const distribution<float> & example_weights;
    const Perceptron & perceptron;

    Lock lock;
    double & correct;
    double & rmse;
    double & total;

    Accuracy_Job_Info(const boost::multi_array<float, 2> & decorrelated,
                      const std::vector<Label> & labels,
                      const distribution<float> & example_weights,
                      const Perceptron & perceptron,
                      double & correct, double & rmse, double & total)
        : decorrelated(decorrelated), labels(labels),
          example_weights(example_weights), perceptron(perceptron),
          correct(correct), rmse(rmse), total(total)
    {
    }

    void calc(int x_start, int x_end)
    {
        double sub_total = 0.0, sub_rmse = 0.0, sub_correct = 0.0;

        float scratch1[perceptron.max_units], scratch2[perceptron.max_units];
        float * input = scratch1, * output = scratch2;

        size_t nl = perceptron.label_count();

        bool regression_problem
            = perceptron.feature_space()->info(perceptron.predicted()).type()
            == REAL;

        for (unsigned x = x_start;  x < x_end;  ++x) {
            float w = (example_weights.empty() ? 1.0 : example_weights[x]);
            if (w == 0.0) continue;

            // Skip the first layer since we've already decorrelated
            // Second layer calculated directly over input
            perceptron.layers[1]->apply(&decorrelated[x][0], input);

            for (unsigned l = 2;  l < perceptron.layers.size();  ++l) {
                perceptron.layers[l]->apply(input, output);
                std::swap(input, output);
            }

            if (regression_problem) {
                double error = input[0] - labels[x].value();
                sub_correct += w * fabs(error);
                sub_rmse    += w * error * error;
                sub_total   += w;
            }
            else {
                Correctness c = correctness(input, input + nl, labels[x]);
                sub_correct += w * c.possible * c.correct;
                sub_rmse    += w * c.possible * c.margin * c.margin;
                sub_total   += w * c.possible;
            }
        }

        Guard guard(lock);
        correct += sub_correct;
        rmse += sub_rmse;
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

std::pair<float, float>
Perceptron::
accuracy(const boost::multi_array<float, 2> & decorrelated,
         const std::vector<Label> & labels,
         const distribution<float> & example_weights) const
{
    PROFILE_FUNCTION(t_accuracy);

    double correct = 0.0;
    double rmse = 0.0;
    double total = 0.0;

    unsigned nx = decorrelated.shape()[0];

    Accuracy_Job_Info info(decorrelated, labels, example_weights, *this,
                           correct, rmse, total);
    static Worker_Task & worker = Worker_Task::instance(num_threads() - 1);
    
    int group;
    {
        int parent = -1;  // no parent group
        group = worker.get_group(NO_JOB,
                                 format("Perceptron::accuracy() under %d", parent),
                                 parent);
        Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                     boost::ref(worker),
                                     group));
        
        int nx_per_job = 512;

        /* Do 2048 examples per job. */
        for (unsigned x = 0;  x < nx;  x += nx_per_job) {
            int end = std::min(x + nx_per_job, nx);
            worker.add(Accuracy_Job(info, x, end),
                       format("Perceptron::accuracy() %d-%d under %d",
                              x, end, group),
                       group);
        }
    }
    
    worker.run_until_finished(group);
    
    return make_pair(correct / total, sqrt(rmse / total));
}

std::string Perceptron::print() const
{
    string result = format("{ Perceptron: %zd layers, %zd inputs, %zd outputs\n",
                           layers.size(), features.size(),
                           (layers.size() ? layers.back()->outputs() : 0));

    result += "  features:\n";
    for (unsigned f = 0;  f < features.size();  ++f)
        result += format("    %d %s\n", f,
                         feature_space()->print(features[f]).c_str());
    result += "\n";

    for (unsigned i = 0;  i < layers.size();  ++i)
        result += format("  layer %d\n", i) + layers[i]->print();

    result += "}";

    return result;
}

std::vector<ML::Feature> Perceptron::all_features() const
{
    return features;
}

Output_Encoding
Perceptron::
output_encoding() const
{
    return OE_PM_ONE;
}

std::vector<int>
Perceptron::
parse_architecture(const std::string & arch)
{
    Parse_Context context(arch, &arch[0], &arch[0] + arch.size());

    vector<int> result;
    while (context) {
        if (context.match_literal('%')) {
            if (context.match_literal('i'))
                result.push_back(-1);
            else context.exception("expected i after %");
        }
        else result.push_back(context.expect_unsigned());
        
        if (!context.match_literal('_')) context.expect_eof();
    }

    return result;
}

void Perceptron::add_layer(const boost::shared_ptr<Layer> & layer)
{
    layer->validate();

    layers.push_back(layer);
    if (layers.size() == 1 || layer->inputs() > max_units)
        max_units = layer->inputs();
    if (layer->outputs() > max_units)
        max_units = layer->outputs();
}

void Perceptron::clear()
{
    max_units = 0;
    layers.clear();
    features.clear();
}

size_t Perceptron::parameters() const
{
    size_t result = 0;
    for (unsigned l = 1;  l < layers.size();  ++l)
        result += layers[l]->parameter_count();
    return result;
}

namespace {

static const std::string PERCEPTRON_MAGIC = "PERCEPTRON";
static const compact_size_t PERCEPTRON_VERSION = 0;


} // file scope

void Perceptron::serialize(DB::Store_Writer & store) const
{
    store << PERCEPTRON_MAGIC << PERCEPTRON_VERSION
          << compact_size_t(label_count());
    feature_space()->serialize(store, predicted_);

    store << compact_size_t(features.size());
    for (unsigned i = 0;  i < features.size();  ++i)
        feature_space()->serialize(store, features[i]);

    /* Now the layers... */
    store << compact_size_t(layers.size());
    for (unsigned i = 0;  i < layers.size();  ++i)
        layers[i]->serialize(store);

    store << string("END PERCEPTRON");
}

void Perceptron::
reconstitute(DB::Store_Reader & store,
             const boost::shared_ptr<const Feature_Space> & feature_space)
{
    /* Implement the strong exception guarantee, except for the store. */
    string magic;
    compact_size_t version;
    store >> magic >> version;
    if (magic != PERCEPTRON_MAGIC)
        throw Exception("Attempt to reconstitute \"" + magic
                                + "\" with perceptrons reconstitutor");
    if (version > PERCEPTRON_VERSION)
        throw Exception(format("Attemp to reconstitute perceptrons "
                               "version %zd, only <= %zd supported",
                               version.size_,
                               PERCEPTRON_VERSION.size_));
    
    compact_size_t label_count(store);

    predicted_ = MISSING_FEATURE;
    feature_space->reconstitute(store, predicted_);

    Perceptron new_me(feature_space, predicted_, label_count);

    compact_size_t nfeat(store);
    new_me.features.resize(nfeat);
    for (unsigned i = 0;  i < nfeat;  ++i)
        feature_space->reconstitute(store, new_me.features[i]);

    /* Now the layers... */
    compact_size_t nlayers(store);
    new_me.layers.resize(nlayers);

    new_me.max_units = 0;
    for (unsigned i = 0;  i < nlayers;  ++i) {
        new_me.layers[i].reset(new Dense_Layer<float>());
        new_me.layers[i]->reconstitute(store);
        new_me.max_units = max<int>(new_me.max_units,
                                    new_me.layers[i]->inputs());
        new_me.max_units = max<int>(new_me.max_units,
                                    new_me.layers[i]->outputs());
    }

    string s;
    store >> s;
    if (s != "END PERCEPTRON")
        throw Exception("problem with perceptron reconstitution");
    
    swap(new_me);
}

Perceptron * Perceptron::make_copy() const
{
    return new Perceptron(*this);
}

boost::multi_array<float, 2>
Perceptron::
decorrelate(const Training_Data & data) const
{
    PROFILE_FUNCTION(t_decorrelate);

    if (layers.empty())
        throw Exception("Perceptron::decorrelate(): need to train decorrelation "
                        "first");

    size_t nx = data.example_count();
    size_t nf = features.size();

    boost::multi_array<float, 2> result(boost::extents[nx][nf]);
    
    float input[nf];

    for (unsigned x = 0;  x < nx;  ++x) {
        extract_features(data[x], &input[0]);
        layers[0]->apply(input, &result[x][0]);
    }
    
    return result;
}


/*****************************************************************************/
/* REGISTRATION                                                              */
/*****************************************************************************/

namespace {

Register_Factory<Classifier_Impl, Perceptron>
    PERCEPTRON_REGISTER("PERCEPTRON");

} // file scope

} // namespace ML

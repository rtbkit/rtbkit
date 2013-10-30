/* boosted_stumps.cc
   Jeremy Barnes, 6 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Implementation of boosted decision stumps.
*/

#include "boosted_stumps.h"
#include "classifier_persist_impl.h"
#include "jml/math/xdiv.h"
#include "jml/arch/simd_vector.h"
#include "boosted_stumps_impl.h"
#include "stump_predict.h"
#include "jml/utils/environment.h"
#include <boost/timer.hpp>
#include <boost/progress.hpp>
#include "jml/utils/floating_point.h"
#include "jml/utils/vector_utils.h"
#include "boosting_core.h"
#include "boosting_training.h"
#include "config_impl.h"
#include "binary_symmetric.h"
#include "jml/utils/worker_task.h"
#include "jml/utils/guard.h"
#include <iostream>
#include <boost/bind.hpp>


using namespace std;
using namespace DB;

namespace ML {

namespace {

Env_Option<bool> profile("PROFILE_BOOSTED_STUMPS", false);

double t_train = 0.0, t_update = 0.0, t_predict = 0.0, t_accuracy = 0.0;

struct Stats {
    ~Stats()
    {
        if (profile) {
            cerr << "Boosted stumps profile: " << endl;
            cerr << "  train:          " << t_train       << "s" << endl;
            cerr << "  update:         " << t_update      << "s" << endl;
            cerr << "  predict:        " << t_predict     << "s" << endl;
            cerr << "  accuracy:       " << t_accuracy    << "s" << endl;
        }
    }
} stats;

class Function_Profiler {
public:
    boost::timer * t;
    double & var;
    Function_Profiler(double & var)
        : t(0), var(var)
    {
        if (profile) t = new boost::timer();
    }
    ~Function_Profiler()
    {
        if (t) var += t->elapsed();
        delete t;
    }
};

#define PROFILE_FUNCTION(var) \
Function_Profiler __profiler(var);

} // file scope


/*****************************************************************************/
/* BOOSTED_STUMPS                                                            */
/*****************************************************************************/

Boosted_Stumps::Boosted_Stumps()
{
}

Boosted_Stumps::
Boosted_Stumps(const std::shared_ptr<const Feature_Space> & feature_space,
               const Feature & predicted)
    : Classifier_Impl(feature_space, predicted)
{
    output = RAW;
}

Boosted_Stumps::
Boosted_Stumps(DB::Store_Reader & reader,
               const std::shared_ptr<const Feature_Space> & feature_space)
{
    this->reconstitute(reader, feature_space);
}

Boosted_Stumps::
Boosted_Stumps(const std::shared_ptr<const Feature_Space> & feature_space,
               const Feature & predicted,
               size_t label_count)
    : Classifier_Impl(feature_space, predicted, label_count)
{
}

namespace {

/** Class for the results which updates a whole distribution at once. */
struct Results_Dist {
    Results_Dist(distribution<float> & results,
                 const Feature_Space & fs)
        : results(results), fs(fs) {}

    void operator () (const distribution<float> & dist, float weight,
                      const Feature & feature) const
    {
        //cerr << "feature = " << fs.print(feature) << endl;
        //cerr << "  adding " << weight << " * " << dist;
        results += weight * dist;
        //cerr << "  result now " << results << endl;
    }

    distribution<float> & results;
    const Feature_Space & fs;
};

/** Class for the results which updates a single distribution at once. */
struct Results_Single {
    Results_Single(float & result, int label)
        : result(result), label(label) {}
    
    JML_ALWAYS_INLINE
    void operator () (const distribution<float> & dist, float weight,
                      const Feature & feature) const
    {
        result += weight * dist[label];
    }
    
    float & result;
    int label;
    
};

} // file scope

distribution<float>
Boosted_Stumps::
predict(const Feature_Set & features,
        PredictionContext * context) const
{
    PROFILE_FUNCTION(t_predict);
    distribution<float> result(label_count());
    if (bias.size()) result += bias;
    predict_core(features, Results_Dist(result, *feature_space()));
    
    for (unsigned i = 0;  i < result.size();  ++i) {
        if (!finite(result[i])) {
            cerr << "result = " << result << endl;
            throw Exception("Boosted_Stumps::predict(): non-finite result");
        }
    }

    //cerr << "result = " << result << endl;

    //result =- result.max();

    //result = exp(result);
    //result.normalize();
    //result -= 0.5;

    double total = 0.0;

    if (output == LOGIT || output == LOGIT_NORM) {
        for (unsigned i = 0;  i < result.size();  ++i) {
            /* Avoid an overflow from the exp. */
            if (result[i] > fp_traits<float>::max_exp_arg * 0.9)
                result[i] = fp_traits<float>::max_exp_arg * 0.9;
            double e = exp(result[i]);
            double x = e / (e + (1.0 / e));
            total += x;
            result[i] = x;
        }
        if (output == LOGIT_NORM) {
            if ((float)total == 0.0F) {
                cerr << "warning: boosted stumps says no results are correct"
                     << endl;
                result.fill(1);  // assign all elements
                result.normalize();
            }
            else result /= total;
        }
    }

    for (unsigned i = 0;  i < result.size();  ++i) {
        if (!finite(result[i])) {
            distribution<float> result2(label_count());
            if (bias.size()) result2 += bias;
            predict_core(features, Results_Dist(result2, *feature_space()));
            
            cerr << "result = " << result << endl;
            cerr << "output of predict_core = " << result2 << endl;
            cerr << "total = " << total << endl;

            for (unsigned i = 0;  i < result2.size();  ++i) {
                /* Avoid an overflow from the exp. */
                if (result2[i] > fp_traits<float>::max_exp_arg * 0.9)
                    result2[i] = fp_traits<float>::max_exp_arg * 0.9;
                double e = exp(result2[i]);
                double x = e / (e + (1.0 / e));
                total += x;
                result[i] = x;
                cerr << "  i = " << i << " e = " << e << " x = " << x
                     << endl;
            }

            cerr << "after logit: result = " << result2 << endl;

            throw Exception("non-finite result");
        }
    }
    
    return result;
}

float
Boosted_Stumps::
predict(int label, const Feature_Set & features,
        PredictionContext * context) const
{
    PROFILE_FUNCTION(t_predict);
    if (label < 0 || label >= label_count())
        throw Exception(format("Boosted_Stumps::predict(int, Feature_Set): "
                               "Attempt to predict label %d with label_count "
                               " %zd", label, label_count()));

    if (output == LOGIT_NORM) {
        /* Need to predict all, so we know how to normalize.  Could be much
           slower. */
        return predict(features)[label];
    }

    
    float result = 0.0;
    if (bias.size()) result += bias[label];
    predict_core(features, Results_Single(result, label));
    if (output == LOGIT) {
        double e = exp(result);
        result = e / (e + 1.0 / e);
    }
    return result;
}

Boosted_Stumps::iterator Boosted_Stumps::
insert(const Stump & stump, float weight)
{
    if (stump.split.feature() == MISSING_FEATURE) return end();

    iterator it = find(stump.split);
    if (it == end())
        return iterator
            (stumps.insert(make_pair(stump.split,
                                     stump.scaled(weight))).first);
    else it->merge(stump, weight);
    return it;
}

void Boosted_Stumps::insert(const std::vector<Stump> & stumps)
{
    distribution<float> scales(stumps.size(), 1.0 / stumps.size());
    insert(stumps, scales);
}

void Boosted_Stumps::
insert(const std::vector<Stump> & stumps, const distribution<float> & scale)
{
    if (scale.size() != stumps.size())
        throw Exception(format("Boosted_Stumps::insert(): stumps.size() = %zd"
                               " != scale.size() = %zd", stumps.size(),
                               scale.size()));
    for (unsigned i = 0;  i < stumps.size();  ++i)
        insert(stumps[i], scale[i]);
}

#if 0

namespace {

struct Accuracy_Job_Info {
    const Training_Data & data;
    const distribution<float> & example_weights;
    const Boosted_Stumps & stumps;
    const Optimization_Info & opt_info;
    boost::multi_array<float, 2> & output;
    bool bin_sym;

    Lock lock;
    double & correct;
    double & margin;
    
    Accuracy_Job_Info(const Training_Data & data,
                      const distribution<float> & example_weights,
                      const Boosted_Stumps & stumps,
                      const Optimization_Info & opt_info,
                      boost::multi_array<float, 2> & output,
                      bool bin_sym,
                      double & correct, double & margin)
        : data(data), example_weights(example_weights),
          stumps(stumps), opt_info(opt_info),
          output(output), bin_sym(bin_sym), correct(correct), margin(margin)
    {
    }

    void calc(unsigned start_x, unsigned end_x)
    {
        double sub_correct = 0.0, sub_margin = 0.0;

        int nl = output.shape()[1];

        const std::vector<Label> & labels
            = data.index().labels(stumps.predicted());
        
        if (bin_sym) {
            for (Boosted_Stumps::const_iterator it = stumps.begin();
                 it != stumps.end();  ++it) {
                
                typedef Binsym_Updater<Boosting_Predict> Updater;
                typedef Update_Weights<Updater> Update;
                Updater updater;
                Update update(updater);
                
                update(*it, opt_info, 1.0, output, data, start_x, end_x);
            }

            /* Now for the scoring. */
            typedef Binsym_Scorer Scorer;
            Scorer scorer;
            if (example_weights.empty()) {
                for (unsigned x = start_x;  x < end_x;  ++x)
                    sub_correct += scorer(labels[x], &output[x][0],
                                          &output[x][0]);
            }
            else {
                for (unsigned x = start_x;  x < end_x;  ++x)
                    sub_correct += scorer(labels[x], &output[x][0],
                                          &output[x][0] + 1)
                        * example_weights[x];
            }
        }
        else {
            for (Boosted_Stumps::const_iterator it = stumps.begin();
                 it != stumps.end();  ++it) {
                
                typedef Normal_Updater<Boosting_Predict> Updater;
                typedef Update_Weights<Updater> Update;
                Updater updater(nl);
                Update update(updater);
                
                update(*it, opt_info, 1.0, output, data, start_x, end_x);
            }

            /* Now for the scoring. */
            typedef Normal_Scorer Scorer;
            Scorer scorer;
            if (example_weights.empty()) {
                for (unsigned x = start_x;  x < end_x;  ++x)
                    sub_correct += scorer(labels[x], &output[x][0],
                                          &output[x][0]);
            }
            else {
                for (unsigned x = start_x;  x < end_x;  ++x)
                    sub_correct += scorer(labels[x], &output[x][0],
                                          &output[x][0] + 1)
                        * example_weights[x];
            }
        }

        Guard guard(lock);
        correct += sub_correct;
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
Boosted_Stumps::
accuracy(const Training_Data & data,
         const distribution<float> & example_weights,
         const Optimization_Info * opt_info_ptr) const
{
    PROFILE_FUNCTION(t_accuracy);
    unsigned nx = data.example_count();
    unsigned nl = label_count();
    
    boost::multi_array<float, 2> scores(boost::extents[nx][nl]);

    bool bin_sym = convert_bin_sym(scores, data, predicted_, all_features());
    
    double correct = 0.0, margin = 0.0;

    Optimization_Info new_opt_info;
    const Optimization_Info & opt_info
        = (opt_info_ptr ? *opt_info_ptr : new_opt_info);

    Accuracy_Job_Info info(data, example_weights, *this, opt_info,
                           scores, bin_sym,
                           correct, margin);

    static Worker_Task & worker = Worker_Task::instance(num_threads() - 1);
    
    int group;
    {
        int parent = -1;  // no parent group
        group = worker.get_group(NO_JOB,
                                 format("Boosted_Stumps::accuracy under %d", parent),
                                 parent);
        Call_Guard guard(boost::bind(&Worker_Task::unlock_group,
                                     boost::ref(worker),
                                     group));
        
        unsigned job_ex = 2048 / scores.shape()[1];
        
        for (unsigned x = 0;  x < data.example_count();  x += job_ex) {
            unsigned end = std::min(x + job_ex, nx);
            worker.add(Accuracy_Job(info, x, end),
                       format("Boosted_Stumps::accuracy() job %d-%d under %d",
                              x, end, group),
                       group);
        }
    }

    worker.run_until_finished(group);
    
    double total
        = (example_weights.empty() ? (float)nx : example_weights.total());
    
    return make_pair(correct / total, margin / total);
}

#else
std::pair<float, float>
Boosted_Stumps::
accuracy(const Training_Data & data,
         const distribution<float> & example_weights,
         const Optimization_Info * opt_info_ptr) const
{
    return Classifier_Impl::accuracy(data, example_weights, opt_info_ptr);
}
#endif

std::string Boosted_Stumps::print() const
{
    string result;
    for (const_iterator it = begin();  it != end();  ++it) {
        result += it->print();
        result += '\n';
    }
    return result;
}

std::vector<ML::Feature> Boosted_Stumps::all_features() const
{
    std::vector<ML::Feature> result;
    for (stumps_type::const_iterator it = stumps.begin();
         it != stumps.end();  ++it)
        result.push_back(it->first.feature());
    make_vector_set(result);
    return result;
}

Output_Encoding
Boosted_Stumps::
output_encoding() const
{
    switch (output) {
    case RAW: return OE_PM_INF;
    case LOGIT:
    case LOGIT_NORM:
        return OE_PROB;
    default:
        throw Exception("Boosted_Stumps::output_encoding(): uknown encoding");
    }
}

void Boosted_Stumps::calc_sum_missing()
{
    distribution<double> totals(label_count());
    for (const_iterator it = begin();  it != end();  ++it) {
        distribution<double> this_val(it->action.pred_missing.begin(),
                                      it->action.pred_missing.end());
        totals += this_val;
    }

    sum_missing = distribution<float>(totals.begin(), totals.end());
}

namespace {

static const std::string BOOSTED_STUMPS_MAGIC = "BOOSTED_STUMPS";
static const compact_size_t BOOSTED_STUMPS_VERSION = 4;

void serialize_dist(const distribution<float> & dist,
                    DB::Store_Writer & store)
{
    bool non_zero = false;
    for (unsigned i = 0;  i < dist.size();  ++i) {
        //if (dist[i] != 0.0) { non_zero = true;  break; }
        if (abs(dist[i]) > 1e-10) { non_zero = true;  break; }
    }

    if (non_zero) {
        bool pos_neg = (dist.size() == 2 && (abs(dist[0] + dist[1]) < 1e-10));
        if (pos_neg)
            store << compact_const(1) << dist[0];
        else store << dist;
    }
    else store << compact_const(0);
}

void reconstitute_dist(distribution<float> & dist,
                       DB::Store_Reader & store,
                       size_t nl)
{
    dist.clear();
    compact_size_t size(store);

    if (size == 0) dist.resize(nl);
    else if (size == 1) {
        dist.resize(2);
        store >> dist[0];
        dist[1] = -dist[0];
    }
    else {
        if (size != nl)
            throw Exception
                (format("reconstituting Boosted_Stumps dist of wrong size: "
                        "is %zd, should be %zd", size.size_, nl));

        dist.resize(size);
        for (unsigned i = 0;  i < nl;  ++i)
            store >> dist[i];
    }
}

} // file scope

COMPACT_PERSISTENT_ENUM_DECL(Boosted_Stumps::Output);
COMPACT_PERSISTENT_ENUM_IMPL(Boosted_Stumps::Output);

void Boosted_Stumps::serialize(DB::Store_Writer & store) const
{
    store << BOOSTED_STUMPS_MAGIC << BOOSTED_STUMPS_VERSION
          << compact_const(label_count());
    feature_space()->serialize(store, predicted_);
    store << output << bias;
    store << compact_const(stumps.size());

    for (stumps_type::const_iterator it = stumps.begin();
         it != stumps.end();  /* no inc */) {

        const Feature & feat = it->first.feature();
        feature_space()->serialize(store, feat);
        
        /* Work out how many entries we have with this feature. */
        int nf = 0;
        stumps_type::const_iterator first = it;
        while (it != stumps.end() && it->first.feature() == feat) {
            ++nf;
            ++it;
        }
        
        store << compact_const(nf);
        std::swap(it, first);

        for (unsigned i = 0;  i < nf;  ++i,++it) {
            const Stump & stump = it->second;
            stump.serialize_lw(store);
        }

        if (it != first)
            throw Exception("Boosted_Stumps::serialize(): logic error");
    }
}

void Boosted_Stumps::
reconstitute(DB::Store_Reader & store,
             const std::shared_ptr<const Feature_Space> & features)
{
    /* Implement the strong exception guarantee, except for the store. */
    string magic;
    compact_size_t version;
    store >> magic >> version;
    if (magic != BOOSTED_STUMPS_MAGIC)
        throw Exception("Attempt to reconstitute \"" + magic
                                + "\" with boosted stumps reconstitutor");
    if (version > BOOSTED_STUMPS_VERSION)
        throw Exception(format("Attemp to reconstitute boosted stumps "
                               "version %zd, only <= %zd supported",
                               version.size_,
                               BOOSTED_STUMPS_VERSION.size_));
    
    compact_size_t label_count(store);

    predicted_ = MISSING_FEATURE;
    if (version >= 4) { // added in version 4
        features->reconstitute(store, predicted_);
    }

    Boosted_Stumps new_me(features, predicted_, label_count);
    
    if (version >= 1) {  // added in version 1
        store >> new_me.output;
    }
    else new_me.output = RAW;

    if (version >= 2) // added in version 2
        store >> new_me.bias;
    else new_me.bias.clear();

    //ofstream dump("stump-load-dump.txt");
    
    if (version >= 3) {
        /* Load compact version... */
        compact_size_t stump_count(store);
        int ns = 0;

        while (ns < stump_count) {
            Feature feature;
            features->reconstitute(store, feature);
            compact_size_t nsf(store);  // number of stumps for this feature

            for (unsigned i = 0;  i < nsf;  ++i) {
                Stump stump(features, predicted_, label_count);
                stump.reconstitute_lw(store, feature);
                new_me.insert(stump);
                ++ns;

#if 0
                dump << "stump " << ns << ": " << features->print(stump.feature)
                     << " " << stump.arg << " " << stump.pred_true << " "
                     << stump.pred_false << " " << stump.pred_missing << endl;
#endif
            }
        }
        
        if (ns != stump_count)
            throw Exception("Boosted_Stumps::reconstitute(): logic error");

        calc_sum_missing();
    }
    else {
        compact_size_t stump_count(store);

        /* Load non-compact versions... */
        for (unsigned i = 0;  i < stump_count;  ++i) {
            Stump stump(store, features);
#if 0
            dump << "stump " << i << ": " << features->print(stump.feature)
                 << " " << stump.arg << " " << stump.pred_true << " "
                 << stump.pred_false << " " << stump.pred_missing << endl;
#endif

            new_me.insert(stump);
        }

        calc_sum_missing();
    }

    swap(new_me);

    //cerr << "predicted_ = " << predicted_ << endl;
}

Boosted_Stumps * Boosted_Stumps::make_copy() const
{
    return new Boosted_Stumps(*this);
}

void Boosted_Stumps::combine(const Boosted_Stumps & with, float scale)
{
    /* Check they are compatible. */
    if (with.label_count() != label_count())
        throw Exception("Attempt to combine two boosted stumps with different "
                        "label counts.");
    if (with.feature_space().get() != feature_space().get())
        throw Exception("Attempt to combine two boosted stumps with different "
                        "feature spaces.");

    for (const_iterator it = with.begin();  it != with.end();  ++it)
        insert(*it, scale);
}

bool
Boosted_Stumps::
merge_into(const Classifier_Impl & other, float weight)
{
    /* If the other is a stump, we merge it. */
    const Stump * as_stump = dynamic_cast<const Stump *>(&other);
    if (as_stump) {
        insert(*as_stump, weight);
        return true;
    }
    const Boosted_Stumps * as_stumps
        = dynamic_cast<const Boosted_Stumps *>(&other);
    if (as_stumps) {
        combine(*as_stumps, weight);
        return true;
    }
    
    return false;
}

Classifier_Impl *
Boosted_Stumps::
merge(const Classifier_Impl & other, float weight) const
{
    /* If the other is a boosted stumps or a stump, we can merge it. */

    const Stump * as_stump = dynamic_cast<const Stump *>(&other);
    if (as_stump) {
        auto_ptr<Boosted_Stumps> me_copy(make_copy());
        me_copy->insert(*as_stump, weight);
        return me_copy.release();
    }

    const Boosted_Stumps * as_stumps
        = dynamic_cast<const Boosted_Stumps *>(&other);
    if (as_stumps) {
        auto_ptr<Boosted_Stumps> me_copy(make_copy());
        me_copy->combine(*as_stumps, weight);
        return me_copy.release();
    }
    
    return 0;
}


/*****************************************************************************/
/* REGISTRATION                                                              */
/*****************************************************************************/

namespace {

Register_Factory<Classifier_Impl, Boosted_Stumps>
    STUMP_REGISTER("BOOSTED_STUMPS");

} // file scope
} // namespace ML

ENUM_INFO_NAMESPACE
  
const Enum_Opt<ML::Boosted_Stumps::Output>
Enum_Info<ML::Boosted_Stumps::Output>::OPT[3] = {
    { "raw",         ML::Boosted_Stumps::RAW        },
    { "logit",       ML::Boosted_Stumps::LOGIT      },
    { "logit_norm",  ML::Boosted_Stumps::LOGIT_NORM } };

const char * Enum_Info<ML::Boosted_Stumps::Output>::NAME
    = "Boosted_Stumps::Output";

END_ENUM_INFO_NAMESPACE


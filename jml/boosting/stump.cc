/* stump.cc
   Jeremy Barnes, 6 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All righs reserved.
   $Source$

   A decision stumps algorithm.  Implementation.
*/

#include "stump.h"
#include "classifier_persist_impl.h"
#include "training_data.h"
#include <algorithm>
#include "jml/utils/sgi_algorithm.h"
#include <boost/progress.hpp>
#include "jml/utils/smart_ptr_utils.h"
#include "jml/utils/pair_utils.h"
#include "jml/utils/vector_utils.h"
#include <iomanip>
#include "jml/utils/environment.h"
#include "config_impl.h"
#include "boosted_stumps.h"
#include "jml/stats/distribution_ops.h"
#include "jml/utils/profile.h"

using namespace std;
using namespace DB;


namespace ML {

size_t num_bucketed = 0, num_non_bucketed = 0;
size_t num_real_early = 0, num_real_not_early = 0;
size_t num_real = 0, num_boolean = 0, num_presence = 0, num_categorical = 0;
size_t num_bucket_early = 0, num_bucket_not_early = 0;

double t_train_stump = 0.0;

namespace {

Env_Option<bool> profile("PROFILE_BOOSTED_STUMPS", false);

struct Stats {
    ~Stats()
    {
        if (profile) {
            cerr << "Stumps profile: " << endl;
            cerr << "  train_all:    " << t_train_stump << endl;
            cerr << "  real:         " << num_real << endl;
            cerr << "    early:      " << num_real_early << endl;
            cerr << "    not early:  " << num_real_not_early << endl;
            cerr << "  boolean:      " << num_boolean << endl;
            cerr << "  presence:     " << num_presence << endl;
            cerr << "  categorical:  " << num_categorical << endl;
            cerr << "  bucketed:     " << num_bucketed << endl;
            cerr << "    early:      " << num_bucket_early << endl;
            cerr << "    not early:  " << num_bucket_not_early << endl;
            cerr << "  non bucketed: " << num_non_bucketed << endl;
        }
    }
} stats;


} // file scope

/*****************************************************************************/
/* ACTION                                                                    */
/*****************************************************************************/

namespace {

void merge_dists(distribution<float> & d1, const distribution<float> & d2,
                 float other_weight)
{
    if (d1.size() != d2.size()) throw Exception("merge_dists");
    d1 += other_weight * d2;
}

} // file scope

void
Action::
merge(const Action & other, float other_weight)
{
    merge_dists(pred_true, other.pred_true, other_weight);
    merge_dists(pred_false, other.pred_false, other_weight);
    merge_dists(pred_missing, other.pred_missing, other_weight);
}

std::string
Action::
print() const
{
    string result;

    if (pred_true.size() == 2
        && abs(pred_true[0] + pred_true[1]) < 0.00001
        && pred_false.size() == 2
        && abs(pred_false[0] + pred_false[1]) < 0.0001)
        result += format("%6.3f %6.3f",
                         pred_false[1], pred_true[1]);
    else result += ostream_format(pred_false)
             + " " + ostream_format(pred_true);
    if (abs(pred_missing).total() > 0.00001)
        result += " " + ostream_format(pred_missing);

    return result;
}

void
Action::
serialize(DB::Store_Writer & store) const
{
    store << (unsigned char)0 << pred_false << pred_true << pred_missing;
}

void
Action::
reconstitute(DB::Store_Reader & store)
{
    unsigned char ver;
    store >> ver;
    if (ver != 0)
        throw Exception("Action::reconstitute(): invalid version");
    store >> pred_false >> pred_true >> pred_missing;
}


/*****************************************************************************/
/* STUMP                                                                     */
/*****************************************************************************/

Stump::Stump()
{
}

Stump::Stump(std::shared_ptr<const Feature_Space> feature_space,
             const Feature & predicted)
    : Classifier_Impl(feature_space, predicted),
      encoding(OE_PROB)
{
}

Stump::
Stump(std::shared_ptr<const Feature_Space> feature_space,
      const Feature & predicted, size_t label_count)
    : Classifier_Impl(feature_space, predicted, label_count),
      encoding(OE_PROB)
{
}

Stump::
Stump(const Feature & predicted,
      const Feature & feature,
      float arg,
      const distribution<float> & pred_true,
      const distribution<float> & pred_false,
      const distribution<float> & pred_missing,
      Update update,
      std::shared_ptr<const Feature_Space> feature_space,
      float Z)
    : Classifier_Impl(feature_space, predicted, pred_true.size()),
      split(feature, arg,
            Split::get_op_from_feature(feature_space->info(feature))),
      Z(Z),
      action(pred_true, pred_false, pred_missing)
{
    encoding = update_to_encoding(update);
}

#if 0
Stump::
Stump(const Feature & predicted,
      const Feature & feature, float arg,
      const distribution<float> & pred_there,
      Update update,
      std::shared_ptr<const Feature_Space> feature_space,
      float Z)
    : Classifier_Impl(feature_space, predicted, pred_there.size()),
      feature(feature), relation(get_relation()), arg(arg), Z(Z),
      pred_true(pred_there), pred_false(pred_there.size(), 0.0)
{
    encoding = update_to_encoding(update);
}
#endif

void Stump::swap(Stump & other)
{
    Classifier_Impl::swap(other);
    split.swap(other.split);
    action.swap(other.action);
    std::swap(Z, other.Z);
    std::swap(encoding, other.encoding);
}

void Stump::scale(float scale)
{
    action *= scale;
}

Stump Stump::scaled(float scale) const
{
    Stump result = *this;
    result.scale(scale);
    return result;
}

void Stump::merge(const Stump & other, float other_weight)
{
    if (label_count() == 0) {
        *this = other;
        scale(other_weight);
        return;
    }

    if (split != other.split
        || label_count() != other.label_count()
        || encoding != other.encoding) {
        cerr << "features: " << feature_space()->print(split.feature()) << ", "
             << feature_space()->print(other.split.feature()) << endl;
        //cerr << "arg: " << arg << ", " << other.arg << endl;
        cerr << "label_count: " << label_count() << ", "
             << other.label_count() << endl;
        throw Exception("Attempt to merge incompatible stumps");
    }
    
    action.merge(other.action, other_weight);
}

Stump::~Stump()
{
}

Stump * Stump::make_copy() const
{
    return new Stump(*this);
}


Label_Dist
Stump::predict(const Feature_Set & features,
               PredictionContext * context) const
{
    Split::Weights weights = split.apply(features);
    return action.apply(weights);
}

float
Stump::predict(int label, const Feature_Set & features,
               PredictionContext * context) const
{
    // TODO: could be optimised.  However, the boosted stumps don't call this,
    // so it is a bit of a moot point.  (We are unlikely to use it without
    // boosting!)
    return predict(features, context).at(label);
}

std::string Stump::print() const
{
    string result = split.print(*feature_space());
    result += " " + action.print();
    return result;
}    

std::string Stump::summary() const
{
    string result;

    result = format("%5.3f ", Z);

    if (action.pred_true.size() == 2
        && abs(action.pred_true[0] + action.pred_true[1]) < 0.00001
        && action.pred_false.size() == 2
        && abs(action.pred_false[0] + action.pred_false[1]) < 0.0001)
        result += format("%6.3f %6.3f",
                         action.pred_false[1], action.pred_true[1]);
    else result += ostream_format(action.pred_false)
             + " " + ostream_format(action.pred_true);
    if (abs(action.pred_missing).total() > 0.00001)
        result += " " + ostream_format(action.pred_missing);

    result += split.print(*feature_space());

    return result;

#if 0
                if (stump.label_count() == 2
                    && (abs(stump.pred_true[0]+stump.pred_true[1]) < 0.001)) {
                    cerr << format(" %5.2f %5.2f", stump.pred_false[1],
                                   stump.pred_true[1]);
                    if (abs(stump.pred_missing[1]) > 0.01)
                        cerr << format(" %5.2f", stump.pred_missing[1]);
                    else cerr << "      ";
                }
                else if (nl <= 5)
                    for (unsigned i = 0;  i < std::min(nl, 5U);  ++i)
                        cerr << format("%6.2f", stump.pred_true[i]);
                else if (verbosity > 3)
                    for (unsigned i = 0;  i < nl;  ++i)
                        cerr << format("%6.2f", stump.pred_true[i]);
#endif
}

std::vector<ML::Feature> Stump::all_features() const
{
    std::vector<ML::Feature> result;
    result.push_back(split.feature());
    return result;
}

namespace {

static const std::string STUMP_MAGIC = "STUMP";
static const compact_size_t STUMP_VERSION = 4;

} // file scope

void Stump::serialize(DB::Store_Writer & store) const
{
    store << STUMP_MAGIC << STUMP_VERSION << compact_size_t(label_count());
    feature_space()->serialize(store, predicted_);
    serialize_lw(store);
    store << encoding;
}

void
Stump::
serialize_lw(DB::Store_Writer & store) const
{
    split.serialize(store, *feature_space());
    action.serialize(store);
    store << Z;
}

void
Stump::
reconstitute_lw(DB::Store_Reader & store, const Feature & feature)
{
    split.reconstitute(store, *feature_space());
    action.reconstitute(store);
    store >> Z;
}

Stump::Stump(DB::Store_Reader & store,
             const std::shared_ptr<const Feature_Space> & feature_space)
{
    string magic;
    compact_size_t version;
    store >> magic >> version;
    if (magic != STUMP_MAGIC)
        throw Exception("Attempt to reconstitute \"" + magic
                        + "\" with stump reconstitutor");
    if (version > STUMP_VERSION)
        throw Exception(format("Attemp to reconstitute stump version %zd, only "
                               "<= %zd supported", version.size_,
                               STUMP_VERSION.size_));
    
    compact_size_t label_count(store);

    predicted_ = MISSING_FEATURE;
    if (version > 0)
        feature_space->reconstitute(store, predicted_);
    
    Classifier_Impl::init(feature_space, predicted_, label_count);
    
    split.reconstitute(store, *feature_space);
    action.reconstitute(store);

    store >> Z;

    if (version >= 3)
        store >> encoding;
    else encoding = OE_PM_INF;
    
#if 0
    if (action.pred_true.size() != label_count
        || action.pred_false.size() != label_count
        || action.pred_missing.size() != label_count) {
        string arg_ser = ostream_format(arg);
        string true_ser = ostream_format(action.pred_true);
        string false_ser = ostream_format(action.pred_false);
        string missing_ser = ostream_format(action.pred_missing);
        string feature_ser = feature_space->print(feature);
        throw Exception("Stump::Stump(reconst): bad params: label_count = "
                        + ostream_format(label_count)
                        + " arg = " + arg_ser
                        + " feature = " + feature_ser
                        + " true = " + true_ser
                        + " false = " + false_ser
                        + " missing = " + missing_ser);
    }
#endif
}

void Stump::
reconstitute(DB::Store_Reader & store,
             const std::shared_ptr<const Feature_Space> & features)
{
    /* Implement the strong exception guarantee. */
    Stump new_me(store, features);
    swap(new_me);
}

Classifier_Impl *
Stump::
merge(const Classifier_Impl & other, float weight) const
{
    const Stump * as_stump = dynamic_cast<const Stump *>(&other);
    if (as_stump) {
        auto_ptr<Boosted_Stumps>
            result(new Boosted_Stumps(feature_space(), predicted()));
        result->insert(*this, 1.0);
        result->insert(*as_stump, weight);
        return result.release();
    }
    
    return 0;
}

Output_Encoding
Stump::
output_encoding() const
{
    return encoding;
}

Output_Encoding Stump::update_to_encoding(Update update_alg)
{
    switch (update_alg) {
    case NORMAL: return OE_PM_INF;
    case GENTLE: return OE_PM_ONE;
    case PROB:   return OE_PROB;
    default:
        throw Exception("Stump::update_to_encoding(): unknown update alg");
    };
}

/*****************************************************************************/
/* REGISTRATION                                                              */
/*****************************************************************************/

namespace {

Register_Factory<Classifier_Impl, Stump> STUMP_REGISTER("STUMP");

} // file scope



} // namespace ML

ENUM_INFO_NAMESPACE

const Enum_Opt<ML::Stump::Update>
Enum_Info<ML::Stump::Update>::OPT[3] = {
    { "normal",      ML::Stump::NORMAL   },
    { "gentle",      ML::Stump::GENTLE   },
    { "prob",        ML::Stump::PROB     } };

const char * Enum_Info<ML::Stump::Update>::NAME
   = "Stump::Update";

END_ENUM_INFO_NAMESPACE

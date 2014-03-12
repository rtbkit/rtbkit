/* perceptron.cc
   Jeremy Barnes, 16 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

   Wrapper around the MLP class to go with the rest of the classifier
   framework.  Implementation.
*/

#include "perceptron.h"
#include "jml/boosting/classifier_persist_impl.h"
#include "jml/utils/profile.h"
#include "jml/utils/environment.h"
#include "jml/algebra/matrix_ops.h"
#include "jml/boosting/training_index.h"
#include "jml/algebra/irls.h"
#include "jml/utils/vector_utils.h"
#include <iomanip>
#include <boost/random/lagged_fibonacci.hpp>
#include <boost/random/uniform_01.hpp>
#include "jml/utils/parse_context.h"
#include "jml/stats/distribution_simd.h"
#include "jml/stats/distribution_ops.h"
#include "jml/boosting/evaluation.h"
#include "jml/arch/simd_vector.h"
#include "jml/algebra/lapack.h"
#include "jml/boosting/config_impl.h"
#include "jml/utils/worker_task.h"
#include "jml/utils/guard.h"
#include <boost/bind.hpp>
#include "jml/arch/backtrace.h"
#include "dense_layer.h"

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
{
}

Perceptron::
Perceptron(const std::shared_ptr<const Feature_Space> & feature_space,
               const Feature & predicted)
    : Classifier_Impl(feature_space, predicted)
{
}

Perceptron::
Perceptron(DB::Store_Reader & reader,
           const std::shared_ptr<const Feature_Space> & feature_space)
{
    this->reconstitute(reader, feature_space);
}

Perceptron::
Perceptron(const std::shared_ptr<const Feature_Space> & feature_space,
           const Feature & predicted,
           size_t label_count)
    : Classifier_Impl(feature_space, predicted, label_count)
{
}

Perceptron::
Perceptron(const Perceptron & other)
    : Classifier_Impl(other), features(other.features),
      layers(other.layers, Deep_Copy_Tag()), output(other.output)
{
}

Perceptron &
Perceptron::
operator = (const Perceptron & other)
{
    Perceptron new_me(other);
    swap(new_me);
    return *this;
}

float
Perceptron::
predict(int label, const Feature_Set & features,
        PredictionContext * context) const
{
    return predict(features).at(label);
}

distribution<float>
Perceptron::
predict(const Feature_Set & fs,
        PredictionContext * context) const
{
    PROFILE_FUNCTION(t_predict);

    float input[layers.inputs()];
    extract_features(fs, input);

    distribution<float> output(layers.outputs());
    layers.apply(input, &output[0]);

    return this->output.decode(output);
}

std::string
Perceptron::
print() const
{
    string result = format("{ Perceptron: %zd layers, %zd inputs, %zd outputs\n",
                           layers.size(), features.size(),
                           layers.outputs());

    result += "  features:\n";
    for (unsigned f = 0;  f < features.size();  ++f)
        result += format("    %d %s\n", f,
                         feature_space()->print(features[f]).c_str());
    result += "\n";

    for (unsigned i = 0;  i < layers.size();  ++i)
        result += format("  layer %d\n", i) + layers[i].print();

    result += "}";

    return result;
}

std::vector<ML::Feature>
Perceptron::
all_features() const
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

void Perceptron::add_layer(const std::shared_ptr<Layer> & layer)
{
    layer->validate();
    layers.add(layer);
}

void Perceptron::clear()
{
    layers.clear();
    features.clear();
}

size_t Perceptron::parameters() const
{
    size_t result = 0;
    for (unsigned l = 1;  l < layers.size();  ++l)
        result += layers[l].parameter_count();
    return result;
}

namespace {

static const std::string PERCEPTRON_MAGIC = "PERCEPTRON";
static const compact_size_t PERCEPTRON_VERSION = 2;


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
    layers.serialize(store);
    output.serialize(store);

    store << string("END PERCEPTRON");
}

void
Perceptron::
reconstitute(DB::Store_Reader & store,
             const std::shared_ptr<const Feature_Space> & feature_space)
{
    /* Implement the strong exception guarantee, except for the store. */
    string magic;
    compact_size_t version;
    store >> magic >> version;
    if (magic != PERCEPTRON_MAGIC)
        throw Exception("Attempt to reconstitute \"" + magic
                                + "\" with perceptrons reconstitutor");
    if (version != PERCEPTRON_VERSION)
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
    new_me.layers.reconstitute(store);
    new_me.output.reconstitute(store);

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
        layers[0].apply(input, &result[x][0]);
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

/* naive_bayes_generator.cc
   Jeremy Barnes, 15 March 2006
   Copyright (c) 2006 Jeremy Barnes  All rights reserved.
   $Source$

   Generator for glz_classifiers.
*/

#include "naive_bayes_generator.h"
#include "registry.h"
#include <boost/timer.hpp>
#include <boost/progress.hpp>
#include "training_index.h"
#include "weighted_training.h"
#include "stump_training.h"
#include "stump_training_core.h"
#include "jml/stats/distribution_ops.h"
#include "jml/utils/smart_ptr_utils.h"


using namespace std;


namespace ML {


/*****************************************************************************/
/* NAIVE_BAYES_GENERATOR                                                     */
/*****************************************************************************/

Naive_Bayes_Generator::
Naive_Bayes_Generator()
{
    defaults();
}

Naive_Bayes_Generator::~Naive_Bayes_Generator()
{
}

void
Naive_Bayes_Generator::
configure(const Configuration & config)
{
    Classifier_Generator::configure(config);
    
    config.find(trace,        "trace");
    config.find(feature_prop, "feature_prop");
}

void
Naive_Bayes_Generator::
defaults()
{
    Classifier_Generator::defaults();
    trace = 0;
    feature_prop = 1.0;
}

Config_Options
Naive_Bayes_Generator::
options() const
{
    Config_Options result = Classifier_Generator::options();
    result
        .add("trace", trace, "0-",
             "trace execution of training in a very fine-grained fashion")
        .add("feature_prop", feature_prop, "0.0<N<=1.0",
             "which proportion of features do we look at");

    return result;
}

void
Naive_Bayes_Generator::
init(std::shared_ptr<const Feature_Space> fs, Feature predicted)
{
    Classifier_Generator::init(fs, predicted);
    model = Naive_Bayes(fs, predicted);
}

std::shared_ptr<Classifier_Impl>
Naive_Bayes_Generator::
generate(Thread_Context & context,
         const Training_Data & training_set,
         const Training_Data & validation_set,
         const distribution<float> & training_ex_weights,
         const distribution<float> & validate_ex_weights,
         const std::vector<Feature> & features, int) const
{
    boost::timer timer;

    Feature predicted = model.predicted();

    boost::multi_array<float, 2> weights
        = expand_weights(training_set, training_ex_weights, predicted);

    Naive_Bayes current
        = train_weighted(context, training_set, weights, features);

    if (verbosity > 2) {
        cerr << endl << "Learned Bayes classifier function: " << endl;
        int nl = current.label_count();
        cerr << "feature                       split";
        if (nl == 2 && false)
            cerr << "       label1";
        else
            for (unsigned l = 0;  l < nl;  ++l)
                cerr << format("    label%-4d", l);
        cerr << endl;
        
        for (unsigned i = 0;  i < current.features.size();  ++i) {
            string feat
                = current.feature_space()
                ->print(current.features[i].feature);
            
            string spaces = feat;
            std::fill(spaces.begin(), spaces.end(), ' ');
            
            for (unsigned j = 0;  j < 3;  ++j) {
                if (j == 2) {
                    float sum_missing = 0.0;
                    
                    for (unsigned l = 0;  l < nl;  ++l)
                        sum_missing += current.probs[i][j][l];
                    
                    if (sum_missing == 0.0) continue;
                }
                if (j == 0) cerr << format("%-30s", feat.c_str());
                else cerr << format("%-30s", spaces.c_str());
                
                if (j == 0) cerr << format(" <  %13f", current.features[i].arg);
                else if (j == 1)
                    cerr << format(" >= %13f", current.features[i].arg);
                else cerr << " MISSING         ";
                
                if (nl == 2 && false)
                    cerr << format("%9.4f", current.probs[i][j][1]);
                else
                    for (unsigned l = 0;  l < nl;  ++l)
                        cerr << format("%9.4f", current.probs[i][j][l]);
                cerr << endl;
            }
            cerr << endl;
        }
    }
    
    return make_sp(current.make_copy());
}

namespace {

/*****************************************************************************/
/* NAIVE_BAYES_ACCUM                                                         */
/*****************************************************************************/

template<class W, class Z, class Tracer = No_Trace>
struct Naive_Bayes_Accum {

    Naive_Bayes_Accum(const Feature_Space & fs, size_t nl, float epsilon,
                      Tracer tracer = Tracer())
        : tracer(tracer), w(2), arg(0.5), z(1.0), fs(fs), nl(nl),
          epsilon(epsilon)
    {
    }

    Tracer tracer;
    Z calc_z;

    W w;
    float arg;
    float z;
    Feature feature;

    const Feature_Space & fs;
    size_t nl;
    float epsilon;

    /* Results go in here. */
    vector<Naive_Bayes::Bayes_Feature> features;
    vector<distribution<float> > dist_true;
    vector<distribution<float> > dist_false;
    vector<distribution<float> > dist_missing;

    /** Method that gets called when we start a new feature.  We use it to
        pre-cache part of the work from the Z calculation, as we are
        assured that the MISSING buckets of W will never change after this
        method is called.
    */
    bool start(const Feature & feature, const W & w, double & missing)
    {
        bool optional = fs.info(feature).optional();
        missing = calc_z.missing(w, optional);
        return true;  // Yes, we can continue...
    }

    /** Method that gets called when we have found a potential split point. */
    float add(const Feature & feature, const W & w, float arg, float z,
              double missing)
    {
        if (tracer)
            tracer("naive bayes accum", 3)
                << "  accum: feature " << feature << " arg " << arg
                << "  z " << z << "  " << fs.print(feature) << endl;
        
        if (z < this->z || this->z == 1.0) {
            // A better one.  This replaces whatever we had accumulated so
            // far.
            this->z = z;
            this->w = w;
            this->arg = arg;
            this->feature = feature;
        }
        
        return z;
    }

    /** Method that gets called when we have found a potential split point. */
    float add(const Feature & feature, const W & w, float arg,
              double missing)
    {
        float z = calc_z.non_missing(w, missing);
        return add(feature, w, arg, z, missing);
    }

    /** Method that gets called when we have found a potential split point. */
    float add_presence(const Feature & feature, const W & w, float arg,
                       double missing)
    {
        float z = calc_z.non_missing_presence(w, missing);
        return add(feature, w, arg, z, missing);
    }

    /** Called once we have tried all split points for a feature.  We know
        where that split point is and the W matrix.  From this, we can
        calculate the probabilities.
    */
    void finish(const Feature & feature)
    {
        Naive_Bayes::Bayes_Feature bayes_feature;
        bayes_feature.feature = feature;
        bayes_feature.arg = arg;

        features.push_back(bayes_feature);

        distribution<float> d_true(nl), d_false(nl), d_missing(nl);

        for (unsigned l = 0; l < nl;  ++l) {
            float w_false   = w(l, false,   true);
            float w_true    = w(l, true,    true);
            float w_missing = w(l, MISSING, true);

            float total = (w_false + w_true + w_missing) * epsilon;

            d_true[l]    = log(std::max(1.0F, w_true    / total));
            d_false[l]   = log(std::max(1.0F, w_false   / total));
            if (!fs.info(feature).optional())
                d_missing[l] = log(std::max(1.0F, w_missing / total));
        }

        dist_true.push_back(d_true);
        dist_false.push_back(d_false);
        dist_missing.push_back(d_missing);

        if (tracer)
            tracer("naive bayes accum", 3)
                << "  finish: feature " << feature << " arg " << arg
                << "  z " << z << "  " << fs.print(feature) << endl
                << "  w: " << w.print() << endl
                << "  epsilon: " << epsilon << endl
                << "  dist_true: " << d_true << endl
                << "  dist_false: " << d_false << endl
                << "  dist_missing: " << d_missing << endl;

        /* Set up for the next go around... */
        z = 1.0;
        arg = 0.5;
    }

    void results(Naive_Bayes & output) const
    {
        vector<pair<Naive_Bayes::Bayes_Feature, int> > sorted;
        sorted.reserve(features.size());
        for (unsigned i = 0;  i < features.size();  ++i)
            sorted.push_back(make_pair(features[i], i));
        sort_on_first_ascending(sorted);
        
        boost::multi_array<float, 3> probs(boost::extents[features.size()][3][nl]);
        distribution<double> missing(nl);
        output.features
            = vector<Naive_Bayes::Bayes_Feature>
                (first_extractor(sorted.begin()),
                 first_extractor(sorted.end()));
        
        for (unsigned f = 0;  f < features.size();  ++f) {
            for (unsigned l = 0;  l < nl;  ++l) {
                int f2 = sorted[f].second;
                probs[f][false][l]   = dist_false[f2][l];
                probs[f][true][l]    = dist_true[f2][l];
                probs[f][MISSING][l] = dist_missing[f2][l];
                missing[l]          += dist_missing[f2][l];
            }
        }

        swap_multi_arrays(output.probs, probs);
        output.missing_total
            = distribution<float>(missing.begin(), missing.end());
    }
};

} // file scope

Naive_Bayes
Naive_Bayes_Generator::
train_weighted(Thread_Context & context,
               const Training_Data & data,
               const boost::multi_array<float, 2> & weights,
               const std::vector<Feature> & features_) const
{
    Naive_Bayes result = model;

    vector<Feature> features = features_;

    for (int i = 0;  i < features.size();  ++i) {
        if (features[i] == model.predicted()) {
            std::swap(features[i], features.back());
            features.pop_back();
        }
    }

    /* Test the given proportion of possible features, and return the best. */
    if (feature_prop < 1.0)
        std::random_shuffle(features.begin(), features.end());
    
    /* Choose the top proportion of them. */
    size_t nf = (size_t)((float)features.size() * feature_prop);
    size_t nl = result.label_count();
    size_t nx = data.example_count();
    
    if (nf == 0)
        throw Exception("Naive_Bayes::train_weighted: attempt to train with "
                        "no features (or feature_prop too low)");
    
    if (result.features.size() < 100) nf = result.features.size();
           
    const vector<Label> & labels = data.index().labels(model.predicted());

    /* Work out the label priors.  These are smoothed as well. */
    distribution<double> class_weights(nl);
    for (unsigned x = 0;  x < nx;  ++x) {
        float total_weight = 0.0;
        if (weights.shape()[1] == 1)
            total_weight = nl * weights[x][0];
        else
            for (unsigned l = 0;  l < nl;  ++l)
                total_weight += weights[x][l];
        class_weights[labels[x]] += total_weight;
    }
    
    for (unsigned l = 0;  l < nl;  ++l)
        class_weights[l] = std::max(class_weights[l], 1.0 / (nx * nl));

    class_weights.normalize();

    //cerr << "class_weights = " << class_weights << endl;

    result.label_priors
        = distribution<float>(class_weights.begin(), class_weights.end());

    {    
        typedef W_normal W;
        typedef Z_normal Z;
        
        typedef Naive_Bayes_Accum<W, Z, Stream_Tracer> Accum;
        typedef Stump_Trainer<W, Z, Stream_Tracer> Trainer;
        
        float epsilon = 1.0 / (nx * nl);
        
        Accum accum(*model.feature_space(), nl, epsilon, trace);
        Trainer trainer(trace);
        
        trainer.test_all(features, data, model.predicted(), LW_Array<const float>(weights),
                         accum);
        
        accum.results(result);
    }

    return result;
}


/*****************************************************************************/
/* REGISTRATION                                                              */
/*****************************************************************************/

namespace {

Register_Factory<Classifier_Generator, Naive_Bayes_Generator>
    NAIVE_BAYES_REGISTER("naive_bayes");

} // file scope

} // namespace ML

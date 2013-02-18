/* glz_classifier_generator.cc
   Jeremy Barnes, 15 March 2006
   Copyright (c) 2006 Jeremy Barnes  All rights reserved.
   $Source$

   Generator for glz_classifiers.
*/

#include "glz_classifier_generator.h"
#include "registry.h"
#include <boost/timer.hpp>
#include <boost/progress.hpp>
#include "training_index.h"
#include "weighted_training.h"
#include "jml/utils/smart_ptr_utils.h"
#include "jml/algebra/matrix_ops.h"
#include "jml/algebra/lapack.h"
#include "jml/arch/timers.h"

using namespace std;


namespace ML {


/*****************************************************************************/
/* GLZ_CLASSIFIER_GENERATOR                                                  */
/*****************************************************************************/

GLZ_Classifier_Generator::
GLZ_Classifier_Generator()
{
    defaults();
}

GLZ_Classifier_Generator::~GLZ_Classifier_Generator()
{
}

void
GLZ_Classifier_Generator::
configure(const Configuration & config)
{
    Classifier_Generator::configure(config);

    config.find(add_bias, "add_bias");
    config.find(do_decode, "decode");
    config.find(link_function, "link_function");
    config.find(normalize, "normalize");
    config.find(ridge_regression, "ridge_regression");
    config.find(feature_proportion, "feature_proportion");
}

void
GLZ_Classifier_Generator::
defaults()
{
    Classifier_Generator::defaults();
    link_function = LOGIT;
    add_bias = true;
    do_decode = true;
    normalize = true;
    ridge_regression = true;
    feature_proportion = 1.0;
}

Config_Options
GLZ_Classifier_Generator::
options() const
{
    Config_Options result = Classifier_Generator::options();
    result
        .add("add_bias", add_bias,
             "add a constant bias term to the classifier?")
        .add("decode", do_decode,
             "run the decoder (link function) after classification?")
        .add("link_function", link_function,
             "which link function to use for the output function");

    return result;
}

void
GLZ_Classifier_Generator::
init(std::shared_ptr<const Feature_Space> fs, Feature predicted)
{
    Classifier_Generator::init(fs, predicted);
    model = GLZ_Classifier(fs, predicted);
}

std::shared_ptr<Classifier_Impl>
GLZ_Classifier_Generator::
generate(Thread_Context & thread_context,
         const Training_Data & training_data,
         const boost::multi_array<float, 2> & weights,
         const std::vector<Feature> & features,
         float & Z,
         int) const
{
    boost::timer timer;

    Feature predicted = model.predicted();

    GLZ_Classifier current(model);
    
    train_weighted(thread_context, training_data, weights, features, current);
    
    if (verbosity > 2) {
        cerr << endl << "Learned GLZ function: " << endl;
        cerr << "link: " << current.link << endl;
        int nl = current.feature_space()->info(predicted).value_count();
        cerr << "feature                                    ";
        if (nl == 2 && false)
            cerr << "       label1";
            else
                for (unsigned l = 0;  l < nl;  ++l)
                    cerr << format("    label%-4d", l);
        cerr << endl;

        for (unsigned i = 0;  i < current.features.size() + current.add_bias;
             ++i) {

            if (i == current.features.size()) {
                cerr << format("%-40s", "BIAS");
            }
            else {
                string feat
                    = current.feature_space()
                    ->print(current.features[i].feature);
                cerr << format("%-36s", feat.c_str());
                
                switch (current.features[i].type) {
                case GLZ_Classifier::Feature_Spec::VALUE:
                    cerr << " VAL";
                    break;
                case GLZ_Classifier::Feature_Spec::VALUE_IF_PRESENT:
                    cerr << " VIP";
                    break;
                case GLZ_Classifier::Feature_Spec::PRESENCE:
                    cerr << " PRS";
                    break;
                default:
                    throw Exception("invalid type");
                }
            }
            
            if (nl == 2 && false)
                cerr << format("%13f", current.weights[1][i]);
            else
                for (unsigned l = 0;  l < nl;  ++l)
                    cerr << format("%13f", current.weights[l][i]);
            cerr << endl;
        }
        cerr << endl;
    }

    Z = 0.0;
    
    return make_sp(current.make_copy());
}

float
GLZ_Classifier_Generator::
train_weighted(Thread_Context & thread_context,
               const Training_Data & data,
               const boost::multi_array<float, 2> & weights,
               const std::vector<Feature> & unfiltered,
               GLZ_Classifier & result) const
{
    /* Algorithm:
       1.  Convert training data to a dense format;
       2.  Train on each column
    */

    result = model;
    result.features.clear();
    result.add_bias = add_bias;
    result.link = (do_decode ? link_function : LINEAR);

    Feature predicted = model.predicted();
    
    for (unsigned i = 0;  i < unfiltered.size();  ++i) {
        if (unfiltered[i] == model.predicted())
            continue;  // don't use the label to predict itself

        // If we don't want to use all features then take a random subset
        if (feature_proportion < 1.0
            && thread_context.random01() > feature_proportion)
            continue;

        GLZ_Classifier::Feature_Spec spec(unfiltered[i]);

        // Can't use a feature that has multiple occurrences
        if (!data.index().only_one(unfiltered[i])) continue;

        if (data.index().exactly_one(unfiltered[i])) {
            // Feature that's always there but constant has no information
            if (data.index().constant(unfiltered[i])) continue;
            result.features.push_back(spec);
        }
        else {
            if (!data.index().constant(unfiltered[i])) {
                spec.type = GLZ_Classifier::Feature_Spec::VALUE_IF_PRESENT;
                result.features.push_back(spec);
            }
            spec.type = GLZ_Classifier::Feature_Spec::PRESENCE;
            result.features.push_back(spec);
        }
    }
    
    size_t nl = result.label_count();        // Number of labels
    bool regression_problem = (nl == 1);
    size_t nx = data.example_count();        // Number of examples
    size_t nv = result.features.size();      // Number of variables
    if (add_bias) ++nv;

    distribution<double> total_weight(nx);
    for (unsigned x = 0;  x < nx;  ++x) {
        for (unsigned l = 0;  l < weights.shape()[1];  ++l)
            total_weight[x] += weights[x][l];
    }

    size_t nx2 = (total_weight != 0.0).count();

    //cerr << "nx = " << nx << " nv = " << nv << " nx * nv = " << nx * nv
    //     << endl;

    Timer t;

    /* Get the labels by example. */
    const vector<Label> & labels = data.index().labels(predicted);
    
    // Use double precision, we have enough memory (<= 1GB)
    // NOTE: always on due to issues with convergence
    boost::multi_array<double, 2> dense_data(boost::extents[nv][nx2]);  // training data, dense
        
    distribution<double> model(nx2, 0.0);  // to initialise weights, correct
    vector<distribution<double> > w(nl, model);       // weights for each label
    vector<distribution<double> > correct(nl, model); // correct values

    cerr << "setup: " << t.elapsed() << endl;
    t.restart();
        
    int x2 = 0;
    for (unsigned x = 0;  x < nx;  ++x) {
        if (total_weight[x] == 0.0) continue;

        distribution<float> decoded = result.decode(data[x]);
        if (add_bias) decoded.push_back(1.0);
            
        //cerr << "x = " << x << "  decoded = " << decoded << endl;
            
        /* Record the values of the variables. */
        assert(decoded.size() == nv);
        for (unsigned v = 0;  v < decoded.size();  ++v) {
            if (!isfinite(decoded[v])) decoded[v] = 0.0;
            dense_data[v][x2] = decoded[v];
        }
            
        /* Record the correct label. */
        if (regression_problem) {
            correct[0][x2] = labels[x].value();
            w[0][x2] = weights[x][0];
        }
        else if (nl == 2 && weights.shape()[1] == 1) {
            correct[0][x2] = (double)(labels[x] == 0);
            correct[1][x2] = (double)(labels[x] == 1);
            w[0][x2] = weights[x][0];
        }
        else {
            for (unsigned l = 0;  l < nl;  ++l) {
                correct[l][x2] = (double)(labels[x] == l);
                w[l][x2] = weights[x][l];
            }
        }

        ++x2;
    }

    if (x2 != nx2)
        throw Exception("x2 not nx2");

    cerr << "marshalling: " << t.elapsed() << endl;
    t.restart();

    distribution<double> means(nv), stds(nv, 1.0);

    /* Scale */
    for (unsigned v = 0;  v < nv && normalize;  ++v) {
        double total = 0.0;

        for (unsigned x = 0;  x < nx2;  ++x)
            total += dense_data[v][x];

        double mean = total / nx2;

        double std_total = 0.0;
        for (unsigned x = 0;  x < nx2;  ++x)
            std_total
                += (dense_data[v][x] - mean)
                *  (dense_data[v][x] - mean);
            
        double std = sqrt(std_total / nx2);

        if (std == 0.0 && mean == 1.0) {
            // bias column
            std = 1.0;
            mean = 0.0;
        }
        else if (std == 0.0)
            std = 1.0;
            
        double std_recip = 1.0 / std;

        for (unsigned x = 0;  x < nx2;  ++x)
            dense_data[v][x] = (dense_data[v][x] - mean) * std_recip;

        means[v] = mean;
        stds[v] = std;
    }

    cerr << "normalization: " << t.elapsed() << endl;
    t.restart();

    int nlr = nl;
    if (nl == 2) nlr = 1;
        
    /* Perform a GLZ for each label. */
    result.weights.clear();
    double extra_bias = 0.0;
    for (unsigned l = 0;  l < nlr;  ++l) {
        //cerr << "l = " << l << "  correct[l] = " << correct[l]
        //     << " w = " << w[l] << endl;
            
        distribution<double> trained
            = perform_irls(correct[l], dense_data, w[l], link_function,
                           ridge_regression);

        trained /= stds;
        extra_bias = - (trained.dotprod(means));

        if (extra_bias != 0.0) {
            if (!add_bias)
                throw Exception("extra bias but nowhere to put it");
            trained.back() += extra_bias;
        }
        
        //cerr << "l = " << l <<"  param = " << param << endl;
            
        result.weights.push_back(trained.cast<float>());
    }

    cerr << "irls: " << t.elapsed() << endl;
    t.restart();
        
    if (nl == 2) {
        // weights for second label are the mirror of those of the first
        // label
        result.weights.push_back(-1.0F * result.weights.front());
    }

    //cerr << "glz_classifier: irls time " << t.elapsed() << "s" << endl;
    
    return 0.0;
}


/*****************************************************************************/
/* REGISTRATION                                                              */
/*****************************************************************************/

namespace {

Register_Factory<Classifier_Generator, GLZ_Classifier_Generator>
    GLZ_CLASSIFIER_REGISTER("glz");

} // file scope

} // namespace ML

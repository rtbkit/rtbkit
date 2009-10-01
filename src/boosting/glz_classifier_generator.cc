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
#include "utils/smart_ptr_utils.h"


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
}

void
GLZ_Classifier_Generator::
defaults()
{
    Classifier_Generator::defaults();
    link_function = LOGIT;
    add_bias = true;
    do_decode = true;
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
init(boost::shared_ptr<const Feature_Space> fs, Feature predicted)
{
    Classifier_Generator::init(fs, predicted);
    model = GLZ_Classifier(fs, predicted);
}

boost::shared_ptr<Classifier_Impl>
GLZ_Classifier_Generator::
generate(Thread_Context & context,
         const Training_Data & training_data,
         const boost::multi_array<float, 2> & weights,
         const std::vector<Feature> & features,
         float & Z,
         int) const
{
    boost::timer timer;

    Feature predicted = model.predicted();

    GLZ_Classifier current(model);
    
    //for (unsigned i = 0;  i < 20;  ++i)
    //    cerr << "weights[" << i << "][0] = " << weights[i][0] << endl;

    //boost::multi_array<float, 2> weights
    //    = expand_weights(training_data, weights_, predicted);

    train_weighted(training_data, weights, features, current);
    
    if (verbosity > 2) {
        cerr << endl << "Learned GLZ function: " << endl;
        int nl = current.feature_space()->info(predicted).value_count();
        cerr << "feature                                 ";
        if (nl == 2 && false)
            cerr << "       label1";
            else
                for (unsigned l = 0;  l < nl;  ++l)
                    cerr << format("    label%-4d", l);
        cerr << endl;

        for (unsigned i = 0;  i < current.features.size() + current.add_bias;
             ++i) {
            string feat;
            if (i == current.features.size()) feat = "BIAS";
            else feat = current.feature_space()->print(current.features[i]);
            cerr << format("%-40s", feat.c_str());
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
train_weighted(const Training_Data & data,
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
        if (data.index().exactly_one(unfiltered[i]))
            result.features.push_back(unfiltered[i]);
    }
    
    size_t nl = result.label_count();        // Number of labels
    bool regression_problem = (nl == 1);
    size_t nx = data.example_count();        // Number of examples
    size_t nv = result.features.size();      // Number of variables
    if (add_bias) ++nv;

    //cerr << "nx = " << nx << " nv = " << nv << " nx * nv = " << nx * nv
    //     << endl;

    boost::timer t;

    /* Get the labels by example. */
    const vector<Label> & labels = data.index().labels(predicted);
    
    if ((unsigned long)nv * (unsigned long)nx <= 100000000 || true) {
        // Use double precision, we have enough memory (<= 1GB)
        // NOTE: always on due to issues with convergence
        boost::multi_array<double, 2> dense_data(boost::extents[nv][nx]);  // training data, dense
        
        distribution<double> model(nx, 0.0);  // to initialise weights, correct
        vector<distribution<double> > w(nl, model);       // weights for each label
        vector<distribution<double> > correct(nl, model); // correct values
        
        for (unsigned x = 0;  x < nx;  ++x) {
            distribution<float> decoded = result.decode(data[x]);
            if (add_bias) decoded.push_back(1.0);
            
            //cerr << "x = " << x << "  decoded = " << decoded << endl;
            
            /* Record the values of the variables. */
            assert(decoded.size() == nv);
            for (unsigned v = 0;  v < decoded.size();  ++v) {
                if (!isfinite(decoded[v])) decoded[v] = 0.0;
                dense_data[v][x] = decoded[v];
            }
            
            /* Record the correct label. */
            if (regression_problem) {
                correct[0][x] = labels[x].value();
                w[0][x] = weights[x][0];
            }
            else if (nl == 2 && weights.shape()[1] == 1) {
                correct[0][x] = (double)(labels[x] == 0);
                correct[1][x] = (double)(labels[x] == 1);
                w[0][x] = weights[x][0];
            }
            else {
                for (unsigned l = 0;  l < nl;  ++l) {
                    correct[l][x] = (double)(labels[x] == l);
                    w[l][x] = weights[x][l];
                }
            }
        }
        
        /* Remove linearly dependent columns. */
        vector<int> dest = remove_dependent(dense_data);
        
        int nlr = nl;
        if (nl == 2) nlr = 1;
        
        /* Perform a GLZ for each label. */
        result.weights.clear();
        for (unsigned l = 0;  l < nlr;  ++l) {
            //cerr << "l = " << l << "  correct[l] = " << correct[l]
            //     << " w = " << w[l] << endl;
            distribution<double> trained
                = run_irls(correct[l], dense_data, w[l], link_function);
            
            distribution<float> param(nv);
            for (unsigned v = 0;  v < nv;  ++v)
                if (dest[v] != -1) param[v] = trained[dest[v]];
            
        
            //cerr << "l = " << l <<"  param = " << param << endl;
            
            result.weights
                .push_back(distribution<float>(param.begin(), param.end()));
        }
        
        if (nl == 2) {
            // weights for second label are the mirror of those of the first
            // label
            result.weights.push_back(-1.0F * result.weights.front());
        }
    }
    else {
        // Use single precision to avoid memory problems
        // (NOTE: disabled; it is too unstable with single precision)
        boost::multi_array<float, 2> dense_data(boost::extents[nv][nx]);  // training data, dense
        
        distribution<float> model(nx, 0.0);  // to initialise weights, correct
        vector<distribution<float> > w(nl, model);       // weights for each label
        vector<distribution<float> > correct(nl, model); // correct values
        
        for (unsigned x = 0;  x < nx;  ++x) {
            distribution<float> decoded = result.decode(data[x]);
            if (add_bias) decoded.push_back(1.0);
            
            //cerr << "x = " << x << "  decoded = " << decoded << endl;
            
            /* Record the values of the variables. */
            assert(decoded.size() == nv);
            for (unsigned v = 0;  v < decoded.size();  ++v) {
                if (!isfinite(decoded[v])) decoded[v] = 0.0;
                dense_data[v][x] = decoded[v];
            }
            
            /* Record the correct label. */
            if (regression_problem) {
                correct[0][x] = labels[x].value();
                w[0][x] = weights[x][0];
            }
            else {
                for (unsigned l = 0;  l < nl;  ++l) {
                    correct[l][x] = (float)(labels[x] == l);
                    w[l][x] = weights[x][l];
                }
            }
        }
        
        /* Remove linearly dependent columns. */
        vector<int> dest = remove_dependent(dense_data);
        
        boost::timer t;
        
        int nlr = nl;
        if (nl == 2) nlr = 1;
        
        /* Perform a GLZ for each label. */
        result.weights.clear();
        for (unsigned l = 0;  l < nlr;  ++l) {
            //cerr << "l = " << l << "  correct[l] = " << correct[l]
            //     << " w = " << w[l] << endl;
            distribution<float> trained
                = run_irls(correct[l], dense_data, w[l], link_function);
            
            distribution<float> param(nv);
            for (unsigned v = 0;  v < nv;  ++v)
                if (dest[v] != -1) param[v] = trained[dest[v]];
            
        
            //cerr << "l = " << l <<"  param = " << param << endl;
            
            result.weights
                .push_back(distribution<float>(param.begin(), param.end()));
        }
        
        if (nl == 2) {
            // weights for second label are the mirror of those of the first
            // label
            result.weights.push_back(-1.0F * result.weights.front());
        }
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

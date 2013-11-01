/* probabilizer.cc
   Jeremy Barnes, 13 June 2003
   Copyright (c) 2003 Jeremy Barnes.  All rights reserved.
   $Source$

*/

#include "probabilizer.h"
#include "registry.h"
#include "training_data.h"
#include "training_index.h"
#include "classifier.h"
#include "jml/stats/moments.h"
#include "jml/stats/distribution_ops.h"
#include "jml/stats/distribution_simd.h"
#include "jml/math/xdiv.h"
#include "registry.h"
#include "jml/utils/environment.h"
#include "jml/utils/vector_utils.h"
#include <cmath>
#include <fstream>
#include "config_impl.h"
#include "jml/utils/exc_assert.h"
#include "jml/arch/backtrace.h"
#include "jml/arch/simd_vector.h"


using namespace std;
using namespace ML::DB;

using std::sqrt;


namespace ML {

namespace {
Env_Option<bool> debug("DEBUG_PROBABILIZER", false);
}


/*****************************************************************************/
/* GLZ_PROBABILIZER                                                          */
/*****************************************************************************/

GLZ_Probabilizer::GLZ_Probabilizer()
    : link(LOGIT)
{
}

GLZ_Probabilizer::~GLZ_Probabilizer()
{
}

bool debugProbabilizer = false;

distribution<float>
GLZ_Probabilizer::apply(const distribution<float> & input) const
{
    static std::mutex lock;
    std::unique_lock<std::mutex> guard(lock, std::defer_lock);

    //cerr << endl << endl;
    //backtrace();
    //cerr << "applying: params = " << params << " input = " << input << endl;

    if (params.size() == 0)
        throw Exception("applying untrained glz probabilizer");
    if (params.size() != input.size()) {
        cerr << "input = " << input << endl;
        cerr << "params.size() = " << params.size() << endl;
        throw Exception("applying glz probabilizer to wrong num classes");
    }

    bool regression = params.size() == 1;

    distribution<float> input2(input);  // add inputs
    if (!regression) input2.push_back(input.max());      // add max term
    input2.push_back(1.0);              // add bias term

    if (debug & 2) cerr << "link = " << link << endl;

    if (debug & 2) cerr << "input2 = " << input2 << endl;
    
    distribution<float> result(input.size());
    
    for (unsigned i = 0;  i < input.size();  ++i)
        result[i] = apply_link_inverse(SIMD::vec_dotprod_dp(&input2[0], &params[i][0], input2.size()), link);
    
    if (debug & 2) cerr << "result = " << result << endl;

    if (result.total() > 0.0 && !regression) result.normalize();
    
    if (debug & 2) cerr << "normalized = " << result << endl;

#if 1
    bool all_ok = true;
    for (unsigned i = 0;  i < result.size();  ++i)
        if (!isfinite(result[i])) all_ok = false;

    if (!all_ok || (result.total() == 0.0 && link != LINEAR)) {
        guard.lock();
        cerr << "applying GLZ_Probabilizer lead to zero results: " << endl;
        cerr << "  input = " << input << endl;
        cerr << "  input2 = " << input2 << endl;
        cerr << "  result = " << result << endl;
        for (unsigned i = 0;  i < params.size();  ++i)
            cerr << "  params[" << i << "] = " << params[i] << endl;
        for (unsigned i = 0;  i < params.size();  ++i) {
            cerr << "  params[" << i << "] * input2 = "
                 << params[i] * input2 << " (total "
                 << (params[i] * input2).total() << ")" << endl;
        }
        for (unsigned i = 0;  i < input.size();  ++i)
            result[i]
                = apply_link_inverse((input2 * params[i]).total(), link);
        cerr << "  result before normalize = " << result << endl;
        cerr << "  total = " << result.total() << endl;
    }
#endif

    if (debugProbabilizer) {
        guard.lock();
        cerr << "probabilizer " << params << " input " << input << " output "
             << result << endl;
    }
    
    return result;
}

GLZ_Probabilizer
GLZ_Probabilizer::
construct_sparse(const distribution<double> & params, size_t label_count,
                 Link_Function link)
{
    /* If x, y and z are the params, then we create a matrix
       
       x 0 0 ... 0 y z
       0 x 0 ... 0 y z
       0 0 x ... 0 y z
       : : :     : : :
       0 0 0 ... x y z
     */

    GLZ_Probabilizer result;
    result.link = link;

    result.params.clear();

    size_t ol = label_count + 2;  // number of columns to create

    for (unsigned l = 0;  l < label_count;  ++l) {
        distribution<float> expanded(ol, 0.0);
        expanded[l]      = params[0];
        expanded[ol - 2] = params[1];
        expanded[ol - 1] = params[2];
        
        result.params.push_back(expanded);
    }

    return result;
}

void GLZ_Probabilizer::
add_data_sparse(std::vector<distribution<double> > & data,
                const std::vector<float> & output, int correct_label)
{
    size_t nl = output.size();
    if (nl == 0 || correct_label < 0 || correct_label >= nl)
        throw Exception("GLZ_Probabilizer::add_data_sparse: input invariant");
    
    double max = *std::max_element(output.begin(), output.end());
    distribution<double> entry(4);
    for (unsigned l = 0;  l < nl;  ++l) {
        entry[0] = output[l];
        entry[1] = max;
        entry[2] = (l == correct_label);

        if (l == correct_label) entry[3] = 1.0;
        else entry[3] = 1.0 / (nl - 1);

        data.push_back(entry);
    }
}

void GLZ_Probabilizer::
add_data_sparse(std::vector<distribution<double> > & data,
                const Training_Data & training_data,
                const Classifier_Impl & classifier,
                const Optimization_Info & opt_info)
{
    size_t nx = training_data.example_count();

    const vector<Label> & labels
        = training_data.index().labels(classifier.predicted());
    
    /* Go through the training examples one by one, and record the
       outputs. */
    for (unsigned x = 0;  x < nx;  ++x) {
        distribution<float> output
            = classifier.predict(training_data[x], opt_info);
        add_data_sparse(data, output, labels[x]);
    }
}

distribution<double> GLZ_Probabilizer::
train_sparse(const std::vector<distribution<double> > & data,
             Link_Function link,
             const distribution<double> & weights_)
{
    size_t nd = data.size();
    
    /* Convert to the correct data structures. */

    /* If weights passed in were empty, then we use uniform weights. */
    distribution<double> weights = weights_;
    if (weights.empty()) weights.resize(nd, 1.0);

    boost::multi_array<double, 2> outputs(boost::extents[3][nd]);  // value, max, bias
    distribution<double> correct(nd);
    
    for (unsigned d = 0;  d < nd;  ++d) {
        outputs[0][d] = data[d][0];
        outputs[1][d] = data[d][1];
        outputs[2][d] = 1.0;
        correct[d]    = data[d][2];
        weights[d]   *= data[d][3];
    }
    
    /* Perform the GLZ. */
    distribution<double> param(3, 0.0);

    Ridge_Regressor regressor;
    return run_irls(correct, outputs, weights, link, regressor);
}

void GLZ_Probabilizer::
train_glz_regress(const boost::multi_array<double, 2> & outputs,
                  const distribution<double> & correct,
                  const distribution<float> & weights,
                  bool debug)
{
    distribution<double> w(weights.begin(), weights.end());

    /* Perform the GLZ. */
    distribution<double> param = run_irls(correct, outputs, w, link);

    params.clear();
    params.push_back(distribution<float>(param.begin(), param.end()));
}

void GLZ_Probabilizer::
train_identity_regress(const boost::multi_array<double, 2> & outputs,
                       const distribution<double> & correct,
                       const distribution<float> & weights,
                       bool debug)
{
    size_t ol = outputs.shape()[0];

    /* Make the parameterization be an identity function. */
    params.clear();
    distribution<float> p(ol, 0.0);
    p[0] = 1.0;
    params.push_back(p);

    if (debug)
        cerr << "params for " << 0 << " = " << params[0] << endl;
}

distribution<float>
GLZ_Probabilizer::
train_one_mode0(const boost::multi_array<double, 2> & outputs,
                const distribution<double> & correct,
                const distribution<double> & w,
                const vector<int> & dest) const
{
    size_t ol = dest.size();

    distribution<double> param = run_irls(correct, outputs, w, link);
    
    /* Expand the learned distribution back into its proper size,
       putting back the removed columns as zeros. */
    distribution<float> output(ol, 0.0);
    for (unsigned i = 0;  i < ol;  ++i)
        output[i] = (dest[i] == -1 ? 0.0 : param[dest[i]]);

    return output;
}

void GLZ_Probabilizer::
train_mode0(boost::multi_array<double, 2> & outputs,
            const std::vector<distribution<double> > & correct,
            const distribution<int> & num_correct,
            const distribution<float> & weights,
            bool debug)
{
    size_t nl = outputs.shape()[0] - 2;
    size_t nx = outputs.shape()[1];

    /* Train the mode2 over all of the data.  We use these ones for where we
       don't have enough data or we get a negative value for our output. */
    train_mode2(outputs, correct, num_correct, weights, false);

    /* Remove any linearly dependent rows. */
    vector<int> dest = remove_dependent(outputs);

    /* Learn glz as a whole matrix.  Needs lots of data to work well. */
    distribution<double> w(weights.begin(), weights.end());
        
    /* Perform a GLZ for each column.  TODO: really use the GLZ class. */
    for (unsigned l = 0;  l < nl;  ++l) {
        if (num_correct[l] > 0 && num_correct[l] <= nx) {
            distribution<float> trained
                = train_one_mode0(outputs, correct[l], w, dest);
            if (trained[l] > 0.0) params[l] = trained;
        }
        if (debug)
            cerr << "params for " << l << " = " << params.at(l) << endl;
    }
}

distribution<float>
GLZ_Probabilizer::
train_one_mode1(const boost::multi_array<double, 2> & outputs,
                const distribution<double> & correct,
                const distribution<double> & w,
                int l) const
{
    size_t ol = outputs.shape()[0];
    size_t nx = outputs.shape()[1];

    //cerr << "ol = " << ol << " nx = " << nx << endl;

    boost::multi_array<double, 2> outputs2(boost::extents[3][nx]);  // value, max, bias
    for (unsigned x = 0;  x < nx;  ++x) {
        outputs2[0][x] = outputs[l][x];
        outputs2[1][x] = outputs[ol - 2][x];
        outputs2[2][x] = outputs[ol - 1][x];
    }
    
    vector<int> dest = remove_dependent(outputs2);
    
    //cerr << "dest = " << dest << endl;

    distribution<double> param(3, 0.0);
    param = run_irls(correct, outputs2, w, link);
    
    /* Put the 3 columns back into ol columns, so that the output is
       still compatible. */
    distribution<float> param2(ol, 0.0);
    param2[l]      = (dest[0] == -1 ? 0.0 : param[dest[0]]);
    param2[ol - 2] = (dest[1] == -1 ? 0.0 : param[dest[1]]);
    param2[ol - 1] = (dest[2] == -1 ? 0.0 : param[dest[2]]);
    
    return param2;
}


void GLZ_Probabilizer::
train_mode1(const boost::multi_array<double, 2> & outputs,
            const std::vector<distribution<double> > & correct,
            const distribution<int> & num_correct,
            const distribution<float> & weights,
            bool debug)
{
    size_t nl = outputs.shape()[0] - 2;

    /* Train the mode2 over all of the data.  We use these ones for where we
       don't have enough data or we get a negative value for our output. */
    train_mode2(outputs, correct, num_correct, weights, false);

    distribution<double> w(weights.begin(), weights.end());
        
    /* Perform a GLZ for each column.  TODO: really use the GLZ class. */
    for (unsigned l = 0;  l < nl;  ++l) {
        if (num_correct[l] > 0) {
            distribution<float> trained
                = train_one_mode1(outputs, correct[l], w, l);
            if (trained[l] > 0.0) params[l] = trained;
        }
        
        if (debug)
            cerr << "params for " << l << " = " << params[l] << endl;
    }
}

void GLZ_Probabilizer::
train_mode2(const boost::multi_array<double, 2> & outputs,
            const std::vector<distribution<double> > & correct,
            const distribution<int> & num_correct,
            const distribution<float> & weights,
            bool debug)
{
    size_t nl = outputs.shape()[0] - 2;
    size_t ol = outputs.shape()[0];
    size_t nx = outputs.shape()[1];

    //debug = true;

    /* Are there any linearly dependent columns in the outputs?  If so, we
       don't try to include them. */
    boost::multi_array<double, 2> independent = outputs;

    if (debug) {
        for (unsigned l = 0;  l < nl;  ++l)
        for (unsigned x = 0;  x < nx;  ++x)
            independent[l][x] = outputs[l][x];
        
        cerr << "training mode 2" << endl;
    }

    vector<distribution<double> > reconstruct;
    vector<int> removed = remove_dependent_impl(independent, reconstruct);

    if (debug) {
        for (unsigned x = 0;  x < std::min<size_t>(nx, 10);  ++x) {
            for (unsigned l = 0;  l < ol;  ++l)
                cerr << format("%10f ", outputs[l][x]);
            cerr << " : ";
            
            for (unsigned l = 0;  l < nl;  ++l)
                cerr << format("%3f ", correct[l][x]);
            
            cerr << "  -->  ";
            for (unsigned l = 0;  l < nl;  ++l)
                cerr << format("%10f ", independent[l][x]);
            cerr << endl;
        }

        cerr << "removed = " << removed << endl;
        for (unsigned i = 0;  i < reconstruct.size();  ++i)
            cerr << "reconstruct[" << i << "] = " << reconstruct[i] << endl;
        
        cerr << outputs.shape()[0] << "x" << outputs.shape()[1]
             << " matrix" << endl;
    }
    
    distribution<bool> skip(removed.size());
    int nlu = 0;  // nl unskipped
    for (unsigned l = 0;  l < nl;  ++l) {
        skip[l] = removed[l] == -1;
        if (!skip[l]) ++nlu;
    }

    if (debug) {
        cerr << "skip = " << skip << endl;
        cerr << "nlu = " << nlu << endl;
        cerr << "weights.min() = " << weights.min()
             << " weights.max() = " << weights.max()
             << " weights.total() = " << weights.total()
             << endl;
        cerr << "num_correct = " << num_correct << endl;
    }

    // In the case that one output is always less than another, we'll have
    // a problem with the max column being identical to the higher output.
    // In this case, we end up trying to skip all of the columns.
    //
    // Solutions to this are:
    // 1.  Using ridge regression;
    // 2.  if all labels are skipped, then skip the input instead

    bool ridge = true;   // can't think of where we wouldn't want it...
    
    if (nlu == 0) {
        nlu = nl;
        for (unsigned i = 0;  i < nlu;  ++i)
            skip[i] = false;
        ridge = true;
    }
    

    // very underparamerized version, to avoid overfitting
    // We learn a single GLZ, with the data as all of the labels together.
    // The predictions here get weighted in such a way that the correct
    // ones are more important than the incorrect ones.
        
    distribution<double> w(nx * nlu, 1.0);
        
    /* Get the entire enormous data set. */
    boost::multi_array<double, 2> outputs2(boost::extents[3][nx * nlu]);  // value, max, bias
    distribution<double> correct2(nx * nlu);

    for (unsigned x = 0;  x < nx;  ++x) {
        int li = 0;
        for (unsigned l = 0;  l < nl;  ++l) {
            if (skip[l]) continue;
            outputs2[0][x*nlu + li] = outputs[l     ][x];
            outputs2[1][x*nlu + li] = outputs[ol - 2][x];
            outputs2[2][x*nlu + li] = outputs[ol - 1][x];
            correct2   [x*nlu + li] = correct[l][x];
            w          [x*nlu + li] = 1.0;

            /* Note: this doesn't work when we have a large number of
               classes.  It causes all of the outputs to have a high
               result, which in turn causes them to all be about 1/nl
               when normalized.
            */
            //if (correct[l][x] == 1.0) w[x*nl + l] = nl;
            //else w[x*nl + l] = 1.0;

            w[x*nlu + li] *= weights[x];
            ++li;
        }
    }
        
    /* Perform the GLZ. */

    distribution<double> param;
    if (ridge) {
        Ridge_Regressor regressor;
        param = run_irls(correct2, outputs2, w, link, regressor);
    }
    else {
        param = run_irls(correct2, outputs2, w, link);
    }
    
    if (debug)
        cerr << "param = " << param << endl;

    /* Constuct it from the results. */
    *this = construct_sparse(param, nl, link);
    if (debug)
        for (unsigned l = 0;  l < nl;  ++l)
            cerr << "params for " << l << " = " << params[l] << endl;
}

void GLZ_Probabilizer::
train_mode3(const boost::multi_array<double, 2> & outputs,
            const std::vector<distribution<double> > & correct,
            const distribution<int> & num_correct,
            const distribution<float> & weights,
            bool debug)
{
    size_t nl = outputs.shape()[0] - 2;
    size_t ol = outputs.shape()[0];

    /* Make the parameterization be an identity function. */
    params.clear();
    distribution<float> p(ol, 0.0);
    for (unsigned l = 0;  l < nl;  ++l) {
        p[l] = 1.0;
        params.push_back(p);
        p[l] = 0.0;
    }
    if (debug)
        for (unsigned l = 0;  l < nl;  ++l)
            cerr << "params for " << l << " = " << params[l] << endl;
}   

void GLZ_Probabilizer::
train_mode4(const boost::multi_array<double, 2> & outputs,
            const std::vector<distribution<double> > & correct,
            const distribution<int> & num_correct,
            const distribution<float> & weights,
            bool debug)
{
    size_t nl = outputs.shape()[0] - 2;
    boost::multi_array<double, 2> outputs_nodep = outputs;
    
    /* Remove any linearly dependent rows. */
    vector<int> dest = remove_dependent(outputs_nodep);

    /* Get our weights vector ready. */
    distribution<double> w(weights.begin(), weights.end());

    /* Cutoff points. */
    int min_mode0_examples = 50;
    int min_mode1_examples = 20;

    /* Train the mode2 over all of the data.  We use this as a fallback. */
    train_mode2(outputs, correct, num_correct, weights, false);
    
    /* Now select between them for each row. */
    for (unsigned l = 0;  l < nl;  ++l) {
        distribution<float> trained;
        if (num_correct[l] >= min_mode0_examples)
            trained = train_one_mode0(outputs_nodep, correct[l], w, dest);
        else if (num_correct[l] >= min_mode1_examples)
            trained = train_one_mode1(outputs, correct[l], w, l);
        if (trained.size() && trained[l] > 0.0) params[l] = trained;
        
        if (debug)
            cerr << "params for " << l << " (" << num_correct[l] << " ex) = "
                 << params[l] << endl;
    }
}

void GLZ_Probabilizer::
train_mode5(const boost::multi_array<double, 2> & outputs,
            const std::vector<distribution<double> > & correct,
            const distribution<int> & num_correct,
            const distribution<float> & weights,
            bool debug)
{
    size_t nl = outputs.shape()[0] - 2;
    size_t ol = outputs.shape()[0];
    size_t nx = outputs.shape()[1];

    //cerr << "nl = " << nl << " nx = " << nx << " ol = " << ol << endl;

    if (nl != 2)
        throw Exception("GLZ_Probabilizer::train_mode5(): "
                        "not a binary classifer");
    
    /* We learn the zero label only */
    distribution<double> w(nx, 1.0);
        
    /* Get the entire enormous data set. */
    boost::multi_array<double, 2> outputs2(boost::extents[3][nx]);  // value, max, bias
    distribution<double> correct2(nx);
    
    for (unsigned x = 0;  x < nx;  ++x) {
        outputs2[0][x] = outputs[0][x];
        outputs2[1][x] = outputs[2][x];
        outputs2[2][x] = outputs[3][x];
        correct2   [x] = correct[0][x];
        w          [x] = weights[x];
    }

    //cerr << "correct2 = " << correct2 << endl;
    //cerr << "w = " << w << endl;

    //cerr << "performing GLZ" << endl;
    
    /* Perform the GLZ. */
    distribution<double> param = run_irls(correct2, outputs2, w, link);
    
    /* Construct it from the results.  We want the label 1 to be 1 - label 0,
       so we just learn a single label 0. */

     /* If x, y and z are the params, then we create a matrix
       
        x  0  y  z
       -x  0 -y 1-z
     */

    //cerr << "param = " << param << endl;

    params.clear();

    distribution<float> expanded(ol, 0.0);
    expanded[0] = param[0];
    expanded[2] = param[1];
    expanded[3] = param[2];
    params.push_back(expanded);

    expanded[0] = -param[0];
    expanded[2] = -param[1];
    expanded[3] = 1.0 - param[2];  // add one to bias so we get 1-
    params.push_back(expanded);

    if (debug)
        for (unsigned l = 0;  l < nl;  ++l)
            cerr << "params for " << l << " = " << params[l] << endl;
}

namespace {

/** Small object to record the values of predictions once they're ready */
struct Write_Output {
    boost::multi_array<double, 2> & outputs;
    int nl;
    bool regression_problem;

    Write_Output(boost::multi_array<double, 2> & outputs, int nl,
                 bool regression_problem)
        : outputs(outputs), nl(nl), regression_problem(regression_problem)
    {
        if (!regression_problem)
            ExcAssertEqual(nl + 2, outputs.shape()[0]);
        else ExcAssertEqual(nl, 1);
    }

    void operator () (int example, const float * vals)
    {
        //cerr << "writing output: example " << example << " vals[0] "
        //     << vals[0] << " nl " << nl << " sz " << outputs.shape()[0]
        //     << "x" << outputs.shape()[1] << endl;
        ExcAssertLess(example, outputs.shape()[1]);

        if (!regression_problem) {
            float max_output = *vals;
            for (unsigned l = 0;  l < nl;  ++l) {
                outputs[l][example] = vals[l];
                max_output = std::max(max_output, vals[l]);
            }
            outputs[nl][example]     = max_output;  // maximum output
            outputs[nl + 1][example] = 1.0;     // bias term
        }
        else {
            outputs[0][example] = *vals;
            outputs[1][example] = 1.0;
        }
    }
};

} // file scope

void GLZ_Probabilizer::
train(const Training_Data & training_data,
      const Classifier_Impl & classifier,
      const Optimization_Info & opt_info,
      const distribution<float> & weights,
      int mode, const string & link_name)
{
    if (link_name != "") link = parse_link_function(link_name);

    //cerr << "training in mode " << mode << " link = " << link
    //     << " weights " << weights.size() << endl;

    const vector<Label> & labels
        = training_data.index().labels(classifier.predicted());
    
    size_t nl = training_data.label_count(classifier.predicted());
    if (classifier.label_count() != nl)
        throw Exception
            (format("classifier (%zd) and training data (%zd) have "
                    "different label counts", classifier.label_count(),
                    nl));

    size_t nx = training_data.example_count();

    if (weights.size() != nx)
        throw Exception(format("GLZ_Probabilizer::train(): passed %zd examples "
                               "but %zd weights", nx, weights.size()));
    
    if (nx == 0 && mode != 3)
        throw Exception("GLZ_Probabilizer::train(): passed 0 examples");

    /* In order to train this, we make a vector of the output of the
       classifier for each variable. */

    bool regression_problem = (nl == 1);

    size_t ol = regression_problem ? 2 : nl + 2;
    
    //bool debug = false;
    
    boost::multi_array<double, 2> outputs(boost::extents[ol][nx]);
    distribution<double> model(nx, 0.0);
    vector<distribution<double> > correct(nl, model);
    distribution<int> num_correct(nl);

    /* Go through the training examples one by one, and record the
       outputs. */
    classifier.predict(training_data,
                       Write_Output(outputs, nl, regression_problem),
                       &opt_info);

    //for (unsigned i = 0;  i < 10;  ++i)
    //    cerr << "outputs[" << 0 << "][" << i << "] = "
    //         << outputs[0][i] << endl;
    
    for (unsigned x = 0;  x < nx;  ++x) {
        if (regression_problem) {
            correct[0][x] = labels[x].value();
        }
        else {
            num_correct[labels[x]] += 1;
            for (unsigned l = 0;  l < nl;  ++l) {
                correct[l][x] = (float)(labels[x] == l);
            }
        }
    }

    //cerr << "num_correct = " << num_correct << endl;
    //cerr << "num_correct.total() = " << num_correct.total()
    //     << "  nx = " << nx << endl;

    //cerr << "before training: link = " << link << endl;
    //cerr << "mode = " << mode << endl;
    //cerr << "nl = " << nl << " regression_problem = " << regression_problem
    //     << endl;

    if (regression_problem) {
        switch (mode) {
        case 0:
        case 3:
            train_identity_regress(outputs, correct[0], weights, debug);
            break;

        case 1:
        case 2:
            train_glz_regress(outputs, correct[0], weights, debug);
            break;
            
        default:
            throw Exception(format("unknown GLZ probabilizer regression mode %d",
                                   mode));
        }
    }
    else {
        switch (mode) {
            
        case 0:
            train_mode0(outputs, correct, num_correct, weights, debug);
            break;
            
        case 1:
            train_mode1(outputs, correct, num_correct, weights, debug);
            break;
            
        case 2:
            train_mode2(outputs, correct, num_correct, weights, debug);
            break;
            
        case 3:
            train_mode3(outputs, correct, num_correct, weights, debug);
            break;
            
        case 4:
            train_mode4(outputs, correct, num_correct, weights, debug);
            break;

        case 5:
            train_mode5(outputs, correct, num_correct, weights, debug);
            break;
            
        default:
            throw Exception(format("unknown GLZ probabilizer mode %d", mode));
        }
    }

    //cerr << "trained probabilizer: " << print() << endl;

    //cerr << "after training: link = " << link << endl;
}

void GLZ_Probabilizer::
train(const Training_Data & training_data,
      const Classifier_Impl & classifier,
      const Optimization_Info & opt_info,
      int mode, const string & link_name)
{
    distribution<float> weights(training_data.example_count(), 1.0);
    train(training_data, classifier, opt_info, weights, mode, link_name);
}

void
GLZ_Probabilizer::
init(const Classifier_Impl & classifier,
     int mode, const std::string & link_name,
     const std::vector<float> & params)
{
    if (link_name != "") link = parse_link_function(link_name);

    size_t nl = classifier.label_count();
    bool regression_problem = (nl == 1);

    if (regression_problem) {
        throw Exception("GLZ_Probabilizer::init(): not impl for regression");

#if 0
        switch (mode) {
        case 0:
        case 3:
            train_identity_regress(outputs, correct[0], weights, debug);
            break;

        case 1:
        case 2:
            train_glz_regress(outputs, correct[0], weights, debug);
            break;
            
        default:
            throw Exception(format("unknown GLZ probabilizer regression mode %d",
                                   mode));
        }
#endif
    }
    else {
        switch (mode) {

        case 1: {
            if (params.size() != 3 * nl)
                throw Exception("GLZ_Probabilizer::init(): "
                                "mode 1 needs 3 values per label");

            this->link = link;
            this->params.clear();
            size_t ol = nl + 2;  // number of columns to create

            for (unsigned l = 0;  l < nl;  ++l) {
                distribution<float> expanded(ol, 0.0);
                expanded[l]      = params[0 + l * 3];
                expanded[ol - 2] = params[1 + l * 3];
                expanded[ol - 1] = params[2 + l * 3];
        
                this->params.push_back(expanded);
            }

            break;
        }

        case 2: {
            /* Perform the GLZ. */
            if (params.size() != 3)
                throw Exception("GLZ_Probabilizer::init(): "
                                "mode 2 needs 3 values");

            distribution<double> param(params.begin(), params.end());
            *this = construct_sparse(param, nl, link);

            for (unsigned l = 0;  l < nl;  ++l)
                cerr << "params for " << l << " = " << params[l] << endl;
            break;
        }

        default:
            throw Exception(format("GLZ_Probabilizer::init(): "
                                   "can't init mode %d", mode));
        }
    }
}

size_t GLZ_Probabilizer::domain() const
{
    return params.size();
}

size_t GLZ_Probabilizer::range() const
{
    if (params.empty()) return 0;
    else return params[0].size();
}

Output_Encoding GLZ_Probabilizer::output_encoding(Output_Encoding) const
{
    return OE_PROB;
}

bool GLZ_Probabilizer::positive() const
{
    for (unsigned l = 0;  l < params.size();  ++l)
        if (params[l][l] < 0.0) return false;
    return true;
}

std::string GLZ_Probabilizer::print() const
{
    string result = "GLZ_Probabilizer { link = " + ML::print(link);
    for (unsigned i = 0;  i < params.size();  ++i)
        result += format(" params[%d] = ", i) + ostream_format(params[i]);
    result += " }";
    return result;
}

namespace {

static const std::string GLZP_MAGIC="GLZ_PROBABILIZER";
static const compact_size_t GLZP_VERSION = 1;

} // file scope

void GLZ_Probabilizer::serialize(DB::Store_Writer & store) const
{
    store << GLZP_MAGIC << GLZP_VERSION << link << params;
}

void GLZ_Probabilizer::reconstitute(DB::Store_Reader & store)
{
    std::string magic;
    compact_size_t version;

    store >> magic >> version;
    if (magic != GLZP_MAGIC)
        throw Exception("Attempt to reconstitute \"" + magic
                        + "\" with glz probabilizer reconstitutor");
    if (version > GLZP_VERSION)
        throw Exception(format("Attemp to reconstitute glzp version %zd, only "
                               "<= %zd supported", version.size_,
                               GLZP_VERSION.size_));

    // link added in version 1
    if (version == 0) link = LOGIT;
    else store >> link;

    //cerr << "link = " << link << endl;
    
    store >> params;

    //cerr << "loaded probabilizer: params = " << params << " link = "
    //     << link << endl;
}

GLZ_Probabilizer::GLZ_Probabilizer(DB::Store_Reader & store)
{
    reconstitute(store);
}

GLZ_Probabilizer * GLZ_Probabilizer::make_copy() const
{
    return new GLZ_Probabilizer(*this);
}

DB::Store_Writer &
operator << (DB::Store_Writer & store, const GLZ_Probabilizer & prob)
{
    prob.serialize(store);
    return store;
}

DB::Store_Reader &
operator >> (DB::Store_Reader & store, GLZ_Probabilizer & prob)
{
    prob.reconstitute(store);
    return store;
}


/*****************************************************************************/
/* REGISTRATION                                                              */
/*****************************************************************************/

namespace {

//Register_Factory<Probabilizer, GLZ_Probabilizer> GLZ_REG1("GLZ_PROBABILIZER");
Register_Factory<Decoder_Impl, GLZ_Probabilizer> GLZ_REG2("GLZ_PROBABILIZER");

} // file scope


} // namespace ML


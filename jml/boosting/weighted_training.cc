/* weighted_training.cc
   Jeremy Barnes, 16 February 2005
   Copyright (c) 2005 Jeremy Barnes.  All rights reserved.
   $Source$

   Functions to build up training weights.
*/

#include "weighted_training.h"
#include "training_data.h"
#include "jml/utils/parse_context.h"
#include "training_index.h"
#include "jml/utils/floating_point.h"
#include "jml/math/xdiv.h"


using namespace std;


namespace ML {


Weight_Spec
get_weight_spec(const Training_Data & data,
                double beta,
                const Feature & feature,
                bool group_feature)
{
    Weight_Spec result;
    result.group_feature = group_feature;
    result.beta = beta;

    result.feature = feature;

    //cerr << "replications based upon feature " << feature << endl;
    /* Work out the replication factors based upon the values of a
       feature.  We assume they are discreet.  We get boosted or not
       depending upon how frequent the given value of the given
       feature is. */
    
    const Dataset_Index::Freqs & freqs = data.index().freqs(feature);

    double total_count = freqs.total();

    /* This map lets us look up a float value and tell us which
       number of replication it belongs to.  The 0 belongs to those
       for which the value is missing. */
    hash_map<float, int, float_hasher> order;
    int n = 1;
    for (Dataset_Index::Freqs::const_iterator it = freqs.begin();
         it != freqs.end();  ++it, ++n) {
        order[it->first] = n;
    }

    /* Put the count information in a flat array. */
    distribution<float> freqs2(n, 0.0);
    n = 1;
    for (Dataset_Index::Freqs::const_iterator it = freqs.begin();
         it != freqs.end();  ++it, ++n) freqs2[n] = it->second;
    
    /* Find an unused value to represent missing. */
    freqs2[0] = data.example_count() - total_count;

    //cerr << "freqs2 = " << freqs2 << endl;
        
    /* Work out the replication factors based upon the label freqs. */
    distribution<float> replications(n, 1.0);
    
    for (unsigned i = 0;  i < n;  ++i) {
        if (freqs2[i] == 0.0) replications[i] = 0.0;
        else replications[i] = pow(xdiv<double>(1.0, freqs2[i]), beta);
    }
    
    replications.normalize();  replications *= replications.size();

    //cerr << "replications = " << replications << endl;

    /* Now put it back. */
    n = 1;
    for (Dataset_Index::Freqs::const_iterator it = freqs.begin();
         it != freqs.end();  ++it, ++n)
        result.weights[it->first] = replications[n];
    result.missing_weight = replications[0];
    
    return result;
}

vector<Weight_Spec>
get_weight_spec(const Training_Data & data,
                const std::vector<float> & betas,
                const std::vector<Feature> & features,
                const std::vector<bool> & group_features)
{
    vector<Weight_Spec> result;
    if (betas.size() != features.size())
        throw Exception("get_weight_spec(): betas and features "
                        "don't have the same size");

    for (unsigned i = 0;  i < betas.size();  ++i)
        result.push_back(get_weight_spec(data, betas[i], features[i],
                                         group_features[i]));

    return result;
}

Weight_Spec
train_weight_spec(const Training_Data & data,
                  const Weight_Spec & spec)
{
    if (spec.type == Weight_Spec::BY_VALUE)
        return spec;
    else return get_weight_spec(data, spec.beta, spec.feature,
                                spec.group_feature);
}

std::vector<Weight_Spec>
train_weight_spec(const Training_Data & data,
                  const std::vector<Weight_Spec> & untrained)
{
    std::vector<Weight_Spec> result;

    for (unsigned i = 0;  i < untrained.size();  ++i)
        result.push_back(train_weight_spec(data, untrained[i]));

    return result;
}

distribution<float>
apply_weight_spec(const Training_Data & data, const Weight_Spec & spec_)
{
    // TODO: won't calculate proper weight for missing!

    Weight_Spec spec = spec_;

    if (spec.group_feature && spec_.type == Weight_Spec::BY_FREQUENCY) {
        /* Get the group counts specific to this group. */
        spec = get_weight_spec(data, spec.beta, spec.feature);
    }
    
    /* TODO: bug here if spec.missing_weight != 0.0?  Need to check that it
       works OK before removing this exception. */
    if (spec.missing_weight != 0.0)
        throw Exception("apply_weight_spec(): missing_weight != 0.0");

    unsigned content = IC_VALUE | IC_EXAMPLE;
    if (spec.type != Weight_Spec::BY_VALUE)
        content |= IC_COUNT;

    Joint_Index index = data.index().dist(spec.feature, BY_EXAMPLE, content);
    
    distribution<float> result(data.example_count(), spec.missing_weight);
    double total = 0.0;
    bool all_missing = true;
    
    for (Index_Iterator it = index.begin();  it != index.end();  ++it) {

        if (spec.type == Weight_Spec::BY_VALUE) {
            float value = it->value();
            if (value == 0.0) {
#if 0 // not a problem after all
                cerr << "warning: using zero value as weight; changed to 1.0 "
                     << "for feature " << spec.feature << " in example "
                     << it->example()
                     << endl;
                value = 1.0;
#endif // not a problem
            }
            else if (!finite(value))
                throw Exception("apply_weight_spec: non-finite value "
                                + ostream_format(value) + " in weight feature "
                                + data.feature_space()->print(spec.feature)
                                + " in example "
                                + ostream_format(it->example()));
            result[it->example()] = value;
            total += value;
        }
        else {
            float amt;
            if (!it->missing()) {
                map<float, float>::const_iterator pos
                    = spec.weights.find(it->value());
                if (pos == spec.weights.end())
                    amt = spec.missing_weight / it->example_counts();
                else {
                    amt = pos->second / it->example_counts();
                    all_missing = false;
                }
            }
            else amt = spec.missing_weight / it->example_counts();

            result[it->example()] += amt;
            total += amt;
        }
    }

    //cerr << "total = " << total << " data.example_count = "
    //     << data.example_count() << " spec.type = " << spec.type
    //     << " all_missing = " << all_missing << endl;

    if (total == 0.0
        && data.example_count() > 0
        && spec.type == Weight_Spec::BY_FREQUENCY
        && all_missing) {

        /* We get here for small datasets where there is only value A in the
           training set and only value B in the validation set.  We just
           use equal weighting. */

        cerr << "warning: weights were zero due to all missing values; using "
             << "uniform weights" << endl;

        std::fill(result.begin(), result.end(), 1.0);
        total = result.size();
    }
    else if (total == 0.0 && data.example_count() > 0) {
        cerr << "spec: (" << data.feature_space()->print(spec.feature)
             << ", " << spec.beta << ", " << spec.group_feature << ", "
             << (spec.type == Weight_Spec::BY_FREQUENCY
                 ? "BY_FREQUENCY" : "BY_VALUE")
             << ")"
             << endl;

        if (spec.type != Weight_Spec::BY_VALUE) {
            for (map<float, float>::const_iterator it = spec.weights.begin();
                 it != spec.weights.end();  ++it) {
                cerr << "  value "
                     << data.feature_space()->print(spec.feature, it->first)
                     << " has weight " << it->second << endl;
            }
            cerr << "  missing weight = " << spec.missing_weight << endl;
        }

        for (Index_Iterator it = index.begin();  it != index.end();  ++it) {
            cerr << "example " << it->example() << " value "
                 << it->value();
            float value;
            if (spec.type == Weight_Spec::BY_VALUE)
                value = it->value();
            else {
                if (!it->missing()) {
                    map<float, float>::const_iterator pos
                        = spec.weights.find(it->value());
                    if (pos == spec.weights.end()) value = 0.0;
                    else value = pos->second / it->example_counts();
                }
                else value = spec.missing_weight / it->example_counts();
            }
            cerr << " weight = " << value << endl;
        }

        //cerr << "dataset: " << endl;
        //data.dump(cerr);

        throw Exception("apply_weight_spec: no values had any weight");
    }
    
    result /= total;

#if 0
    cerr << "apply_weight_spec (" << data.feature_space()->print(spec.feature)
         << ", " << spec.beta << ", " << spec.group_feature << ") returned ";
    if (result.size() > 100)
        cerr << (distribution<float>(result.begin(), result.begin() + 100)
                 * result.size())
             << endl;
    else cerr << (result * result.size()) << endl;
#endif    

    return result;
}

distribution<float>
apply_weight_spec(const Training_Data & data,
                  const std::vector<Weight_Spec> & specs)
{
    distribution<float> result(data.example_count(), 1.0);

    for (unsigned i = 0;  i < specs.size();  ++i)
        result *= apply_weight_spec(data, specs[i]);

    result.normalize();

    return result;
}

boost::multi_array<float, 2>
expand_weights(const Training_Data & data,
               const distribution<float> & weights,
               const Feature & predicted)
{
    if (weights.size() != data.example_count())
        throw Exception("expand_weights(): weights and data "
                        "sizes don't match");
    
    int nl = data.label_count(predicted);
    
    boost::multi_array<float, 2> result
        (boost::extents[data.example_count()][nl]);
    double recip = 1.0 / (nl * weights.total());

    for (unsigned x = 0;  x < data.example_count();  ++x)
        for (unsigned l = 0;  l < nl;  ++l)
            result[x][l] = weights[x] * recip;

    return result;
}

std::vector<Weight_Spec>
parse_weight_spec(const Feature_Space & fs,
                  string equalize_name,
                  float default_beta,
                  string weight_spec,
                  Feature group_feature)
{
    //if (weight_spec != "" && beta != 0.0)
    //    throw Exception("both a weight spec and equalize beta were specified; "
    //                    "use only one!");
    
    vector<Weight_Spec> result;

    if (weight_spec != "") {
        Parse_Context context("weight-spec option",
                              &*weight_spec.begin(), &*weight_spec.end());

        while (context) {
            Weight_Spec spec;

            string feature_name
                = context.expect_text(",/", "expected feature name");
            //cerr << "got feature name " << feature_name << endl;
            try {
                fs.parse(feature_name, spec.feature);
            }
            catch (const std::exception & exc) {
                context.exception("Parsing weight spec: "
                                  + string(exc.what()));
            }

            /* Guess the type, based upon whether its a real variable or
               not. */
            spec.type
                = fs.info(spec.feature).value_count() == 0
                ? Weight_Spec::BY_VALUE : Weight_Spec::BY_FREQUENCY;
            
            /* Allow override of type if wrongly guessed */
            if (context && context.match_literal("/V"))
                spec.type = Weight_Spec::BY_VALUE;
            else if (context && context.match_literal("/F"))
                spec.type = Weight_Spec::BY_FREQUENCY;
            
            if (spec.type == Weight_Spec::BY_FREQUENCY) {
                if (context && context.match_literal('/')) {
                    spec.beta = context.expect_float();
                }
                else spec.beta = default_beta;
                
                if (context.match_literal('/')) {
                    context.expect_literal('G');
                    spec.group_feature = true;
                }
                else spec.group_feature = (spec.feature == group_feature);

                if (context) context.match_literal(',');
            }

            result.push_back(spec);
        }
    }
    else {
        Weight_Spec spec;
        fs.parse(equalize_name, spec.feature);
        spec.beta = default_beta;
        spec.group_feature = false;
        
        result.push_back(spec);
    }
    
    return result;
}

} // namespace ML


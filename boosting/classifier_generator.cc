/* classifier_generator.cc
   Jeremy Barnes, 15 March 2006
   Copyright (c) 2006 Jeremy Barnes  All rights reserved.
   $Source$

*/

#include "classifier_generator.h"
#include "registry.h"
#include "jml/utils/filter_streams.h"


using namespace std;


namespace ML {


/*****************************************************************************/
/* CLASSIFIER_GENERATOR                                                      */
/*****************************************************************************/

Classifier_Generator::
~Classifier_Generator()
{
}

void
Classifier_Generator::
init(std::shared_ptr<const Feature_Space> fs, Feature predicted)
{
    this->feature_space = fs;
    this->predicted = predicted;
    nl = feature_space->info(predicted).value_count();
}

void
Classifier_Generator::
configure(const Configuration & config)
{
    config.find(verbosity, "verbosity");
    config.find(profile, "profile");
    config.find(validate, "validate");
}

void
Classifier_Generator::
defaults()
{
    verbosity = 2;
    profile = false;
    validate = false;
}

Config_Options
Classifier_Generator::
options() const
{
    Config_Options result;
    result
        .add("verbosity", verbosity, "0-5",
             "verbosity of information from training")
        .add("profile", profile, "whether or not to profile")
        .add("validate", validate, "perform expensive internal validation");

    return result;
}

std::shared_ptr<Classifier_Impl>
Classifier_Generator::
generate(Thread_Context & context,
         const Training_Data & training_data,
         const Training_Data & validation_data,
         const std::vector<Feature> & features,
         int recursion) const
{
    if (recursion > 3)
        throw Exception("Classifier_Generator::generate(): recursion! "
                        "at least one generate method must be overridden");

    distribution<float> training_weights(training_data.example_count(), 1.0);
    distribution<float> validation_weights(validation_data.example_count(), 1.0);
    return generate(context, training_data, validation_data, training_weights,
                    validation_weights, features, recursion + 1);
}

std::shared_ptr<Classifier_Impl>
Classifier_Generator::
generate(Thread_Context & context,
         const Training_Data & training_data,
         const Training_Data & validation_data,
         const distribution<float> & training_weights,
         const distribution<float> & validation_weights,
         const std::vector<Feature> & features,
         int recursion) const
{
    if (recursion > 3)
        throw Exception("Classifier_Generator::generate(): recursion! "
                        "at least one generate method must be overridden");

    return generate(context, training_data, training_weights, features,
                    recursion + 1);
}

std::shared_ptr<Classifier_Impl>
Classifier_Generator::
generate(Thread_Context & context,
         const Training_Data & training_data,
         const distribution<float> & ex_weights,
         const std::vector<Feature> & features,
         int recursion) const
{
    if (recursion > 3)
        throw Exception("Classifier_Generator::generate(): recursion! "
                        "at least one generate method must be overridden");

    size_t nx = training_data.example_count();

    /* Expand the weights */
    boost::multi_array<float, 2> weights(boost::extents[nx][nl]);

    if (ex_weights.empty())
        std::fill(weights.data(), weights.data() + nx * nl, 1.0 / nl * nx);
    else {
        double tot = ex_weights.total();
        if (ex_weights.size() != nx)
            throw Exception("Classifier_Generator::generate(): "
                            "wrong sized example weights");
        else if (abs(tot) < 1e-10)
            throw Exception("Classifier_Generator::generate(): "
                            "zero or nearly zero example weights total");
        
        float norm = 1.0 / (ex_weights.total() * nl);

        for (unsigned x = 0;  x < nx;  ++x) {
            if ((ex_weights[x] < 0.0) || ex_weights[x] > 1e10)
                throw Exception("Classifier_Generator::generate(): weight "
                                "out of range");
            double val = ex_weights[x] * norm;
            std::fill(&weights[x][0], &weights[x][0] + nl, val);
        }
    }

    float Z = 0.0;
    return generate(context, training_data, weights, features, Z,
                    recursion + 1);
}

std::shared_ptr<Classifier_Impl>
Classifier_Generator::
generate(Thread_Context & context,
         const Training_Data & training_data,
         const boost::multi_array<float, 2> & weights,
         const std::vector<Feature> & features,
         float & Z,
         int recursion) const
{
    if (recursion > 3)
        throw Exception("Classifier_Generator::generate(): recursion! "
                        "at least one generate method must be overridden");

    size_t nx = training_data.example_count();

    if ((weights.shape()[0] != nx)
        || (weights.shape()[1] != nl && nl != 2 && weights.shape()[1] != 1))
        throw Exception("Classifier_Generator::generate(): "
                        "weights array has the wrong dimensions"
                        + format("(%dx%x), should be (%dx%d)",
                                 (int)weights.shape()[0],
                                 (int)weights.shape()[1],
                                 (int)nx,
                                 (int)nl));
    
    /* Generate some normal example weights as the average of those here. */
    distribution<float> ex_weights(nx);

    double total = 0.0;
    for (unsigned x = 0;  x < nx;  ++x) {
        double ex_total = 0.0;
        for (unsigned l = 0;  l < weights.shape()[1];  ++l)
            ex_total += weights[x][l];
        if (nl == 2 && weights.shape()[1] == 1)
            ex_total *= 2.0;
        ex_weights[x] = ex_total;
        total += ex_total;
    }
    
    ex_weights *= nx / total;

    return generate(context, training_data, ex_weights, features,
                    recursion + 1);
}

std::ostream &
Classifier_Generator::
log(const std::string & module, int level) const
{
    //cerr << "level " << level << " verbosity " << verbosity << endl;

    static filter_ostream cnull("");

    if (level <= verbosity)
        return cerr;
    else return cnull;
}

std::string
Classifier_Generator::
type() const
{
    return demangle(typeid(*this).name());
}


/*****************************************************************************/
/* FACTORIES                                                                 */
/*****************************************************************************/

boost::tuple<std::string, std::string>
get_type(const std::string & name, const Configuration & config)
{
    std::string name2 = name;
    if (config.count(name) && config[name] != "") {
        // allow "default=lso1" to work
        name2 = config[name];
    }
    
    string type_key = (name2 == "" ? string("type") : name2 + ".type");

    if (!config.count(type_key, true))
        throw Exception("key " + type_key + " not found in config ("
                        "initial was " + name + ".type)");
    
    std::string key = config.find_key(type_key, true);

    if (config.prefix() != ""
        && key.find(config.prefix() + '.') != string::npos)
        key = string(key, config.prefix().size() + 1);

    name2 = key;
    name2.resize(std::max<int>(0, name2.size() - 5));  // remove the ".type"
    
    return boost::make_tuple(config[key], name2);
}

std::shared_ptr<Classifier_Generator>
get_trainer(const std::string & name, const Configuration & config)
{
    std::string type, name2;
    boost::tie(type, name2) = get_type(name, config);
    
    if (type == "") {
        //cerr << config << endl;
        throw Exception("Object with name \"" + name + "\" has no type "
                        "in configuration file with prefix "
                        + config.prefix());
    }
    
    std::shared_ptr<Classifier_Generator> result
        = Registry<Classifier_Generator>::singleton().create(type);

    Configuration config2(config, name, Configuration::PREFIX_APPEND);
    result->configure(config2);

    return result;
}

} // namespace ML


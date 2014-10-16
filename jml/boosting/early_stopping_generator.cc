/* early_stopping_generator.cc
   Jeremy Barnes, 17 March 2006
   Copyright (c) 2006 Jeremy Barnes  All rights reserved.
   $Source$

   Sets up separate training and validation sets for early stopping.
*/

#include "early_stopping_generator.h"
#include "jml/arch/demangle.h"
#include "jml/utils/sgi_numeric.h"


using namespace std;


namespace ML {


/*****************************************************************************/
/* EARLY_STOPPING_GENERATOR                                                  */
/*****************************************************************************/

Early_Stopping_Generator::
~Early_Stopping_Generator()
{
}

void
Early_Stopping_Generator::
configure(const Configuration & config)
{
    Classifier_Generator::configure(config);
    config.find(validate_split, "validate_split");
}

void
Early_Stopping_Generator::
defaults()
{
    Classifier_Generator::defaults();
    validate_split = 0.30;
}

Config_Options
Early_Stopping_Generator::
options() const
{
    Config_Options result = Classifier_Generator::options();
    result
        .add("validation_split", validate_split, "0<N<=1",
             "how much of training data to hold off as validation data");
    
    return result;
}

std::shared_ptr<Classifier_Impl>
Early_Stopping_Generator::
generate(Thread_Context & context,
         const Training_Data & training_data,
         const distribution<float> & ex_weights,
         const std::vector<Feature> & features,
         int recursion) const
{
    if (recursion > 10)
        throw Exception("Early_Stopping_Generator::generate(): recursion");

    if (validate_split <= 0.0 || validate_split >= 1.0)
        throw Exception("invalid validate split value");

    float train_prop = 1.0 - validate_split;

    int nx = ex_weights.size();

    Thread_Context::RNG_Type rng = context.rng();

    distribution<float> in_training(nx);
    vector<int> tr_ex_nums(nx);
    std::iota(tr_ex_nums.begin(), tr_ex_nums.end(), 0);
    std::random_shuffle(tr_ex_nums.begin(), tr_ex_nums.end(), rng);
    for (unsigned i = 0;  i < nx * train_prop;  ++i)
        in_training[tr_ex_nums[i]] = 1.0;
    distribution<float> not_training(nx, 1.0);
    not_training -= in_training;
    
    distribution<float> example_weights(nx);
    
    /* Generate our example weights. */
    for (unsigned i = 0;  i < nx;  ++i)
        example_weights[rng(nx)] += 1.0;
    
    distribution<float> training_weights
        = in_training * example_weights * ex_weights;

    //cerr << "in_training.total() = " << in_training.total() << endl;
    //cerr << "example_weights.total() = " << example_weights.total()
    //     << endl;
    //cerr << "ex_weights.total() = " << ex_weights.total() << endl;
        

    if (training_weights.total() == 0.0)
        throw Exception("training weights were empty");

    training_weights.normalize();
    
    distribution<float> validate_weights
        = not_training * example_weights * ex_weights;

    if (validate_weights.total() == 0.0)
        throw Exception("validate weights were empty");

    validate_weights.normalize();

    return generate(context, training_data, training_data,
                    training_weights, validate_weights,
                    features, recursion + 1);
}

} // namespace ML

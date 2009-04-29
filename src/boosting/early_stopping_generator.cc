/* early_stopping_generator.cc
   Jeremy Barnes, 17 March 2006
   Copyright (c) 2006 Jeremy Barnes  All rights reserved.
   $Source$

   Sets up separate training and validation sets for early stopping.
*/

#include "early_stopping_generator.h"

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

boost::shared_ptr<Classifier_Impl>
Early_Stopping_Generator::
generate(Thread_Context & context,
         const Training_Data & training_data,
         const distribution<float> & ex_weights,
         const std::vector<Feature> & features,
         int recursion) const
{
    throw Exception("Early_Stopping_Generator::generate(): not implemented");
}

} // namespace ML

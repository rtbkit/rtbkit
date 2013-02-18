/* null_classifier_generator.cc
   Jeremy Barnes, 15 March 2006
   Copyright (c) 2006 Jeremy Barnes  All rights reserved.
   $Source$

   Generator for null_classifiers.
*/

#include "null_classifier_generator.h"
#include "registry.h"
#include "jml/utils/smart_ptr_utils.h"


using namespace std;


namespace ML {

/*****************************************************************************/
/* NULL_CLASSIFIER_GENERATOR                                                 */
/*****************************************************************************/

Null_Classifier_Generator::
Null_Classifier_Generator()
{
    defaults();
}

Null_Classifier_Generator::~Null_Classifier_Generator()
{
}

void
Null_Classifier_Generator::
configure(const Configuration & config)
{
    Classifier_Generator::configure(config);
}

void
Null_Classifier_Generator::
defaults()
{
    Classifier_Generator::defaults();
}

Config_Options
Null_Classifier_Generator::
options() const
{
    Config_Options result = Classifier_Generator::options();
    return result;
}

void
Null_Classifier_Generator::
init(std::shared_ptr<const Feature_Space> fs, Feature predicted)
{
    Classifier_Generator::init(fs, predicted);
    model = Null_Classifier(fs, predicted);
}

std::shared_ptr<Classifier_Impl>
Null_Classifier_Generator::
generate(Thread_Context & context,
         const Training_Data & training_set,
         const Training_Data & validation_set,
         const distribution<float> & training_ex_weights,
         const distribution<float> & validate_ex_weights,
         const std::vector<Feature> & features, int) const
{
    Null_Classifier current(model);
    return make_sp(current.make_copy());
}


/*****************************************************************************/
/* REGISTRATION                                                              */
/*****************************************************************************/

namespace {

Register_Factory<Classifier_Generator, Null_Classifier_Generator>
    NULL_CLASSIFIER_REGISTER("null");

} // file scope

} // namespace ML

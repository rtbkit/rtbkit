/* weight_updating_generator.h                                     -*- C++ -*-
   Jeremy Barnes, 10 May 2006
   Copyright (c) 2006 Jeremy Barnes.  All rights reserved.

   Generator that updates a weights matrix.  Inherits from an early
   stopping generator.
*/

#ifndef __boosting__weight_updating_generator_h__
#define __boosting__weight_updating_generator_h__

#include "early_stopping_generator.h"

namespace ML {


/*****************************************************************************/
/* WEIGHT_UPDATING_GENERATOR                                                 */
/*****************************************************************************/

/** Class to generate a classifier, including tracking a weights matrix
    for the examples.  This is used by boosting and boosted stumps.
*/

class Weight_Updating_Generator : public Early_Stopping_Generator {
public:
    virtual ~Weight_Updating_Generator() {}

    virtual std::shared_ptr<Classifier_Impl>
    generate_and_update(Thread_Context & context,
                        const Training_Data & training_data,
                        boost::multi_array<float, 2> & weights,
                        const std::vector<Feature> & features) const = 0;
};


} // namespace ML



#endif /* __boosting__weight_updating_generator_h__ */

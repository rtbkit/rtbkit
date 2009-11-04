/* dnae.h                                                          -*- C++ -*-
   Jeremy Barnes, 4 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Denoising autoencoder training code.
*/

#ifndef __jml__neural__dnae_h__
#define __jml__neural__dnae_h__


#include "twoway_layer.h"
#include "layer_stack.h"
#include "stats/distribution.h"
#include <vector>


namespace ML {

void
train_dnae(Layer_Stack<Twoway_Layer> & stack,
           const std::vector<distribution<float> > & training_data,
           const std::vector<distribution<float> > & testing_data,
           const Configuration & config,
           Thread_Context & thread_context);


std::pair<double, double>
test_dnae(Layer_Stack<Twoway_Layer> & stack,
          const std::vector<distribution<float> > & data,
          float prob_cleared,
          Thread_Context & thread_context,
          int verbosity);

} // namespace ML

#endif /* __jml__neural__dnae_h__ */

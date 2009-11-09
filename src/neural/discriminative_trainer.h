/* discriminative_trainer.h                                        -*- C++ -*-
   Jeremy Barnes, 9 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Trainer class for discriminative training of neural networks via
   backpropagation.
*/

#ifndef __jml__neural__discriminative_trainer_h__
#define __jml__neural__discriminative_trainer_h__

#include "loss_function.h"


namespace ML {

struct Discriminative_Trainer {

    Layer * layer;
    const std::vector<distribution<float> > & data;
    const std::vector<distribution<float> > & labels;
    const std::vector<float> & labels;



};

struct Discriminative_Tester {

    Layer * layer;
    const std::vector<distribution<float> > & inputs;
    const std::vector<distribution<float> > & data;
    const std::vector<distribution<float> > & labels;
};

} // namespace JML


#endif /* __jml__neural__discriminative_trainer_h__ */

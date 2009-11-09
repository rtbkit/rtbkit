/* discriminative_trainer.h                                        -*- C++ -*-
   Jeremy Barnes, 9 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Trainer class for discriminative training of neural networks via
   backpropagation.
*/

#ifndef __jml__neural__discriminative_trainer_h__
#define __jml__neural__discriminative_trainer_h__

#include "loss_function.h"
#include "layer.h"
#include "utils/configuration.h"

namespace ML {


/*****************************************************************************/
/* DISCRIMINATIVE_TRAINER                                                    */
/*****************************************************************************/

struct Discriminative_Trainer {
public:

    std::pair<double, double>
    train_example(const distribution<float> & data,
                  const distribution<float> & label,
                  Parameters_Copy<double> & updates) const;

    std::pair<double, double>
    train_example(const distribution<float> & data,
                  float label,
                  Parameters_Copy<double> & updates) const;

    std::pair<double, double>
    train_iter(const std::vector<distribution<float> > & data,
               const std::vector<float> & labels,
               Thread_Context & thread_context,
               int minibatch_size, float learning_rate,
               int verbosity,
               float sample_proportion,
               bool randomize_order) const;

    std::pair<double, double>
    train(const std::vector<distribution<float> > & training_data,
          const std::vector<float> & training_labels,
          const std::vector<distribution<float> > & testing_data,
          const std::vector<float> & testing_labels,
          const Configuration & config,
          ML::Thread_Context & thread_context) const;

    std::pair<double, double>
    test(const std::vector<distribution<float> > & data,
         const std::vector<float> & labels,
         ML::Thread_Context & thread_context,
         int verbosity) const;

    Layer * layer;
    const std::vector<distribution<float> > * data;
    const std::vector<distribution<float> > * labels1;
    const std::vector<float> * labels2;
};


} // namespace JML


#endif /* __jml__neural__discriminative_trainer_h__ */

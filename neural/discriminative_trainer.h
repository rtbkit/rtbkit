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
#include "jml/utils/configuration.h"
#include "output_encoder.h"

namespace ML {


/*****************************************************************************/
/* DISCRIMINATIVE_TRAINER                                                    */
/*****************************************************************************/

struct Discriminative_Trainer {
public:

    std::pair<double, double>
    train_example(const distribution<float> & data,
                  const distribution<float> & label,
                  Parameters_Copy<double> & updates,
                  float weight = 1.0) const;
    
    std::pair<double, double>
    train_example(const float * data,
                  Label label,
                  Parameters_Copy<double> & updates,
                  const Output_Encoder & encoder,
                  float weight = 1.0) const;

    std::pair<double, double>
    train_example(const distribution<float> & data,
                  Label label,
                  Parameters_Copy<double> & updates,
                  const Output_Encoder & encoder,
                  float weight = 1.0) const;

    std::pair<double, double>
    train_example(const float * data,
                  const float * label,
                  Parameters_Copy<double> & updates,
                  float weight = 1.0) const;

    std::pair<double, double>
    train_iter(const std::vector<distribution<float> > & data,
               const std::vector<Label> & labels,
               const std::vector<float> & weights,
               const Output_Encoder & encoder,
               Thread_Context & thread_context,
               int minibatch_size,
               float learning_rate,
               int verbosity,
               float sample_proportion,
               bool randomize_order) const;

    std::pair<double, double>
    train_iter(const std::vector<const float *> & data,
               const std::vector<Label> & labels,
               const std::vector<float> & weights,
               const Output_Encoder & encoder,
               Thread_Context & thread_context,
               int minibatch_size,
               float learning_rate,
               int verbosity,
               float sample_proportion,
               bool randomize_order) const;

    std::pair<double, double>
    train(const std::vector<distribution<float> > & training_data,
          const std::vector<Label> & training_labels,
          const std::vector<float> & training_weights,
          const std::vector<distribution<float> > & testing_data,
          const std::vector<Label> & testing_labels,
          const std::vector<float> & testing_weights,
          const Configuration & config,
          ML::Thread_Context & thread_context) const;

    std::pair<double, double>
    test(const std::vector<const float *> & data,
         const std::vector<Label> & labels,
         const std::vector<float> & weights,
         const Output_Encoder & encoder,
         ML::Thread_Context & thread_context,
         int verbosity) const;

    std::pair<double, double>
    test(const std::vector<distribution<float> > & data,
         const std::vector<Label> & labels,
         const std::vector<float> & weights,
         const Output_Encoder & encoder,
         ML::Thread_Context & thread_context,
         int verbosity) const;

    Layer * layer;
};


} // namespace JML


#endif /* __jml__neural__discriminative_trainer_h__ */

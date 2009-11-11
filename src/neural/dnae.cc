/* dnae.cc
   Jeremy Barnes, 4 November 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Denoising Auto Encoder functions.
*/

#include "dnae.h"
#include <boost/progress.hpp>
#include "boosting/worker_task.h"
#include <boost/tuple/tuple.hpp>
#include "arch/threads.h"
#include "utils/guard.h"
#include "utils/configuration.h"
#include <boost/assign/list_of.hpp>
#include "arch/timers.h"
#include <boost/bind.hpp>
#include "auto_encoder_stack.h"


using namespace std;


namespace ML {

typedef double CFloat;
typedef float LFloat;

std::pair<double, double>
Twoway_Layer::
train_iter(const vector<distribution<float> > & data,
           float prob_cleared,
           Thread_Context & thread_context,
           int minibatch_size, float learning_rate,
           int verbosity,
           float sample_proportion,
           bool randomize_order)
{
}


pair<double, double>
test_and_update(const vector<distribution<float> > & data_in,
                vector<distribution<float> > & data_out,
                float prob_cleared,
                Thread_Context & thread_context,
                int verbosity) const
{
}



pair<double, double>
test_dnae(const LayerStackT<Twoway_Layer> & layers,
          const vector<distribution<float> > & data,
          float prob_cleared,
          Thread_Context & thread_context,
          int verbosity) const
{
}

void
train(Auto_Encoder_Stack & stack,
      const std::vector<distribution<float> > & training_data,
      const std::vector<distribution<float> > & testing_data,
      const Configuration & config,
      Thread_Context & thread_context)
{



} // namespace ML

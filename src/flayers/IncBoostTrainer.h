/*                                                                 -*- C++ -*- */
#ifndef INC_BOOST_TRAINER_H
#define INC_BOOST_TRAINER_H

#include "Trainer.h"
///  \class IncBoostTrainer IncBoostTrainerTrainer.h "Utils/IncBoostTrainerTrainer.h"
///  \brief IncBoostTrainer to speedup Incremental unchange parameters (usefull in incremental Architecture).
///
/// class IncBoostTrainer
/// Idea :If you do not backpropagate on some Layers or Connectors,
///       you will lost precious time during fprop
///       to overcome this problem you should use BoostTrainer.
///


class IncBoostTrainer:public Trainer
{
 public:
  DataSet* bds;///boost dataset
  IncBoostTrainer(real *means, real *vars):Trainer(means,vars){bds=NULL;};
  void freeMem(){if(bds)delete(bds);bds=NULL;};
  void cleanBoostDataSetAndReactivate();
  void createBoostDataSet(DataSet *dset);
  void updateBoostDataSet(DataSet *dset);
  void newBoostDataSet(DataSet *dset){createBoostDataSet(dset);updateBoostDataSet(dset);};
  virtual real trainOrTest(DataSet *dset,bool do_train,uint batch_size);
  ~IncBoostTrainer(){freeMem();};
};
#endif

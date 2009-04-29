/*                                                                 -*- C++ -*- */
#ifndef TRAINER_H
#define TRAINER_H

#include "fLayersGeneral.h"
#include "Topology.h"
#include "fDataSet.h"
// WARNING: global topo variable; useful in Layers class
extern Topology *gtopo;

///  \class Trainer Trainer.h "Core/Trainer.h"
///  \brief Trainer represent the Trainer.
///
/// class Trainer (second Core class of this Learning library)
/// Idea :The Trainer train over examples contained into a DataSet and optimize
/// the parameters of the Topology.
/// author: Francis Pieraut (begin in)july 2002 pierautf@iro.umontreal.ca

class Trainer
{
 public:
  Trainer(real *means, real *vars, bool stdout_on=true);
  //we don't want to include any factory !!
  void set(Topology* topo)
    {this->topo=topo;gtopo=topo;};
  // means and vars of inputs
  real* means;
  real* vars;
  Topology *topo;
  real cost;
  real error_classif;
  ///print train error on stdout
  bool stdout_on;
  // options
  real stop_train_time;
  real stop_train_error;
  ///start timer every time you call train...
  bool start_timer;
  ///if true->coreTrainFct return train "on fly" error instead of the real test error
  bool test_on_fly;
  std::ofstream out;
  std::string output_filename;
  ///iteration #
  uint iter_i;
  real* normalize(real *inputs, bool normalize=true);
  real fprop(real *inputs);
  // main fct
  virtual real trainOrTest(DataSet *dset,bool do_train,uint batch_size);
  real test(DataSet *dset){return trainOrTest(dset,false,dset->iter_size_);};
  uint updateSamplingDistribution(DataSet *dset);
  real train(DataSet *dset,DataSet* test_set,int n_epochs,int batch_size);
  real train(DataSet *dset,int n_epochs,int batch_size){return train(dset,NULL,n_epochs,batch_size);};
  real train(DataSet *dset,DataSet* test_set,int n_epochs){return train(dset,test_set,n_epochs,1);};
  real train(DataSet *dset,int n_epochs){return train(dset,NULL,n_epochs,1);};
  real testCurrentExample(){return topo->fprop();};
  virtual void initEpoch(int i){topo->initEpoch(i);};
  ///this fct is usefull when you want to overwrite the core train fct part !!
  /// WARNING: take care of test_on_fly parameter
  virtual real coreTrainFct(DataSet *train_set,uint batch_size)
    {real ret=trainOrTest(train_set,true,batch_size);if(test_on_fly)return ret;else return trainOrTest(train_set,false,batch_size);};
  /// output file stuffs
  void openOutputFile(std::string name){output_filename=name;out.open((name+".out").c_str());};
  ///save and load are define in TrainerUtil.h
  virtual void saveParams(std::ostream & out){topo->saveParams(out);};
  virtual void loadParams(std::istream & in){topo->loadParams(in);};
  // train stop fct (optional)
  void stopTrainAtTime(double time){if (time==0) stop_train_time=FLT_MAX;else stop_train_time=time;};
  void stopTrainAtError(real error){stop_train_error=error;};
  //util
  void displayfbprop(bool all=false);
  Connector* getConnector(std::string name){Connector* con=topo->getConnector(name);if(!con){FERR("unknown connector: "+name);return NULL;}else return con;};
  Layer* getLayer(std::string name){Layer* layer=topo->getLayer(name);if(!layer){FERR("unknown layer: "+name);return NULL;}else return layer;};
  virtual ~Trainer();
};
#endif

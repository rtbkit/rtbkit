#include "ArgvParser.h"
#include "fDataSet.h"

#ifndef PREDEFARGVPARSER
#define PREDEFARGVPARSER

// Definition of usual parameters : data_set,exp,inctopo,optOneHidden

//data_set
extern std::string data_set_description;
extern opt opts_data_set[];
extern noargopt noargopts_data_set[];
//neural network settings
extern std::string nn_description;
extern opt opts_nn[];
extern noargopt noargopts_nn_set[];
//optimal bprop settings
extern std::string optbprop_description;
extern opt opts_optbprop[];
extern noargopt noargopts_optbprop_set[];
//topology incremental
extern std::string inc_topo_description;
extern opt opts_inc_topo[];
//Booster
extern std::string boost_description;
extern opt boost_Opt[];
//EveryNEpochs
extern std::string every_description;
extern opt every_Opt[];

class NNArgvParser:public ArgvParser{
  typedef ArgvParser inherited;
 public:
  NNArgvParser()
    :ArgvParser("NN_SETTINGS",opts_nn,noargopts_nn_set,nn_description)
    {};
};

class BpropTypeArgvParser:public ArgvParser{
  typedef ArgvParser inherited;
 public:
  BpropTypeArgvParser()
    :ArgvParser("OPT_BPROP",opts_optbprop,noargopts_optbprop_set,optbprop_description)
    {};
};

class IncTopoArgvParser:public ArgvParser{
  typedef ArgvParser inherited;
 public:
  IncTopoArgvParser()
    :ArgvParser("INC_TOPO",opts_inc_topo,NULL,inc_topo_description)
    {};
};

class BoostArgvParser:public ArgvParser{
  typedef ArgvParser inherited;
 public:
  BoostArgvParser()
    :ArgvParser("BOOSTER",boost_Opt,NULL,boost_description)
    {};
};

class EveryNEpochsArgvParser:public ArgvParser{
  typedef ArgvParser inherited;
 public:
  EveryNEpochsArgvParser()
    :ArgvParser("EVERYNEPOCHS",every_Opt,NULL,every_description)
    {};
};

class DataSetArgvParser:public ArgvParser{
  typedef ArgvParser inherited;
 public:
  DataSet *train_set_;
  DataSet *valid_set_;
  DataSet *test_set_;
  real *data_;
  real *targets_;
  real *means;//means of train_set
  real *vars;//variance of train_set
  std::string db_name_;
  DataSetArgvParser()
    :ArgvParser("DATASET",opts_data_set,NULL/*noargopts_data_set*/,data_set_description)
    {train_set_=NULL;valid_set_=NULL;test_set_=NULL;};
  virtual void CoherenceParamsValuesChecker();
  virtual void parse(int argc,char* argv[]);
  std::string getStringDescriptor(bool short_descr=true,std::ofstream *output=NULL);
  void splitTrainValidTest();
  void printData();
  void save(std::string filename);
  bool loadData(std::string filename);
  void normalizeDataInputs();
  void normalizeTarget();
  //void Coherence(NNArgvParser* nn_settings);
  ~DataSetArgvParser();
};
#endif


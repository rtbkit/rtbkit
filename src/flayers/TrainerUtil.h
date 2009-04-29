#ifndef TRAINER_UTIL_H
#define TRAINER_UTIL_H

#include "Trainer.h"
#include "ArgvParserContainer.h"
#include "ClassTopology.h"

bool testLoadSaveLayer(Trainer* trainer,ArgvParserContainer *apc);
bool compare(Trainer *trainer1,Trainer* trainer2,DataSet *dset);
void saveTrainer(std::string filename,Trainer* trainer,ArgvParserContainer* apc,std::string progname);
void CreateSplittedConnectionNN(Trainer *trainer,ClassTopology* class_topology,real lr,real dc,uint n_outputs,uint n_hiddens,std::string cost_type);
//you can get back the ArgvParserContainer with apc
Trainer* loadTrainer(std::string filename,ArgvParserContainer* &apc);
Trainer* TrainerFactory(ArgvParserContainer *apc,bool open_outputfile=true);
void generateTestDataSet(Trainer* trainer,ArgvParserContainer *apc,std::string prog_name);
#endif


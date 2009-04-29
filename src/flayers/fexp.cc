#include <stdio.h>
#include "PreDefArgvParser.h"
#include "ArgvParserContainer.h"
#include "NNLayer.h"
#include "Trainer.h"
#include "TrainerUtil.h"
#include <time.h>


using namespace std;


Topology *gtopo;
bool STDOUT=true;
//--------------------------------------------------------------------------
// fexp : Simple neural network Experiment
//--------------------------------------------------------------------------
int main(int argc, char* argv[])
{
  // Arguments parsers
  DataSetArgvParser* ds=new DataSetArgvParser();
  NNArgvParser* nn=new NNArgvParser();
  BpropTypeArgvParser* bpt=new BpropTypeArgvParser();
  ArgvParserContainer *apc=new ArgvParserContainer(argv[0]);
  apc->add(ds);
  apc->add(nn);
  apc->add(bpt);
    // options parsing
  apc->parse(argc,argv);
  if(nn->getBool("reel_random"))
	srand( (unsigned)time(NULL) );
  //create appropriate trainer
  Trainer *trainer=TrainerFactory(apc);
  //save params
  if(nn->getBool("save_params"))
    trainer->topo->openParamsFile(trainer->output_filename,nn->getInt("n_epochs"),true);
  //display fprop and bprop paths
  trainer->displayfbprop(true);
  trainer->train(ds->train_set_,ds->test_set_,nn->getInt("n_epochs"),nn->getInt("batch_size"));
  // optional load/save checkup
  if(nn->getBool("load_save_test"))
    testLoadSaveLayer(trainer,apc);
  if(nn->getBool("save"))
    saveTrainer(trainer->output_filename+".save",trainer,apc,argv[0]);
  cout<<"test on test_set:"<<endl;
  trainer->test(ds->test_set_);
  //memory cleanup
  delete(trainer);
  delete(ds);
  delete(nn);
  delete(apc);
  return 0;
}

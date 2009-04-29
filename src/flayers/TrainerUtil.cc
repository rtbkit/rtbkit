#include "TrainerUtil.h"
#include "NNLayer.h"
#include "PreDefArgvParser.h"
#include "OptTopology.h"
#include "utils/filter_streams.h"

extern bool STDOUT;


using namespace std;
using namespace ML;

bool testLoadSaveLayer(Trainer* trainer,ArgvParserContainer *apc)
{
  DataSetArgvParser* ds=(DataSetArgvParser*)apc->getArgvParser("DATASET");
  ds->train_set_->init();
  real ret_before=trainer->test(ds->train_set_);
  string filename="test.save.params";
  saveTrainer(filename,trainer,apc,"test");
  ArgvParserContainer *new_apc=NULL;
  //saveTrainer("save",trainer,apc,"save");
  Trainer *trainer_cpy=loadTrainer(filename,new_apc);
  //compare(trainer,trainer_cpy,ds->train_set_);
  ds->train_set_->init();
  if (!trainer->topo->params_list.isSame(&(trainer_cpy->topo->params_list)))
    FERR("testLoadSaveLayer -> params_list are not the same! ");
  real ret_after=trainer->test(ds->train_set_);
  bool test_result=(ret_before&&ret_after);
  if (test_result)
    cout<<"testLoadSaveLayer -> ok"<<endl;
  else{
    cout<<endl<<ret_before<<" != "<<ret_after<<endl;
    FERR("testLoadSaveLayer -> is not working properly ");
  }
  return test_result;
}
bool compare(Trainer *trainer1,Trainer* trainer2,DataSet *dset)
{
  real ret1,ret2;
  uint count1=0;uint count2=0;
  for (uint i=0;i<dset->iter_size_;i++){
    trainer1->topo->inl->p=dset->input_;
    trainer2->topo->inl->p=dset->input_;
    ret1=trainer1->testCurrentExample();
    ret2=trainer2->testCurrentExample();
    //cout<<i<<" : "<<ret1<<" "<<ret2<<endl;
    if(ret1!=*dset->targ_)count1++;
    if(ret2!=*dset->targ_)count2++;
    if(ret1!=ret2){
      FERR("compare() -> not the same result");
      return false;
    }
    dset->next();
  }
  if(count1!=count2)
    FERR("ERROR, not the same result count");
  else
    cout<<"compare Error = "<<(real)count1/(real)dset->iter_size_;
  return true;
}

void saveTrainer(string filename,Trainer* trainer,ArgvParserContainer* apc,string progname)
{
  filter_ostream out(filename);

  //saving cmd line (1)
  out<<progname<<" "<<apc->getCmdLine(false)<<endl;
  apc->updateCmdLine();
  //saving cmd line (2)
  out<<progname<<" "<<apc->getCmdLine(false)<<endl;
  //saving cmd line (3)
  out<<progname<<" "<<apc->getCmdLine(true)<<endl;
  if (STDOUT)cout<<"saving Trainer in file:"<<filename<<endl;
  //saving parameters (4)
  trainer->topo->saveParams(out);
}

Trainer* loadTrainer(string filename,ArgvParserContainer* &apc)
{
  filter_istream in(filename);

  DataSetArgvParser* ds=new DataSetArgvParser();
  NNArgvParser* nn_settings=new NNArgvParser();
  BpropTypeArgvParser* bproptypeargs=new BpropTypeArgvParser();
  apc=new ArgvParserContainer("loadTrainer");
  apc->add(ds);
  apc->add(nn_settings);
  apc->add(bproptypeargs);

  char buffer[2560];
  in.getline((char*)&buffer,2560,'\n'); // start long options
  if (STDOUT)cout<<"start options :"<<buffer<<endl;
  in.getline((char*)&buffer,2560,'\n'); // current long options
  in.getline((char*)&buffer,2560,'\n'); // current short options
  if (STDOUT)cout<<"update options :"<<buffer<<endl;
  int size=0;
  char** options=stringToArgv(string(buffer),size);
  apc->parse(size,options);
  Trainer* trainer=TrainerFactory(apc);
  //fill up params list of topology
  trainer->topo->fillUp();
  //load parameters
  trainer->loadParams(in);
  freeStringToArgv(options,size);

  return trainer;
}

void CreateSplittedConnectionNN(Trainer *trainer,ClassTopology* class_topology,real lr,real dc,uint n_outputs,uint n_hiddens,string cost_type)
{
	trainer->set(class_topology);
	Layer** hiddens=(Layer**)malloc(n_outputs*(sizeof(Layer*)));
	for(uint i=0;i<n_outputs;i++){
		//create hidden Layers
		hiddens[i]=LayerFactory(cost_type,"hiddens"+tostring(i),n_hiddens,lr,dc);
		//add hidden Layer
		trainer->topo->addToBack(hiddens[i]);
		//create Connector
		new Connector(trainer->topo->inl,hiddens[i],lr,dc);
		new Connector(hiddens[i],class_topology->class_i[i],lr,dc);
	}
}

Trainer* TrainerFactory(ArgvParserContainer *apc,bool open_outputfile)
{
  //get specific argv parsers
  DataSetArgvParser* ds=(DataSetArgvParser*)apc->getArgvParser("DATASET");
  NNArgvParser* nn=(NNArgvParser*)apc->getArgvParser("NN_SETTINGS");
  BpropTypeArgvParser* bproptypeargs=(BpropTypeArgvParser*)apc->getArgvParser("OPT_BPROP");

  //create Trainer
  Trainer *trainer=new Trainer(ds->means,ds->vars);

  //get some parameters
  real lr=nn->getReal("learning_rate");
  real dc=nn->getReal("dec_const");
  uint n_outputs=ds->getInt("n_classes");
  uint n_inputs=ds->getInt("n_inputs");
  uint n_hiddens=nn->getInt("n_hidden");
  //create new Layers
  Layer *in=LayerFactory("Input","inputs",n_inputs,lr,dc);
  Layer *out=LayerFactory(nn->get("output_cost_type"),"outputs",n_outputs,lr,dc);
  //create Topology
  BpropType bprop_type=(BpropType)nn->getInt("bprop_type");
  bool adapt_lr =(bool)nn->getBool("adapt_lr");
  bool force_all_up_conn=(bool)nn->getBool("force_up_conn");
  bool single_neurons=(bool)nn->getBool("single_neurons");
  Topology *topology=new Topology(in,out,adapt_lr,lr,dc,bprop_type,force_all_up_conn);
  //set trainer
  trainer->set(topology);
  //set output filename
  if(open_outputfile){
    string filename=apc->getStringDescriptor();
    if (nn->get("output_filename")!="?")
      filename=nn->get("output_filename");
    trainer->output_filename=filename;
    trainer->openOutputFile(filename);
  }
  //set save params
  if(nn->getBool("save_params"))
    trainer->topo->openParamsFile(trainer->output_filename,nn->getInt("n_epochs"),true);
  trainer->stopTrainAtTime(nn->getReal("stop_train_time"));
  trainer->stop_train_error=nn->getReal("stop_train_error");
  trainer->test_on_fly=nn->getBool("test_on_fly");
  string topo=nn->get("topo_type");
  uint n=topo.size();//nb of topology to add
  char t;
  for(uint i=0;i<n;i++){
    t=topo[i];
    switch(t){
    // simple perceptron
    case 'p':new Connector(in,out,lr,dc);break;

    // std one hidden neural network
    case 's':{
      //create hidden Layer with one hidden Layer of N
      Layer *hidden=LayerFactory(nn->get("hidden_cost_type"),"hiddens",n_hiddens,lr,dc);
      //add hidden Layer
      trainer->topo->addToBack(hidden);
      //create Connector
      new Connector(trainer->topo->inl,hidden,lr,dc);
      new Connector(hidden,trainer->topo->outl,lr,dc);
    };break;

    //create hidden Layer with N hidden layers of 1 (one hidden)
    case 'O':{
      for(uint i=0;i<n_hiddens;i++){
        Layer *hidden=LayerFactory(nn->get("hidden_cost_type"),"hiddens"+tostring(i),1,lr,dc);
        //add hidden Layer
        trainer->topo->addToBack(hidden);
        //create Connector
        new Connector(trainer->topo->inl,hidden,lr,dc);
        Connector* ho=new Connector(hidden,trainer->topo->outl,lr,dc);
		//init properly outputs weights
        ho->randomInit((real)(1.0/sqrt((double)n_hiddens)));
      }
    };break;

    //connect n_hidden to each outputs
	case 'i':{
	  Layer** hiddens=(Layer**)malloc(n_outputs*(sizeof(Layer*)));
	  for(uint i=0;i<n_outputs;i++){
		//create hidden Layers
		hiddens[i]=LayerFactory(nn->get("hidden_cost_type"),"hiddens"+tostring(i),n_hiddens,lr,dc);
		//add hidden Layer
		trainer->topo->addToBack(hiddens[i]);
		//create Connector
		new Connector(trainer->topo->inl,hiddens[i],lr,dc);
		Connector* con=new Connector(hiddens[i],out->neurons[i],lr,dc);
		//backpropagation link
		out->lcdown.push_back(con);
	  }
    };break;

    //create n_hidden blocks of neurons that are connected to all outputs
    case 'h':{
	  Layer** hiddens=(Layer**)malloc(n_outputs*(sizeof(Layer*)));
	  for(uint i=0;i<n_outputs;i++){
		//create hidden Layers
		hiddens[i]=LayerFactory(nn->get("hidden_cost_type"),"hiddens"+tostring(i),n_hiddens,lr,dc);
		//add hidden Layer
		trainer->topo->addToBack(hiddens[i]);
		//create Connector
		new Connector(trainer->topo->inl,hiddens[i],lr,dc);
		Connector* con=new Connector(hiddens[i],out,lr,dc);
		//backpropagation link
		out->lcdown.push_back(con);
	  }
    };break;

    // Create n_hidden for each outputs and do backprop on some outputs
	case 'c':{
		  ClassTopology *class_topology=new ClassTopology(in,out);
          CreateSplittedConnectionNN(trainer,class_topology,lr,dc,n_outputs,n_hiddens,nn->get("hidden_cost_type"));
          class_topology->bprop_type=nn->getChar("bprop_type");
	};break;

    // Create n_hidden for each outputs and do backprop on some outputs
	case 'C':{
	  ClassTopology *class_topology=new ClassTopology(in,out);
	  class_topology->bprop_type=nn->getChar("bprop_type");
	  trainer->set(class_topology);
	  Layer* hidden=LayerFactory(nn->get("hidden_cost_type"),"hiddens",n_hiddens,lr,dc);
	  //add hidden Layer
	  trainer->topo->addToBack(hidden);
	  new Connector(trainer->topo->inl,hidden,lr,dc);
	  for(uint i=0;i<n_outputs;i++){
		//create Connector
		Connector* con=new Connector(hidden,class_topology->class_i[i],lr,dc);
		//backpropagation link
		out->lcdown.push_back(con);
	  }
    };break;

    // Create 2 hidden layers
    case '2':{
      //create hidden Layer 1
      Layer *hidden1=LayerFactory(nn->get("hidden_cost_type"),"hiddens1",nn->getInt("n_hidden"),lr,dc);
      //create hidden Layer 2
      Layer *hidden2=LayerFactory(nn->get("hidden_cost_type"),"hiddens2",nn->getInt("n_hidden"),lr,dc);
      //add hiddens Layers
      trainer->topo->addToBack(hidden1);
      trainer->topo->addToBack(hidden2);
      //create Connector
      new Connector(trainer->topo->inl,hidden1,lr,dc);
      new Connector(hidden1,hidden2,lr,dc);
      new Connector(hidden2,trainer->topo->outl,lr,dc);
    };break;

	// optimal one hidden topology (only outputs)
    case 'x':{
    	// create hidden layer and connect it to input layer
    	Layer *hiddens=LayerFactory(nn->get("hidden_cost_type"),"hiddens",nn->getInt("n_hidden"),lr,dc);
    	new Connector(in,hiddens,lr,dc);
    	//add hidden Layer
    	topology->addToBack(hiddens);
    	for(uint i=0;i<out->n;i++)
    		new Connector(hiddens,out->neurons[i],lr,dc);
    };break;

    // optimal one hidden topology (outputs and hidden)
        case 'X':{
        	// create hidden layer and connect it to input layer
        	Layer *hiddens=LayerFactory(nn->get("hidden_cost_type"),"hiddens",nn->getInt("n_hidden"),lr,dc);
        	//add hidden Layer
        	if (single_neurons==false)
        		topology->addToBack(hiddens);
        	// connected single hiddens
        	for(uint i=0;i<hiddens->n;i++){
        		if (single_neurons)
        			topology->addToBack(hiddens->neurons[i]);
        		new Connector(in,hiddens->neurons[i],lr,dc);
        		for(uint j=0;j<out->n;j++)
					new Connector(hiddens->neurons[i],out->neurons[j],lr,dc);
        	}
    };break;

    // OptTopopogy experiment
    case 'o':{
        char bprop_topo=bproptypeargs->getChar("topo");
        bool sim_std=bproptypeargs->getBool("sim_std");
        char bprop_type=bproptypeargs->getChar("type");

        switch(bprop_topo){
    	case 'h':{//to experiment OptHiddenTopology
    	    trainer->set(new OptHiddenTopology(in,out,n_hiddens,nn->get("hidden_cost_type"),lr,dc,sim_std));
    	};break;
    	case 'o':{//to experiment OptOutputTopology
    	    trainer->set(new OptOutputTopology(in,out,n_hiddens,nn->get("hidden_cost_type"),lr,dc,sim_std,bprop_type));
    	};break;
        case 'a':{//to experiment OptTopology
    	    trainer->set(new OptTopology(in,out,n_hiddens,nn->get("hidden_cost_type"),lr,dc,sim_std));
    	};break;
    	case 'x':{//to experiment XTopology
            trainer->set(new XTopology(in,out,n_hiddens,nn->get("hidden_cost_type"),lr,dc,sim_std));
    	};break;
    	case 'z':{//to experiment XTopology
            trainer->set(new ZTopology(in,out,n_hiddens,nn->get("hidden_cost_type"),lr,dc,sim_std));
    	};break;
	case 'w':{//to experiment WTopology
            trainer->set(new WTopology(in,out,n_hiddens,nn->get("hidden_cost_type"),lr,dc,sim_std));
    	};break;
    	default:FERR("createTrainer(...): undefine bprop_type :"+bprop_type);break;
    	}
      }// end opt optimal bprop
    }
  }
  //fill up params list of topology
  trainer->topo->fillUp();
  return trainer;
}
void generateTestDataSet(Trainer* trainer,ArgvParserContainer *apc,string ds_name)
{
  ofstream *out=new ofstream(ds_name.c_str());
  //get specific argv parsers
  DataSetArgvParser* ds=(DataSetArgvParser*)apc->getArgvParser("DATASET");
  DataSet* dset=ds->train_set_;
  int n_inputs=ds->getInt("n_inputs");
  real ret=999;
  *out<<dset->iter_size_<<" "<<ds->getInt("n_inputs")<<" "<<ds->getInt("n_classes")<<endl;
  dset->init();
  for (uint i=0;i<dset->iter_size_;i++){
    trainer->topo->inl->p=dset->input_;//set input
    ret=trainer->testCurrentExample();
    printVecReal(dset->input_,n_inputs,false,10,out);
    *out<<" "<<ret<<" "<<*dset->targ_<<")"<<endl;
    dset->next();
  }
  out->close();
}

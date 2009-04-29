#include <math.h>
#include "PreDefArgvParser.h"


using namespace std;


extern bool STDOUT;

// data set
string data_set_choice="data_set choice\n\t(b)reat_cancer,(s)onar,(i)ono,(d)iabete,(p)ima,(l)etters,(U)SPS,(h)ousing\n\t(Q)CelliQue,(q)ISPIQue,(O)CelliOnt,(o)ISPIOnt,(S)wissRoll";
string data_set_description="data_set related parameters";
opt opts_data_set[]={
  opt('t',"train_f","1","train_fraction",REAL),
  opt('v',"valid_f","0","valid fraction",REAL),
  opt('T',"test_f","0","Test fraction",REAL),
  opt('i',"n_inputs","?","nb (i)nputs",INT),
  opt('I',"add_n_inputs_places","0","add input place;usefull to add inputs in incremental system",INT),
  opt('f',"fraction_input","1","fraction of nb_inputs to use",REAL),
  opt('F',"fraction_data","1","fraction of data to use",REAL),
  opt('d',"dataset_filename","letters.dat","data set filename",STRING),
  opt('b',"base_dataset_rep","/home/fpieraut/mlboost/flayers/Datasets/","base dataset repository",STRING),
  opt('c',"n_classes","-1","number of (c)lasses",INT),
  opt('e',"n_examples","0","nb total of examples",INT),
  opt('n',"normalize","1","normalize train,valid and test sets",BOOL),
  opt('N',"regression","0","regression linear mode ->normalize targets;n_classes ==1",BOOL),
  opt('s',"do_shuffle","1","shuffle data_set",BOOL),
  opt('X',"fract_iter","1","fraction of train_set to use by iteration",REAL),
  opt('>',"0","","",NOTDEF)
};
/*noargopt noargopts_data_set[]={
  noargopt(">","","")
};*/

// neural net setting
string nn_description="Neural Network related parameters";
opt opts_nn[]={
  opt('T',"topo_type","?","topology type (OneHidden/Perceptron etc.; see TrainerUtil.h)",STRING),
  opt('o',"output_filename","test","force this output filename,if ?->generate automatic filename",STRING),
  opt('h',"n_hidden","0","nb (h)idden",INT),
  opt('l',"learning_rate","0.1","learning rate",REAL),
  opt('D',"dec_const","0","decrease const",REAL),
  opt('i',"inc_factor","1.1","increase factor for adaptatif lr",REAL),
  opt('d',"dec_factor","0.7","decrease factor for adaptatif lr",REAL),
  opt('e',"n_epochs","0","nb (e)pochs",INT),
  opt('b',"batch_size","1","batch size;stochatic=1",INT),
  opt('s',"save_params","0","save parameters at eachs iterations",BOOL),
  opt('t',"stop_train_time","0","maximum time to train (if 0->no limits)",REAL),
  opt('E',"stop_train_error","0","stop train when this training error has been reach",REAL),
  opt('O',"output_cost_type","SigmMSE","cost function (MSE_SIGM-LOGSOFTMAX)",STRING),
  opt('H',"hidden_cost_type","TanhMSE","cost function of hidden layers",STRING),
  opt('X',"load_save_test","0","to a save and then load test",BOOL),
  opt('S',"save","1","save trainer",BOOL),
  opt('f',"test_on_fly","1","do the test on fly (much faster but approximation)",BOOL),
  opt('r',"reel_random","0","active reel random generator srand( (unsigned)time( NULL ) )",BOOL),
  opt('p',"pseudo_random","0","active pseudo random generator srand(p)",INT),
  opt('a',"adapt_lr","0","activation of adaptatatif lr",BOOL),
  opt('B',"bprop_type","0","bprop type STD=0, DEEP_FIRST=1, MAX_SENSITIVITY=2, MIN_SENSITIVITY=3",INT),
  opt('F',"force_up_conn","0","force all upper connections (deep first bprop)",BOOL),
  opt('1',"single_neurons","1","add connected single neurones (not the layer that contain them); works only with --opt",BOOL),
  opt('>',"0","","",NOTDEF)
};
noargopt noargopts_nn_set[]={
  noargopt("forceup","force_up_conn","1"),
  noargopt("deepfirst","bprop_type","1"),
  noargopt("maxsens","bprop_type","2"),
  noargopt("minsens","bprop_type","3"),
  noargopt("tanh","output_cost_type","TanhMSE"),
  noargopt("sigm","output_cost_type","SigmMSE"),
  noargopt("lsm","output_cost_type","LogSoftMax"),
  noargopt("oh","topo_type","s"),//s for std
  noargopt("ioh","topo_type","i"),
  noargopt("perc","topo_type","p"),
  noargopt("2h","topo_type","2"),
  noargopt("opt","topo_type","X"),
  noargopt("optout","topo_type","x"),
  noargopt("loadsave_test","load_save_test","1"),
  noargopt("no_single","single_neurons","0"),
  noargopt(">","","")
};

// opt bprop settings
string optbprop_description="bprop parameters";
opt opts_optbprop[]={
  opt('b',"topo","s","bprop config",STRING),
  opt('s',"sim_std","0","simulate std backprop",BOOL),
  opt('t',"type","1","bprop config (only on bprop_type=o) e=only output error | s = reversesorted sensitivity S=(same as s + only output error) ",BOOL),
  opt('>',"0","","",NOTDEF)
};
noargopt noargopts_optbprop_set[]={
  noargopt("std","topo","s"),
  noargopt("bprophidden","topo","h"),
  noargopt("bpropoutput","topo","o"),
  noargopt("bpropx","topo","x"),
  noargopt("bpropz","topo","z"),
  noargopt("optimal","topo","o"),
  noargopt(">","","")
};

//incremental
string inc_topo_description="incremental OneHidden related parameters";
opt opts_inc_topo[]={
  opt('a',"block_size","5","n hidden per block(i)ncremental topology",INT),
  opt('n',"n_const_iter","1","nb of constructive iterations",INT),
  opt('f',"freeze_old_units","0","(bool) freeze old units at each iteration",BOOL),
  opt('t',"temp_freeze_old","0","(bool) temporaly freeze old units at each iteration",BOOL),
  opt('o',"opt_out_layer","0","(bool) always optimise output layer",BOOL),
  opt('e',"n_freeze_old_units_epochs","2","nb of (e)pochs when old units are freezed",INT),
  opt('i',"auto_incremental","0","active auto incremental topology base on earlystopping + indicate max iterations",BOOL),
  opt('c',"circular_reoptimization","0","circular reoptimization of added parameters for incremental architecture",BOOL),
  opt('s',"start_iteration","5","start incrementation after i iteration",INT),
  opt('S',"start_n_blocks","1","nb of block at the begining of optimisation",INT),
  opt('m',"multi_layers","0","multi layer mode",BOOL),
  opt('n',"n_blocks","1","numbers of same layer",INT),
  opt('>',"0","","",NOTDEF)
};
//Booster
string boost_description="Booster OneHidden related parameters";
opt boost_Opt[]={
  opt('o',"boost_type","0","Optimisation type (1=linear,2=quadratic,3=cubic,4=lin regression,5=regression layer;6=perturbation)",CASE),
  opt('h',"n_history","5","nb of parameters in history (for extrapolation)",INT),
  opt('a',"extrapolation_after","5","do extrapolation after i iterations",INT),
  opt('f',"extrapolation_factor","5","extrapolation factor",INT),
  opt('s',"extrapolation_start","5","start extrapolation after 's' iteration",INT),
  opt('i',"inc_extrapolation","0","increase extrapolation",REAL),
  opt('b',"do_bprop","1","after extrapolation, do bprop",BOOL),
  opt('>',"0","","",NOTDEF)
};
//EveryNEpochs
string every_description="Every N epochs related parameters";
opt every_Opt[]={
  opt('t',"exp_type","0","experimentation type; 1=perturbation;2=lr increase;",CASE),
  opt('n',"every_n_epochs","5","do exp_type every n_epochs",INT),
  opt('s',"start_epochs","5","start exp_type every start_epochs",INT),
  opt('i',"lr_inc_factor","4","increase learning rate factor",REAL),
  opt('E',"perturb_extremum","0","if you are in perturbation mode, use extremum perturbation values",BOOL),
  opt('p',"perturb_fraction","0.2","if you are in perturbation mode, do perturbation of x% of params values",REAL),
  opt('b',"do_bprop","1","after exp_type, do bprop",BOOL),
  opt('>',"0","","",NOTDEF)
};
void DataSetArgvParser::CoherenceParamsValuesChecker()
{
  ArgvParser::CoherenceParamsValuesChecker();
  // compute total - 1
  real diff =getReal("train_f")+getReal("valid_f")+getReal("test_f") -1 ;
  if ((diff>=0.01) || (diff<=-0.01))
    FERR("sum of train_f + valid_f + test_f != 1");
  if (getChar("dataset_filename")=='?')
    FERR("you didn't choose any data_set!");
}

/*void DataSetArgvParser::Coherence(NNArgvParser* nn_settings)
{
  if(getInt("n_classes")==1){//regression setting
    cout<<"WARNING forcing regression setting because n_classes=1 !";
    nn_settings->set("output_cost_type","Linear");
  }
  }*/
void DataSetArgvParser::parse(int argc,char* argv[])
{
  inherited::parse(argc,argv);
  char data_set_choice='x';//
  if (get("dataset_filename").size()==1)
    data_set_choice=getChar("dataset_filename");
  switch ( data_set_choice ) {
  case 'b':db_name_="breastcancer";set("n_classes","2");break;
  case 's':db_name_="sonar";break;
  case 'i':db_name_="ionosphere";break;
  case 'd':db_name_="diabetes";set("n_classes","2");break;
  case 'p':db_name_="pima";break;
  case 'l':db_name_="letters";break;
  case 'U':db_name_="USPS";break;
  case 'h':db_name_="housing";break;
  case 'x':db_name_=get("base_dataset_rep")+get("dataset_filename");break;
  default:
    string msg="No data_set associate to : "+tostring(data_set_choice)+"\n";
    FERR(msg.c_str());
  }
  if(!loadData(db_name_))
    FERR(("problem with loading file "+db_name_).c_str());
  if(getBool("normalize"))
    normalizeDataInputs();//printData();
  if(getInt("n_classes")==1)
    cout<<"WARNING: are we in regression mode? if yes then --regression 1"<<endl;
  if((getInt("n_classes")==1)&&getBool("regression"))
    normalizeTarget();
  if((getInt("n_classes")!=1)&&getBool("regression"))
    FERR("can't be in regression mode if n_classes=1");
  // reduce matrice dim for test
  real fraction_input=getReal("fraction_input");
  if ((fraction_input<1)&&(fraction_input>0)){
    FERR("reduce inputs size not implemented");
  }
  // split data in train_set,valid_set and train_set
  splitTrainValidTest();
}

void DataSetArgvParser::splitTrainValidTest()
{
  uint n_inputs=getInt("n_inputs")+getInt("add_n_inputs_places");
  uint n_examples=getInt("n_examples");
  //compute size
  uint valid_size = int((real)n_examples*getReal("valid_f"));
  uint test_size = int((real)n_examples*getReal("test_f"));
  uint train_size = n_examples-(valid_size+test_size);
  uint  train_set_iter_size = (uint) (train_size*getReal("fract_iter"));
  DataSet *ds=NULL;
  if(getBool("do_shuffle")==1)
    ds=new RandomDataSet(data_,targets_,n_examples,n_examples,n_inputs,n_inputs);
  else
    ds=new SeqDataSet(data_,targets_,n_examples,n_examples,n_inputs,n_inputs);

  train_set_=ds->getSubSet((uint)0,train_size,train_set_iter_size);
  valid_set_=ds->getSubSet(train_size,valid_size,valid_size);
  test_set_=ds->getSubSet(train_size+valid_size,test_size,test_size);
  valid_set_->init();
  test_set_->init();
  train_set_->init();
  if (STDOUT){
	  cout<<"train examples = "<<train_set_->iter_size_<<endl;
	  cout<<"dataset split ["<<train_set_->size_<<'|'<<valid_set_->size_<<'|'<<test_set_->size_<<"]"<<endl;
  }
}
string DataSetArgvParser::getStringDescriptor(bool short_descr,ofstream *output)
{
  string descr;
  if (short_descr){
    descr+="DB_"+troncString(db_name_)+"_";
  }else{
    descr+="DataBase name : "+db_name_;
  }
  if (output){
    (*output)<<descr<<endl;
  }
  descr+=inherited::getStringDescriptor(short_descr,output);
  return descr;
}
//-------------------------------------------------------------------------------------
bool DataSetArgvParser::loadData(string filename)
{
  ifstream file(filename.c_str());
   // load data info header [n_examples | n_inputs | n_classes]
  int n_examples;
  int n_inputs;
  int n_classes;
  if (file.is_open()){
    //IMPORTANT :order of the 3 first info to read in the dataset file
    file>>n_examples>>n_inputs>>n_classes;
  }else{
    return false;
  }
  // reduce nb of examples
  real fraction_data=getReal("fraction_data");
  if ((fraction_data<1)&&(fraction_data>0)){
    n_examples=(unsigned int)(fraction_data*(real)n_examples);
  }
  set("n_examples",n_examples);
  if (STDOUT){
	  cout<<"DataInfo----------"<<endl;
	  cout<<"n_examples :"<<n_examples<<endl;
	  cout<<"n_inputs   :"<<n_inputs<<endl;
	  cout<<"n_classes  :"<<n_classes<<endl;
  }
  set("n_inputs",n_inputs+getInt("add_n_inputs_places"));
  set("n_classes",n_classes);
  set("n_examples",n_examples);
  n_inputs=getInt("n_inputs");
  uint diff_input_size=getInt("add_n_inputs_places");
  // alloc data_ and targets_
  data_=realCalloc(n_examples,n_inputs);
  targets_=realMalloc(n_examples);
  // load data
  if (STDOUT)cout<<"loading data ...";

  real *class_distribution=realMalloc(n_classes);
  real *pdata=data_;
  real *ptargets=targets_;
  int i;
  for(i=0;i<n_examples;i++){
    for(int j=0;j<n_inputs;j++)
      file>>*pdata++;
    pdata+=diff_input_size;//if (diff_input_size=input_size)pdata is unchanged
    file>>*ptargets;
    if(n_classes>1)
      class_distribution[(int)*ptargets]++;
    *ptargets++;
	if((i%1000==0) && STDOUT){cout<<"*";cout.flush();};
  }
  if (STDOUT)cout<<"ok"<<endl;
  for(i=0;i<n_classes;i++)
    class_distribution[i]/=n_examples;
  if (STDOUT){
	  cout<<"classes distribution: ";
	  printVecReal(class_distribution,n_classes);
  }
  return true;
}
//-------------------------------------------------------------------------------------
void DataSetArgvParser::save(string filename)
{
  ofstream out(filename.c_str());
  if (out.is_open()){
    uint n_examples=getInt("n_examples");
    out<<n_examples<<" "<<get("n_inputs")<<" "<<get("n_classes")<<endl;
    real* ptarg=targets_;
    real* pinput=data_;
    uint input_size=getInt("n_inputs");
    for(uint i=0;i<n_examples;i++){
      printVecReal(pinput++,input_size,false,10,&out);out<<" "<<*ptarg++<<endl;
    }
    cout<<"DataSetArgvParser has been save in :"<<filename<<endl;
    out.close();
  }else
    FERR("DataSetArgvParser::save(string filename)->unable to create file"+filename);
}
//-------------------------------------------------------------------------------------
void DataSetArgvParser::printData()
{
  int n_inputs=getInt("n_inputs");
  int n_examples=getInt("n_examples");
  //print data
  real *pdata=data_;
  real *ptargets=targets_;
  for(int i=0;i<n_examples;i++){
    cout<<i<<": ";
    for(int j=0;j<n_inputs;j++)
      cout<<*pdata++<<" ";
    cout<<"->"<<*ptargets++<<endl;
  }
}
//-------------------------------------------------------------------------------------
void DataSetArgvParser::normalizeDataInputs()
{
  if (STDOUT)
	  cout<<"data normalisation ...";
  uint n_inputs=getInt("n_inputs");
  uint n_examples=getInt("n_examples");
  means=realMalloc(n_inputs);
  vars=realMalloc(n_inputs);
  real *sum=realMalloc(n_inputs);
  if(!means||!vars||!sum)
    FERR("memory allocation problems");
  //pointers
  real *pdata=data_;
  real *psum=sum;
  real *pmean=means;
  real *pvar=vars;
  uint i,j;
  //compute sum
  for(i=0;i<n_examples;i++){
    for(uint j=0;j<n_inputs;j++){
      *psum++ += *pdata++;
    }
    psum=sum;
  }
  //mean
  psum=sum;
  for(j=0;j<n_inputs;j++){
    *pmean++ +=*psum++/n_examples;
  }
  //variance
  pmean=means;
  pdata=data_;
  for(i=0;i<n_examples;i++){
    for(j=0;j<n_inputs;j++,++pdata){
      //remove mean;
      *pdata -= *pmean++;
      *pvar++ +=*pdata*(*pdata);
    }
    pvar=vars;
    pmean=means;
  }
  pvar=vars;
   //variance
  for(j=0;j<n_inputs;j++,pvar++){
    *pvar=sqrt(*pvar/n_examples);
  }
  //normalize data
  pdata=data_;
  pvar=vars;
  for(i=0;i<n_examples;i++,pvar=vars){
    for(j=0;j<n_inputs;j++){
      //divide by variance if !=0
      if(*pvar)
        *pdata++ /=*pvar++;
      else{
        *pvar++;*pdata++;
      }
    }
  }
  if (STDOUT){
	  cout<<"ok"<<endl;
	  cout<<"mean     : ";printVecReal(means,n_inputs);
	  cout<<"variance : ";printVecReal(vars,n_inputs);
  }
  free(sum);
}
//-------------------------------------------------------------------------------------
void DataSetArgvParser::normalizeTarget()
{
  if (STDOUT)cout<<"targets normalisation ...";
  int n_examples=getInt("n_examples");
  real mean=0;
  real var=0;
  real sum=0;
  real diff;
  //pointers
  real *ptargs=targets_;
  int i;
  //compute sum
  for(i=0;i<n_examples;i++)
    sum += *ptargs++;
  //mean
  mean=sum/n_examples;
  ptargs=targets_;
  //variance
  for(i=0;i<n_examples;i++){
    diff=*ptargs++ -mean;
    var+=diff*diff;
  }
  var=sqrt(var/n_examples);
  //normalize targets
  ptargs=targets_;
  for(i=0;i<n_examples;i++){
    // minus mean
    *ptargs -=mean;
    //divide by variance if !=0
    if(var)
      *ptargs /=var;
     *ptargs++;
  }
  if (STDOUT)cout<<"ok"<<endl;
}
DataSetArgvParser::~DataSetArgvParser()
{
  free(data_);
  free(targets_);
  free(means);
  free(vars);
  if (train_set_)
    delete(train_set_);
  if(valid_set_)
    delete(valid_set_);
  if(test_set_)
    delete(test_set_);
}

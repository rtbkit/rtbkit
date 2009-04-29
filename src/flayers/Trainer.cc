#include "Trainer.h"
#include "limits.h"
#include "fTimeMeasurer.h"
#include <iomanip>


using namespace std;


extern fTimeMeasurer time_measurer;
extern bool STDOUT;
Trainer::Trainer(real *means, real *vars,bool stdout_on)
{
  topo=NULL;
  stop_train_error=0;
  stop_train_time=FLT_MAX;
  this->stdout_on=stdout_on;
  start_timer=true;
  test_on_fly=true;
  iter_i=0;
  this->means=means;
  this->vars=vars;
  if (STDOUT)cout<<"Creating Trainer :"<<endl;
}

real* Trainer::normalize(real *inputs, bool allocate)
{
	uint n = topo->inl->n;
	real *pmean=means;
	real *pvar=vars;
	real *ivals=inputs;
	real *ninputs=inputs;
	if (allocate){
		// create normalized inputs
		ninputs=realMalloc(n);
		memcpy(ninputs,inputs,n*sizeof(real));
		ivals=ninputs;
	}

	for(uint i=0;i<topo->inl->n;i++){
		*ivals -= *pmean++;
		//divide by variance if !=0
	    if(*pvar)
	      *ivals++ /=*pvar++;
	    else{
	      *pvar++;
		  *ivals++;
	    }
	}
	return ninputs;
}

real Trainer::fprop(real *inputs)
{
	//set input pointer
	topo->inl->p=normalize(inputs,true);
	real ret = testCurrentExample();
	free(topo->inl->p);
	return ret;
}

real Trainer::trainOrTest(DataSet *dset,bool do_train,uint batch_size)
{
  if(!topo)
    FERR(" Trainer::trainOrTest(...) -> unset topology");
  if(topo->inl->n!=dset->input_size_)
    FERR(" Trainer::trainOrTest(...) -> incompatibility in input size! ("+tostring(topo->inl->n)+"!="+tostring(dset->input_size_)+")");
  if (dset->size_<=0){
    FERR("Trainer:trainOrTest(...) -> empty data set");
    return -1;
  }
  uint count=0;//nb of error
  uint n_jumb_examples=0;//nb of jumb example (sampling distribution)
  long double tot_cost=0;
  real ret=-1;//class # or result
  uint batch_count=0;
  for (uint i=0;i<dset->iter_size_;i++){
    batch_count++;
    topo->inl->p=dset->input_;//set input
    ret=testCurrentExample();
    if (do_train)
      topo->bprop(*dset->targ_,(batch_size==1));
    //batch
    if((batch_size!=1)&&(batch_count==batch_size)){
      topo->update(batch_size);
      batch_count=0;
    }
    //classification count
    if(ret!=*dset->targ_)
      count++;
     tot_cost+=topo->outl->computeCost(*(dset->targ_));
    //cout<<ret<<" -> "<<*dset->targ_<<" "<<*outc<<endl;
    //cout<<ret<<" -> "<<*dset->targ_<<" ";realPow(outc->p,outc->size,2.78);
     dset->Next();
     if(dset->sampling_prob){
       i+=dset->current_jump_counter;
       n_jumb_examples+=dset->current_jump_counter;
     }
  }
  real n_examples=dset->iter_size_-n_jumb_examples;
  error_classif=(real)count/n_examples;
  topo->outl->mean_cost=tot_cost/n_examples;
  if(stdout_on)cout<<" Cost = "<<setw(10)<<tot_cost;
  cost=tot_cost;
  topo->cost=cost;
  if(topo->outl->n>1)
    if(stdout_on)cout<<"  Error = "<<setw(10)<<error_classif*100.0<<"% ";
  if(stdout_on)cout<<endl;
  //printVecReal(topo->outl->p,topo->outl->n);
  return cost;
}
uint Trainer::updateSamplingDistribution(DataSet *dset)
{
  uint count=0;
  if(!topo)
    FERR(" Trainer::trainOrTest(...) -> unset topology");
  if(topo->inl->n!=dset->input_size_)
    FERR(" Trainer::trainOrTest(...) -> incompatibility in input size! ("+tostring(topo->inl->n)+"!="+tostring(dset->input_size_)+")");
  if (dset->size_<0){
    FERR("Trainer:trainOrTest(...) -> empty data set");return 0;
  }
  for (uint i=0;i<dset->iter_size_;i++){
    topo->inl->p=dset->input_;//set input
    if(*dset->targ_==testCurrentExample()){
      dset->reduceSamplingProb_i();
      count++;
    }
    dset->probSamplingNext();
    if(dset->sampling_prob)i+=dset->current_jump_counter;
  }
  return count;
}
real Trainer::train(DataSet *train_set,DataSet* test_set,int n_epochs,int batch_size)
{
  if(topo==NULL){
    FERR("Trainer:trainOrTest(...) -> indefine Topology");return -1;
  }
  if(start_timer)time_measurer.startTimer();
  //stochatic gradient setting
  if (STDOUT){
	  if (batch_size==1)
		cout<<"stochastic gradient "<<endl;
	  else
		cout<<"mini-batch ("<<batch_size<<") "<<endl;
  }
  //variable initialisation
  double time=0;real ret=-1;error_classif=100;
  if(stdout_on&&start_timer)cout<<"train on "<<train_set->iter_size_<<" el."<<endl;
  int i=0;
  while ((i<n_epochs)&&(time<stop_train_time)){
     i++;iter_i++;
    //error classification stop watch
     if((topo->inl->n>1)&&(error_classif<=stop_train_error)&&(i!=0)){
       cout<<"specific error has been reach! :"<<error_classif<<"<="<<stop_train_error<<endl;
      return ret;
    }
    initEpoch(i);
    if(stdout_on)cout<<"#"<<iter_i<<" ";
    time_measurer.stopTimer();
    time=FACTOR_CLK*time_measurer.getStopTime();
    if(stdout_on)cout<<" t= "<<setw(10)<<time;
    ret=coreTrainFct(train_set,batch_size);
    //save results
    if (out.is_open()){
      out<<iter_i<<" "<<setw(10)<<time<<" "<<setw(10)<<ret;
      if(topo->outl->n>1)
        out<<" "<<setw(10)<<error_classif*100.0;
      if(test_set){
        if(test_set->size_){
          trainOrTest(test_set,false,batch_size);
          out<<" "<<setw(10)<<error_classif*100.0;
        }
      }
    }
    out<<endl;
  }
  return ret;
}
Trainer::~Trainer()
{
  if (out.is_open())
    out.close();
  if(stdout_on&&(output_filename!=""))
    cout<<"output filename :"<<endl<<output_filename<<endl;
  if(topo)delete(topo);
  topo=NULL;
}
void Trainer::displayfbprop(bool all)
{
  cout<<"---------fprop-------------"<<endl;
  topo->displayfprop(all);
  cout<<"---------bprop-------------"<<endl;
  topo->displaybprop(all);
  cout<<"---------------------------"<<endl;
}

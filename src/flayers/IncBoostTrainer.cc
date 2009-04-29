#include "IncBoostTrainer.h"
#include <iomanip> 


using namespace std;


void IncBoostTrainer::createBoostDataSet(DataSet *dset)
{
  freeMem();
  real* new_data=realMalloc(topo->outl->n*dset->size_*sizeof(real));
  bds=new SeqDataSet(new_data,dset->targs_,dset->size_,topo->outl->n,false);
}
void IncBoostTrainer::cleanBoostDataSetAndReactivate()
{
	memset(bds->data_,0,topo->outl->n*bds->size_*sizeof(real));
	topo->setfbprop(true,true);
}
void IncBoostTrainer::updateBoostDataSet(DataSet *dset)
{ 
  if(!bds)
    FERR("BoostTrainer::updateBoostDataSet(...);you should create BoostDataSet first");
  dset->init();bds->init();
  real *pi=NULL;
  real *po=NULL;
  for (uint i=0;i<dset->iter_size_;i++){
    topo->inl->p=dset->input_;//set input
    topo->fprop(false);//no apply activation fct
    pi=bds->input_;
    po=topo->outl->p;
    //add new part to output
    for (uint j=0;j<topo->outl->n;j++)
      (*pi++)+=(*po++);
    dset->next();bds->next();
  }
  //desactivate fprop
  topo->setfbprop(false,false);
  //HYPER IMPORTANT : leave do_fprop of ouput layer to true;
  topo->inl->do_fprop=true;
  topo->outl->do_bprop=true;
  topo->outl->do_fprop=true;
}
real IncBoostTrainer::trainOrTest(DataSet *dset,bool do_train,uint batch_size)
{
  if(!bds)
    return Trainer::trainOrTest(dset,do_train,batch_size);
  if(topo->inl->n!=dset->input_size_)
    FERR(" Trainer::trainOrTest(...) -> incompatibility in input size!");
  if (dset->size_<0){
    FERR("Trainer:trainOrTest(...) -> empty data set");return -1;
  }
  uint count=0;
  long double tot_cost=0;
  real ret=-1;//class # or result 
  uint batch_count=0;
  real *po=NULL;
  real *pbo=NULL;
  dset->init();bds->init();//to be safe because we are using a boost dataset
  for (uint i=0;i<dset->iter_size_;i++){
	//cout<<"\n input :";printVecReal(dset->input_,topo->outl->n);
    batch_count++;
    topo->inl->p=dset->input_;//set input
    topo->fprop(false);//do real fprop without applying activation fct
    po=topo->outl->p;//pointer to outputs
	//cout<<"\n output :";printVecReal(topo->outl->p,topo->outl->n);
    pbo=bds->input_;//pointer to boost outputs 
    //add boost outputs
    for(uint i=0;i<topo->outl->n;i++)
      (*po++)+=(*pbo++);
	//cout<<"\n sum before apply :";printVecReal(topo->outl->p,topo->outl->n);
    topo->outl->apply();
	//cout<<"\n sum after apply  :";printVecReal(topo->outl->p,topo->outl->n);
    ret=topo->outl->result();
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
    tot_cost+=topo->outl->computeCost(*dset->targ_);
    //cout<<ret<<" -> "<<*dset->targ_<<" "<<*outc<<endl;
    //cout<<ret<<" -> "<<*dset->targ_<<" ";realPow(outc->p,outc->size,2.78);
    dset->next();bds->next();
  }
  error_classif=count/(real)dset->iter_size_;
  topo->outl->mean_cost=tot_cost/(real)dset->iter_size_;
  if(stdout)cout<<" Cost = "<<setw(10)<<tot_cost;
  cost=tot_cost;
  if(topo->outl->n>1)
    if(stdout)cout<<"  Error = "<<setw(10)<<error_classif*100.0<<"% ";
  if(stdout)cout<<endl;
  return cost;
}

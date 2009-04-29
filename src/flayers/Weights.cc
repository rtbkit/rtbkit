#include "Weights.h"


using namespace std;


real randomu(real n){return (2*n*((double(rand())/RAND_MAX)-0.5));}

Weights::Weights(string name_,uint size,real lr_,real dc_)
{
  name=name_;n=size;lr=lr_;start_lr=lr_;dc=dc_;
  w=realMalloc(n);
  wu=realMalloc(n);
  do_bprop=true;do_fprop=true;
}
void Weights::randomInit(real max)
{
	real *p=w;
	for(uint i=0;i<n;i++)
		*p++=randomu(max);
}
void Weights::perturb(real fraction,bool extremum)
{
  real* p=w;
  for(uint i=0;i<n;i++,++p){
      if(extremum){
	if (randomu(1)>0)
	  (*p)+=fraction*(*p);
	else
	  (*p)-=fraction*(*p);
      }else
	(*p)+=randomu(fraction*(*p)); 
    p++;
  }
}  
ostream& operator<<(ostream& out,Weights& w)
{
	out<<"\nWeights :"<<endl;
	printVecReal(w.w,w.n,false,10,&out);return out;
}
LrWeights::LrWeights(string name_,uint size,real lr_,real dc_):Weights(name_,size,lr_,dc_)
{
  inc_factor=1.1f;dec_factor=0.7f;
  old_weights_updates=realMalloc(n);
  old_weights=realMalloc(n);
  lrs=realMalloc(n);
   //set all lr to 1
  set(lrs,n,lr*inc_factor);//at first iteration, all lrs will be reduce
}
void LrWeights::updateParamsLr()
{
	real *plr=lrs;
	real *owu=old_weights_updates;//old weights update
	real *cwu=wu;//current weights update 
    for(uint i=0;i<n;i++){
		if( ((owu[i]>0)&&(cwu[i]>0)) || ((owu[i]<0)&&(cwu[i]<0)) )
			plr[i]*=inc_factor;
		else
			plr[i]*=dec_factor;
	}
	//printVecReal(lrs,n);
}
void LrWeights::initEpoch(int i)
{
	if(i==1)memcpy(old_weights,w,n*sizeof(real));
	//compute new gradient
	real *pwu=wu; 
    real *pw=w;
	real *pow=old_weights;
    for(uint j=0;j<n;j++)
		*pwu++=(*pw++)-(*pow++);
	//clear all gradient with hack to memory cpy optimisation
	real* temp_old_weights_updates=old_weights_updates;
    old_weights_updates=wu;
    wu=temp_old_weights_updates;
    memset(wu,0,n*sizeof(real));
	//update learning rates
	updateParamsLr();
	memcpy(old_weights,w,n*sizeof(real));
}
LrWeights::~LrWeights()
{
	free(old_weights);
	free(old_weights_updates);
	free(lrs);
	lrs=NULL;old_weights_updates=NULL;old_weights=NULL;
}


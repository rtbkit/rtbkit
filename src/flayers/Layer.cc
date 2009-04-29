#include "Layer.h"
#include "Topology.h"


using namespace std;


extern Topology *gtopo;
extern bool STDOUT;

bool compare_output_sensitivity_max(const Layer *a,const Layer *b){ return a->sensitivity[0]>b->sensitivity[0];}
bool compare_output_sensitivity_min(const Layer *a,const Layer *b){ return a->sensitivity[0]<b->sensitivity[0];}

///
/// default Layer constructor
///
Layer::Layer(string name_,uint size,real lr_,real dc_):Weights(name_,size,lr_,dc_)
{
  classname="Layer";
  p=realMalloc(n);
  targs=realMalloc(n);
  bias=w;
  bias_update=wu;
  sensitivity=realMalloc(n);
  setMinMax(0,1);
  do_fprop=true;do_bprop=true;
  neurons=(Layer**)malloc(n*(sizeof(Layer*)));
  for(uint i=0;i<n;i++){
  	neurons[i]=new Layer(this,i,1);
	lneurons.push_back(neurons[i]);
  }
}
///
/// construct a virtualLayer who is a subLayer of "from" Layer
/// \param from source Layer
/// \param start start position of in "from" Layer
/// \param size size of the new Layer
///
Layer::Layer(Layer* from,uint start,uint size)
	:Weights(from->name+"_"+tostring(start),0,from->lr,from->dc)
{
	n=size;
	if((start+size)>from->n)
		FERR("can't construct InsideLayer");
  p=from->p+start;//Layer connection point
  sensitivity=from->sensitivity+start;//output sensitivity
  // should not be use
  bias=from->bias+start;
  bias_update=from->bias_update+start;
  targs=from->targs+start;
}
///
/// forward-propagation
///
void Layer::fprop()
{
  if(do_fprop){
    //fprop in all connector
    for(CI i=lcup.begin();i!=lcup.end();i++)
      (*i)->fprop();
    for(LI i=lneurons.begin();i!=lneurons.end();i++)
          	(*i)->fprop();
  }
}

void Layer::addUpConnection(Connector* cup)
{
    lcup.push_back(cup);
	if (lcup.size()>0)
		if ((cup->outl->n) != ((*lcup.begin())->outl->n))
			cout<<"WARNING : possible confusion to compute <"<<name<<">HiddenSensitivity();"<<endl;
}

///
/// multiply Lr By a factor
/// \param factor increase factor for all learning rates
///
void Layer::multiplyLrBy(real factor)
{
  Weights::multiplyLrBy(factor);
  //fprop in all connector
  for(CI j=lcup.begin();j!=lcup.end();j++)
    (*j)->multiplyLrBy(factor);
}
///
/// set lr
/// \param new_lr new learning rate
///
void Layer::setLr(real new_lr)
{
  start_lr=new_lr;
  //fprop in all connector
  for(CI j=lcup.begin();j!=lcup.end();j++)
    (*j)->start_lr=new_lr;
}
///
/// initialisation at each epoch
/// \param i iteration #
///
void Layer::initEpoch(int i)
{
  Weights::initEpoch(i);
  //init epoch of all connector
  for(CI j=lcup.begin();j!=lcup.end();j++)
      (*j)->initEpoch(i);
}
///
/// set forward and backward propagation of the
/// current layer and all connector link to this Layer.
/// \param do_fprop_ set do_fprop to do_fprop
/// \param do_bprop_ set do_bprop to do_fprop
///
void Layer::setfbprop(bool do_fprop_,bool do_bprop_)
{
	do_fprop=do_fprop_;do_bprop=do_bprop_;
	for(CI i=lcup.begin();i!=lcup.end();i++)
      (*i)->setfbprop(do_fprop_,do_bprop_);
	for(CI j=lcdown.begin();j!=lcdown.end();j++)
		(*j)->setfbprop(do_fprop_,do_bprop_);
}
void Layer::setfbpropDown(bool do_bprop_)
{
	do_bprop=do_bprop_;
	for(CI j=lcdown.begin();j!=lcdown.end();j++)
		(*j)->setfbprop(this->do_fprop,do_bprop_);
}
///
/// get the first connector with this name else return NULL
///
Connector* Layer::getConnector(string name)
{
  for(CI i=lcup.begin();i!=lcup.end();i++)
    if((*i)->name==name)
      return (*i);
  return NULL;
}
///
/// update = apply gradient
/// Important: update is done like an fprop
///
void Layer::update(uint batch_size)
{
  //update bias
  real* pbu=bias_update;
  real* pb=bias;
#ifndef LRWEIGHTS
  for(uint i=0;i<n;i++)
    (*pb++)-=(*pbu++)*(lr /*/batch_size*/);
#endif
#ifdef LRWEIGHTS
  real* plr=lrs;
  for(uint i=0;i<n;i++)
	(*pb++)-=(*pbu++)*(*plr++);
#endif
  //clean bias gradient
  memset(bias_update,0,n*sizeof(real));
  //fprop in all connector
  for(CI j=lcup.begin();j!=lcup.end();j++)
    (*j)->update(batch_size);
}

void Layer::_biasbprop(bool stochastic)
{
	if(stochastic)
      updateOnFly();
    else
      updateGradient();
}
///
/// back propagation on Layer and down connections
/// \param stochastic if true=updateOnFly() else UpdateGradient()
///
/// TODO check why : ./fexp / -e 5 -h 100 --lsm --oh -l 0.01  (very slow)
/// #1  t= 0          Cost = -3.41331e+06  Error = 33.465    %
void Layer::bprop(real target, bool stochastic)
{
  if(do_bprop){
	if (lcdown.size()>0){
		_biasbprop(stochastic);
		//bprop in all connectors
		for(CI i=lcdown.begin();i!=lcdown.end();i++)
			(*i)->bprop(stochastic);
	}
    // bprop all one neurons layers && (this == gtopo->outl)
	/*
	else if  ( (n>1) )//  && ( (gtopo->bprop_type!=STD) || (gtopo->bprop_type==MIN_SENSITIVITY) ) )
	{
    	switch(gtopo->bprop_type){
			case MAX_SENSITIVITY:lneurons.sort(compare_output_sensitivity_max);break;
    		case MIN_SENSITIVITY:lneurons.sort(compare_output_sensitivity_min);break;

    	}
    	for(LI i=lneurons.begin();i!=lneurons.end();i++){
			(*i)->bprop(target, stochastic);//bpropOfConnector
			// TODO: check this
			//gtopo->fprop();
			//gtopo->outl->ComputeOutputSensitivity(target);
    	}
    }
    */
  }
}

void Layer::bpropDeepFirst(bool stochastic, real target)
{
	gtopo->fprop();
	gtopo->outl->ComputeOutputSensitivity(target);

	_biasbprop(stochastic);
	//bprop in all connectors
	for(CI i=lcdown.begin();i!=lcdown.end();i++)
		(*i)->bpropDeepFirst(stochastic, target);
}

///
/// back propagation only on down connections
///
void Layer::bpropOfConnector(bool stochatic)
{
  if(do_bprop){
    //bprop in all connector
    for(CI i=lcdown.begin();i!=lcdown.end();i++)
      (*i)->bprop(stochatic);
  }
}
///
/// get back result output for regression or class # for classification
///
real Layer::result()
{
  if(n>1) {
    // find max
    real *max=p;uint max_i=0;real *po=p;
    for(uint i=0;i<n;i++,po++){
      if (*po>*max){max_i=i;max=po;};
    }
    return (max_i);//in classification mode, return class id
  }
  else if(n==1)
    return *p;//in regression mode
  else
    return 0;//computeCost(target) will be call later
}
///
/// display path of forward propagation
///
void Layer::displayfprop(bool all)
{
  if(do_fprop||all){
    if((classname!="Input")&&(classname!="Layer"))
      cout<<name<<":apply ("<<classname<<") <"<<do_fprop<<">"<<endl;
    //fprop in all connector
    for(CI i=lcup.begin();i!=lcup.end();i++)
      (*i)->displayfprop(all);
  }
}
///
///display path of back propagation
///
void Layer::displaybprop(bool all)
{
  if(do_bprop||all){
    cout<<"<bias> "<<name<<" <"<<do_bprop<<">"<<endl;
	//fprop in all connector
    for(CI i=lcdown.begin();i!=lcdown.end();i++)
      (*i)->displaybprop(all);

  }
}
///
///compute hidden sensitivity
///
void Layer::ComputeHiddenSensitivity(Connector *cup)
{
  //sensitivity[]=0
  memset(sensitivity,0,n*sizeof(real));

  if (cup!=NULL)
	_updateHiddenSensitivity(cup);
  else{
 	// for all up connections
  	for(CI i=lcup.begin();i!=lcup.end();i++)
		_updateHiddenSensitivity(*i);
  }
  // TODO generalize
  if ((n>1) && lcup.size()==0){
	  for(LI i=lneurons.begin();i!=lneurons.end();i++)
		  (*i)->ComputeHiddenSensitivity();
	  //printVecReal(sensitivity,n);
  }
  // multiply by derivative
  multiplyByDerivative();
}

// Update Hidden sensitivity for upper connection
void Layer::_updateHiddenSensitivity(Connector * cup)
{
	//compute output sensitivity
	real *pls=cup->outl->sensitivity;
	real *plw=cup->weights;
	real *ps=sensitivity;
	for(uint k=0;k<cup->outl->n;k++,pls++,ps=sensitivity){//same as outputs
		for(uint j=0;j<n;j++){//same as hidden
		//sensitivity_[j]+=layer.weights_[k][j]*layer.sensitivity_[k];
		(*ps++)+=*(plw++)*(*pls);
		}
	}
}
///
///compute cost depending on target
/// \param target the answer
///
real Layer::computeCost(real target)
{
  real *ptarg=targs;
  real *pout=p;
  uint i;
  real out_targ=0; //output-target
  //set target
  if(n>1)
    targs[(int)target]=max_targ;
  else targs[0]=target;
  // (ouputs-targets)
  for(i=0;i<n;i++)
    out_targ+= *pout++ - *ptarg++;
  // unset classification target
  if(n>1)
    targs[(int)target]=min_targ;
  return out_targ;
}
///
///constructor
/// WARNING init_outl should be false if you had connector during training
///
Connector::Connector(Layer *in,Layer *out,real lr_,real dc_,bool init_outl):Weights(in->name+"."+out->name,in->n*out->n,lr_,dc_)
{
  classname="Connector";
  inl=in;outl=out;
  inl->addUpConnection(this);
  outl->lcdown.push_back(this);
  if(STDOUT)cout<<"Creating Connector ["<<in->n<<"|"<<out->n<<"] ["<<in->name<<" | "<<out->name<<"]"<<endl;
  weights=w;
  weights_updates=wu;
  randomInit((real)(1.0/sqrt((double)inl->n)));
  if(init_outl)outl->randomInit((real)(1.0/sqrt((double)inl->n)));
}
///
/// forward propagation
/// warning: you should init outputs[]->0
///
void Connector::fprop()
{
  if(do_fprop){
    real *pi=inl->p;
    real *pon=outl->p;
    real *pw=weights;
    uint j,i;
    for(i=0;i<outl->n;i++,pon++,pi=inl->p)
		for(j=0;j<inl->n;j++)*pon+=(*pi++) * (*pw++);
    //cout<<"outputs_nets: "<<outputs_net<<" ";printVecReal(outputs_net,n_outputs_);
  }
}
///
///backward propagation
///
void Connector::bprop(bool stochastic)
{
  if(do_bprop){
    if(stochastic)
      updateOnFly();
    else
      updateGradient();
  }
}

void Connector::bpropDeepFirst(bool stochastic, real target)
{
  if(do_bprop){
	gtopo->fprop();
	gtopo->outl->ComputeOutputSensitivity(target);
	bprop(stochastic);

	//TODO: investigate; inl->cup=this;
	if (gtopo->force_all_upper_connections)
		inl->ComputeHiddenSensitivity();
	else // IMPORTANT: compute sensitivity from current up connection
		inl->ComputeHiddenSensitivity(this);

	inl->bpropDeepFirst(stochastic,target);
  }
}

///
///update "on fly"
///
void Connector::updateOnFly()
{
  if(do_bprop){
    if(lr<=0)
      FERR("WARNING-> lr<=0 ******************************");
    real *pw=weights;
    real *pi=inl->p;
    real *ps=outl->sensitivity;

    uint i,j;
    real psXlr=0;//optimisation ->ps*lr
#ifndef LRWEIGHTS
    //weights update
    for(i=0;i<outl->n;i++,ps++,pi=inl->p){
      psXlr=(*ps)*lr;
      for(j=0;j<inl->n;j++)(*pw++) -=(*pi++) * psXlr;
	}
#endif
#ifdef LRWEIGHTS
	real* plr=lrs;
	for(i=0;i<outl->n;i++,ps++,pi=inl->p)
      for(j=0;j<inl->n;j++)(*pw++) -=(*pi++) * (*ps)*(*plr++);
#endif
  }
}
///
///update gradient
///
void Connector::updateGradient()
{
  if(do_bprop){
    if(lr<=0)
      FERR("WARNING-> lr<=0 ******************************");
    real *pwu=weights_updates;
    real *pi=inl->p;
    real *ps=outl->sensitivity;
    uint i,j;
    //weight_update
    for(i=0;i<outl->n;i++,ps++,pi=inl->p)
      for(j=0;j<inl->n;j++)(*pwu++) +=(*pi++) * (*ps);
  }
}
///
///update weights directly
///
void Connector::update(uint batch_size)
{
  if(do_bprop){
    //backprop error on weights
    real *pwu=weights_updates;
    real *pw=weights;
#ifndef LRWEIGHTS
	for(uint i=0;i<n;i++)//weights_[i]-=weights_updates_[i]*learning_rate_;
      (*pw++)-=(*pwu++)*(lr/* /batch_size*/);
#endif
#ifdef LRWEIGHTS
	real* plr=lrs;
    for(uint i=0;i<n;i++)//weights_[i]-=weights_updates_[i]*learning_rate_;
      //(*pw++)-=(*pwu++)*(lr/* /batch_size*/);
	  (*pw++)-=(*pwu++)*(*plr++/* /batch_size*/);
#endif
    //clear all gradient
    memset(weights_updates,0,outl->n*inl->n*sizeof(real));
  }
}
///
/// << operator
///
ostream& operator<<(ostream& out,Layer& layer)
{
  out<<"\nLayer"<<endl;
  out<<"connection point:";printVecReal(layer.p,layer.n,true,10,&out);
  out<<"bias            :";printVecReal(layer.bias,layer.n,true,10,&out);
  return out;
}
///
/// display forward propagation
///
void Connector::displayfprop(bool all)
{
  if(do_fprop||all)
    cout<<inl->name<<" -> "<<outl->name<<"<"<<do_fprop<<">"<<endl;
}
///
///display backward propagation
///
void Connector::displaybprop(bool all)
{
  if(do_bprop||all)
    cout<<"<weights> "<<outl->name<<" -> "<<inl->name<<"<"<<do_bprop<<">"<<endl;
}

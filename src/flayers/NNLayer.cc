#include "NNLayer.h"


using namespace std;


Layer* LayerFactory(string con_name,string name,uint size,real lr,real dc)
{
  if(con_name=="SigmMSE")
    return new SigmMSE(name,size,lr,dc);
  else if(con_name=="TanhMSE")
    return new TanhMSE(name,size,lr,dc);
  else if(con_name=="TanhMSE2")
    return new TanhMSE2(name,size,lr,dc);
  else if(con_name=="NonSatTanhMSE")
    return new NonSatTanhMSE(name,size,lr,dc);
  else if(con_name=="LogSoftMax")
    return new LogSoftMax(name,size,lr,dc);
  else if(con_name=="Linear")
    return new Linear(name,size,lr,dc);
  else if(con_name=="Input")
    return new Input(name,size,lr,dc); 
  else FERR(("undefine Active Function ->"+con_name).c_str());
  return 0;
}

Layer* FromLayerFactory(string con_name,Layer* from,uint start,uint size)
{
  if(con_name=="SigmMSE")
    return new SigmMSE(from,start,size);
  else if(con_name=="TanhMSE")
    return new TanhMSE(from,start,size);
  else if(con_name=="TanhMSE2")
    return new TanhMSE2(from,start,size);
  else if(con_name=="NonSatTanhMSE")
    return new NonSatTanhMSE(from,start,size);
  else if(con_name=="LogSoftMax")
    return new LogSoftMax(from,start,size);
  else if(con_name=="Linear")
    return new Linear(from,start,size);
  else if(con_name=="Input")
    return new Input(from,start,size); 
  else FERR(("undefine Active Function ->"+con_name).c_str());
  return 0;
}

real SigmMSE::computeCost(real target)
{
  uint i;
  real out_targ=0; //output-target

  //set target
  if(n>1)
    targs[(int)target]=max_targ;
  else targs[0]=target;

  // pow((ouputs-targets),2)
  out_targ = sqr(p - targ).sum();

  // unset classification target
  if(n>1)
    targs[(int)target]=min_targ;
  return out_targ;
}

void LogSoftMax::apply()
{
    // ??? Are we sure ???
    Layer::apply();
    
    real max = p.max();
    p -= max;
    real logsum = log(exp(p).total());
    p -= logsum;
}

real LogSoftMax::computeCost(real target)
{  
    uint i;
    real nll=0;//negativ log likelyhood 
    //set target
    
    if(n>1)
        targs[(int)target]=max_targ;
    else targs[0]=target;
    
    // ??? Are we sure ???
    nll -= targs * -p - (p - targs) * -(p - 1.0f);
    
    // unset classification target
    if(n>1)
        targs[(int)target]=min_targ;
    
    return (nll);
}

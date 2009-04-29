/*                                                                 -*- C++ -*- */
#ifndef WEIGHTS_H
#define WEIGHTS_H

#include "fLayersGeneral.h" 

///  \class Weights Params.h "Core/Params.h" 
///  \brief Weights represent a set of parameters. 
class Weights
{
public:
  Weights(std::string name_,uint size,real lr_,real dc_);
  //name representing this set of parameters
  std::string name;
  //name of the class
  std::string classname;
  //nb of parameters
  uint n;
  //pointer to the parameters array
  real *w;
  //pointer to updates (gradient)
  real *wu;
  //start learning rate
  real start_lr;
  //learning rate
  real lr;
  //decrease constant
  real dc;
  ///backprop activation
  bool do_bprop;
  ///fowardprop activation
  bool do_fprop;
  virtual void setfbprop(bool do_fprop_,bool do_bprop_)
    {do_fprop=do_fprop_;do_bprop=do_bprop_;};
  void randomInit(real max);
  void clearUpdates(){memset(wu,0,n*sizeof(real));};//set updates to 0
  void perturb(real fraction,bool extremum);
  Weights* clone(){Weights *cw=new Weights(name,n,lr,dc);memcpy(w,cw->w,n*sizeof(real));return cw;};
  void copy(Weights *new_w){memcpy(w,new_w,n*sizeof(real));};
  virtual void multiplyLrBy(real factor){start_lr*=factor;};
  virtual void initEpoch(int i){lr=start_lr/(1+dc*i);};
  virtual ~Weights(){free(w);free(wu);};
};
std::ostream& operator<<(std::ostream& out,Weights& w);

///  \class LrWeights Weights.h "Core/Weights.h" 
///  \brief LrWeights represent a set of parameters with a specific lr for each parameters.
///
/// WARNING: if you want to use this class change Layer class hierarchy to Layer: public LrWeights

class LrWeights : public Weights
{
 public:
  LrWeights(std::string name_,uint size,real lr_,real dc_);
  real *old_weights_updates; 
  real *old_weights;
  real *lrs;
  real inc_factor;
  real dec_factor;
  void updateParamsLr();
  void memAlloc();
  void memFree();  
  virtual void initEpoch(int i);
  virtual ~LrWeights();
};
#endif

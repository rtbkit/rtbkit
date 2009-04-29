/*                                                                 -*- C++ -*- */
#ifndef LAYER_H
#define LAYER_H
//#define LRWEIGHTS
#include "fLayersGeneral.h"
#include "WeightsList.h"
#include "Weights.h"
#include "stats/distribution.h"

class Connector;//forward class declaration
class Topology;
///
///  \class Layer Layer.h "Core/Layer.h"
///  \brief Layer class is the abstract representation of the input/output/hiddenLayer of a Neural Network.
///
/// Layer is the Core class of fLayers2 neural network library
/// A Layer can be the input Layer (Input), the hidden Layers (SigmMSE,TanhMSE)
/// or the output Layer (SigmMSE,LogSoftMax,TanhMSE).See also NNLayer.h
/// author: Francis Pieraut; begin in july 2002
class Layer: public Weights
{
 public:
  Layer(std::string name_,uint size,real lr_,real dc_);
  Layer(Layer *from,uint start,uint size);
  // list of single neurons layer
  std::list<Layer*> lneurons;
  Layer** neurons;
  ///list connector to up;usfull for input layer;the list size should be equal to 1 if it is a hidden layer
  std::list<Connector*> lcup;
  ///list connector down
  std::list<Connector*> lcdown;
  ///Layer connection point
  ML::distribution<float> p;
  ///targets
  ML::distribution<float> targs;
  ///output sensitivity
  ML::distribution<float> sensitivity;
  ///bias
  ML::distribution<float> bias;
  ///bias updates (batch mode)
  ML::distribution<float> bias_update;
  ///min target value
  real min_targ;
  ///max target value
  real max_targ;
  ///usefull in Correlation
  real mean_cost;
  //General functions to overwrite
  virtual void fprop();
  virtual void update(uint batch_size);
  void _biasbprop(bool stochastic);
  virtual void bprop(real target, bool stochastic);
  virtual void bpropDeepFirst(bool stochastic,real target);
  virtual void bpropOfConnector(bool stochastic);
  virtual void displayfprop(bool all=false);
  virtual void displaybprop(bool all=false);
#ifndef LRWEIGHTS
  virtual void updateOnFly(){ bias -= sensitivity * lr;};
#endif
#ifdef LRWEIGHTS
  virtual void updateOnFly(){real *plr=lrs;real *ps=sensitivity;real* pb=bias;for(uint i=0;i<n;i++)(*pb++) -=(*ps++)* (*plr++);};
#endif
  virtual void updateGradient(){bias_update += sensitivity;}
  virtual void initEpoch(int i);
  virtual void multiplyLrBy(real factor);
  virtual void setLr(real new_lr);
  //specific functions
  real result();
  void addUpConnection(Connector* cup);
  void clearLayer(){p = 0;}
  void clearBias(){bias = 0;}
  void ComputeOutputSensitivity(real target){PreComputeOutputSensitivity(target);ComputeOutputSensitivity();PostComputeOutputSensitivity(target);};
    void PreComputeOutputSensitivity(real target){sensitivity = 0;  if (n > 1) targs[(int)target] = max_targ;  else targs[0] = target;};
  void PostComputeOutputSensitivity(real target){if(n>1)targs[(int)target]=min_targ;};
  void ComputeHiddenSensitivity(Connector *cup=NULL);
  void _updateHiddenSensitivity(Connector * cup);
  void setTargets(){targs = min_targ;};
  void setMinMax(real min,real max){min_targ=min;max_targ=max;setTargets();};
  ///WARNING: you should call Layer::apply() first ->BECAUSE you need to add the bias and others thing depending on the heritage.
  virtual void apply(){if(do_fprop){p += bias;}}
  virtual void multiplyByDerivative(){};
  virtual void ComputeOutputSensitivity(){};
  virtual void setfbprop(bool do_fprop_,bool do_bprop_);
  virtual void setfbpropDown(bool do_bprop_);
  virtual real computeCost(real target);
  Connector* getConnector(std::string name);
};
bool compare_output_sensitivity_max(const Layer *a,const Layer *b);
bool compare_output_sensitivity_min(const Layer *a,const Layer *b);

bool compare_output_sensitivity(const Layer *a,const Layer *b);
std::ostream& operator<<(std::ostream& out,Layer& layer);

///  \class Connector Connector.h "Core/Connector.h"
///  \brief a Connector is use to link 2 Layers.
class Connector: public Weights
{
 public:
  Connector(Layer *in,Layer *out,real lr=0.1,real dc=0,bool init_outl=true);
  ///input Layer
  Layer *inl;
  ///output Layer
  Layer *outl;
  ///the weights
  real *weights;
  ///gradient of each weights
  real *weights_updates;
  virtual void fprop();
  virtual void update(uint batch_size);
  virtual void bpropDeepFirst(bool stochastic, real target);
  virtual void bprop(bool stochastic);
  virtual void displayfprop(bool all=false);
  virtual void displaybprop(bool all=true);
  virtual void updateOnFly();
  virtual void updateGradient();
  void clearWeights(){memset(weights,0,outl->n*inl->n*sizeof(real));};
  virtual Connector* clone(){return new Connector(inl,outl,lr,dc);};
  virtual ~Connector(){weights=NULL;weights_updates=NULL;};
};
typedef std::list<Layer*>::iterator LI;
typedef std::list<Layer*>::reverse_iterator RLI;
typedef std::list<Connector*>::iterator CI;
typedef std::list<Connector*>::reverse_iterator RCI;

#endif

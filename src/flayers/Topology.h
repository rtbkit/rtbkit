/*                                                                 -*- C++ -*- */
#ifndef TOPO_H
#define TOPO_H

#include "fLayersGeneral.h"
#include "Layer.h"
#include "WeightsList.h"
///
///  \class Topology Topology.h "Core/Topology.h"
///  \brief Topology is Layers link by Connectors.
///
/// class Topology (one of the Core class of FLayers Neural Network Learning library)
/// A Topology define a series of Layers link together by Connectors.
///

// TODO add n min or max neurones
enum BpropType {STD, DEEP_FIRST, MAX_SENSITIVITY, MIN_SENSITIVITY};

class Topology
{
 public:
  Topology(Layer *in,Layer *out,bool adapt_lr=false,
		  real inc=1.1,real decr=0.7,
		  BpropType bprop_type=STD,bool force_all_upper_connections=false);
  ///input Layer
  Layer *inl;
  ///output Layer
  Layer *outl;
  std::ofstream params_out;
  ///contain a reference(pointers) to all parameters
  WeightsList params_list;
  ///middle Layers (incl and outl are not and SHOULD NOT be included in that list)
    std::list<Layer*>llayers;
  ///last fprop result
  real result;
  bool adapt_lr;
  real inc_lr_factor;
  real decr_lr_factor;
  ///use in chagetLr()
  real last_cost;
  ///start increase lr after start_inc decrease of cost
  real start_inc;
  ///counter use in adaptlr
  uint count;
  ///same has trainer->cost
  real cost;
  // type of bprop to apply
  BpropType bprop_type;
  // in first deep, force all upper connection
  bool force_all_upper_connections;

  void addToBack(Layer *layer){llayers.push_back(layer);};
  void addToFront(Layer *layer){llayers.push_front(layer);};
  void multiplyLrBy(real factor);
  void setLr(real new_lr);
  void adaptLr(int i);
  virtual real fprop(bool apply=true);
  virtual void displayfprop(bool all=false);
  virtual void displaybprop(bool all=false);
  virtual void bprop(real target, bool stochastic=true);
  virtual void stdBprop(real target, bool stochastic=true);
  virtual void deepFirstBprop(real target, bool stochastic);
  virtual void sortSensitivityBprop(real target, bool stochastic, bool max=true);
  virtual void update(uint batch_size);
  virtual void fillUp(bool in2out=true);
  virtual void setfbprop(bool do_fprop=true, bool do_bprop=true);
  void saveParams(std::ostream & out){out<<params_list;};
  void loadParams(std::istream & in){in>>params_list;};
  void openParamsFile(std::string name, uint n_epochs, bool octave=true);
  ///this fct is call before each epochs training; when overwrite, call it first
  void initEpoch(int i);
  virtual Connector* getConnector(std::string name);
  Layer *getLayer(std::string name);
  virtual ~Topology(){llayers.clear();params_list.cleanup();};
  };
typedef std::list<Topology*>::iterator TI;
#endif

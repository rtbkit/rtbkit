/*                                                                 -*- C++ -*- */
#ifndef CLASS_TOPO_H
#define CLASS_TOPO_H

#include "Topology.h"
#include "NNLayer.h"

///
///  \class ClassTopology ClassTopology.h "Core/ClassTopology.h" 
///  \brief ClassTopology where you have a Layer for each class (output).
///
class ClassTopology: public Topology
{
 public:
  ClassTopology(Layer *in,Layer *out);
  /// inside output layer
  Layer** class_i;
  real bprop_criterion;
  char bprop_type;///backpropagation type [o=one output at a time;c=only <criterion;else Topology->bprop()]
  void bpropOneOutputAtaTime(real target,bool stochastic);
  void bpropCriterion(real target,bool stochastic);
  void bpropOneHiddenAtaTime(real target,bool stochastic);
  virtual void displayfprop(bool all = false);
  virtual void displaybprop(bool all = false);
  virtual void fillUp(bool all){};
  virtual void bprop(real target,bool stochastic);
  virtual void update(uint batch_size);
  virtual void setfbprop(bool do_fprop,bool do_bprop);
  virtual ~ClassTopology(){};
  };
typedef std::list<Topology*>::iterator TI;
#endif


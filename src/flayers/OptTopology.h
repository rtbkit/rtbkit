#ifndef CLASS_OPTTOPO_H
#define CLASS_OPTTOPO_H

#include "Topology.h"
#include "NNLayer.h"

///
///  \class SimTopology
///  \brief  Abstract class that insure simulation of std back_prop
///

class SimTopology: public Topology
{
 public:
  SimTopology(Layer *in,Layer *out, bool sim_std=false, char bprop_type=true):Topology(in,out)
{ this->sim_std=sim_std;this->bprop_type=bprop_type;}
 bool sim_std;
 char bprop_type;
};

///
///  \class OptHiddenTopology
///  \brief bprop one hidden neuron at the time
///

class OptHiddenTopology: public SimTopology
{
 public:
  OptHiddenTopology(Layer *in,Layer *out, uint n_hiddens, std::string hidden_cost_type,real lr, real dc, bool sim_std);
  virtual void bprop(real target,bool stochastic);
  virtual ~OptHiddenTopology(){};
  };

///
///  \class OptOutputTopology
///  \brief bprop one output neuron at the time (n_output slower)
///

class OptOutputTopology: public SimTopology
{
 public:
  OptOutputTopology(Layer *in,Layer *out, uint n_hiddens, std::string hidden_cost_type,real lr, real dc, bool sim_std, char bprop_type='1');
  /// inside output layer
  Layer** output_i;
  std::list<Layer*> outputs;
  Layer *hiddens;
  //virtual void _bprop(real target, bool stochastic);
  virtual void bprop(real target,bool stochastic);
  virtual ~OptOutputTopology(){};
  };

///
///  \class OptTopology
///  \brief bprop one output & hidden neuron at the time (n_output*n_hidden slower)
///
class OptTopology: public OptOutputTopology
{
 public:
  OptTopology(Layer *in,Layer *out, uint n_hiddens, std::string hidden_cost_type,real lr, real dc, bool sim_std, char bprop_type='1');
  Layer** hidden_i;
  virtual real fprop(bool apply=true);
  //virtual void _bprop(real target, bool stochastic);
  virtual ~OptTopology(){};
  };


///
///  \class XTopology; input/hidden/output are split in 1 neuron
///  \brief use output_i[i]->bpropDeepFirst()
///
class XTopology: public SimTopology
{
   public:
    XTopology(Layer *in,Layer *out, uint n_hiddens, std::string hidden_cost_type,real lr, real dc, bool sim_std);
    /// inside output layer
    Layer** output_i;
    Layer** hidden_i;
    Layer** input_i;
    Layer *hiddens;
    virtual real fprop(bool apply=true);
    virtual void bprop(real target,bool stochastic);
    virtual ~XTopology(){};
 };

///
///  \class ZTopology; hidden/output are split in 1 neuron (same as XTopology expect imput split)
///  \brief use output_i[i]->bpropDeepFirst()
///
class ZTopology: public SimTopology
{
   public:
    ZTopology(Layer *in,Layer *out, uint n_hiddens, std::string hidden_cost_type,real lr, real dc, bool sim_std);
    /// inside output layer
    Layer** output_i;
    Layer** hidden_i;
    Layer *hiddens;
	virtual real fprop(bool apply=true);
    virtual void bprop(real target,bool stochastic);
    virtual ~ZTopology(){};
 };

///
///  \class WTopology; optimize 1 parameters at a time (output->input) bias before weights
///  \brief use output_i[i]->bpropDeepFirst()
///
// backprop on 1 parameter at the time (Hack bprop to simulate 1 update at a time)
// params order is :
// 1) output->bias;
// 2) hidden->output weights;
// 3) hidden->bias
// 4) input->hidden- weights;
//
// Idea: 1) for all parameters
//       2) copy all weight before
//       3) applying bprop
//       4) restore except i

  // This is extremely cpu intensif (~ 1 iter/hrs)
class WTopology: public SimTopology
{
   public:
    WTopology(Layer *in,Layer *out, uint n_hiddens, std::string hidden_cost_type,real lr, real dc, bool sim_std);
    // simulate std bprop
    bool sim_std;
    real *weights;
    virtual void bprop(real target,bool stochastic);
    virtual ~WTopology(){free(weights);};
 };

typedef std::list<Topology*>::iterator TI;
#endif

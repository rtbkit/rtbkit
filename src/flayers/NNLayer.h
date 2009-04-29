/*                                                                 -*- C++ -*- */
#ifndef NNLAYER_H                                                  
#define NNLAYER_H

#include "Layer.h"
#include <math.h>
#include "stats/distribution_ops.h"

class SigmMSE: public Layer {
public:
    SigmMSE(std::string name_,uint size_,real lr_,real dc_)
        : Layer(name_,size_,lr_,dc_)
    {
        classname="SigmMSE";
    }

    SigmMSE(Layer *from,uint start,uint size)
        : Layer(from,start,size)
    {
    }

    virtual void apply()
    {
        Layer::apply();
        p = 1.0f / (exp(-p) + 1.0f);
    }

    virtual void ComputeOutputSensitivity()
    {
        sensitivity = (p - targs) * (p * (1.0f - p));
    }

    virtual void multiplyByDerivative()
    {
        sensitivity *= p * (1.0f - p);
    }

    virtual real computeCost(real target);
};

class TanhMSE: public Layer {
public:
    TanhMSE(std::string name_,uint size_,real lr_,real dc_)
        : Layer(name_,size_,lr_,dc_)
    {
        setMinMax(-1,1);
        classname="TanhMSE";
    }

    TanhMSE(Layer *from,uint start,uint size)
        : Layer(from,start,size)
    {
    }

    virtual void apply()
    {
        Layer::apply();
        p = tanh(p);
    }

    virtual void ComputeOutputSensitivity()
    {
        sensitivity = (p - targs) * -(p * p - 1.0);
    }

    virtual void multiplyByDerivative()
    {
        sensitivity *= -(p * p - 1.0);
    }
};

/// f(x)=tanh(x)+a*x;see tricks of the trade
/// Y. LeCun, L. Bottou, G. Orr and K. Muller: Efficient BackProp, in Orr, G. and Muller K. (Eds), Neural Networks: Tricks of the trade, Springer, 1998
/// Available at http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
/// Accessed 27 April, 2009

class TanhMSE2: public Layer {
public:
    real a;
    TanhMSE2(std::string name_,uint size_,real lr_,real dc_,real a_=0.0001)
        : Layer(name_,size_,lr_,dc_)
    {
        setMinMax(-1,1);
        classname="TanhMSE2";
        a=a_;
    };

    TanhMSE2(Layer *from,uint start,uint size)
        : Layer(from,start,size)
    {
    }
    
    virtual void apply()
    {
        Layer::apply();
        p = tanh(p) + p * a;
    }

    virtual void ComputeOutputSensitivity()
    {
        sensitivity = (p - targs) * -(p * p - 1.0f) + a;
    }

    virtual void multiplyByDerivative()
    {
        sensitivity *= -(p * p - 1.0f) + a;
    }
};

/// non satutation Tanh;see tricks of the trade p 18
class NonSatTanhMSE: public Layer {
public:
    NonSatTanhMSE(std::string name_,uint size_,real lr_,real dc_)
        : Layer(name_,size_,lr_,dc_)
    {
        setMinMax(-1,1);
        classname="NonSatTanhMSE";
    }

    NonSatTanhMSE(Layer *from,uint start,uint size)
        : Layer(from,start,size)
    {
    }

    virtual void apply()
    {
        Layer::apply();
        p = tanh(p * (2.0 / 3.0)) * 1.7159;
    }

    virtual void ComputeOutputSensitivity()
    {
        sensitivity = (p - targs) * (1.7159 * 2.0 / 3.0) * -(p * p - 1.0f);
    }

    virtual void multiplyByDerivative()
    {
        sensitivity *= (p * p - 1.0) * -(1.7159 * 2.0 / 3.0);
    }
};


class LogSoftMax: public Layer {
public:
    LogSoftMax(std::string name_,uint size_,real lr_,real dc_)
        : Layer(name_,size_,lr_,dc_)
    {
        classname="LogSoftMax";
    }

    LogSoftMax(Layer *from,uint start,uint size)
        :Layer(from,start,size)
    {
    }

    virtual void apply();

    virtual void ComputeOutputSensitivity()
    {
        sensitivity = exp(p) - targs;  // ??? exp(p - targs) ???
    }

    virtual void multiplyByDerivative()
    {
        sensitivity /= p;
    }

    virtual real computeCost(real target);
};


class Linear: public Layer {
public:
    Linear(std::string name_,uint size_,real lr_,real dc_)
        :Layer(name_,size_,lr_,dc_)
    {
        classname="Linear";
    }

    Linear(Layer *from,uint start,uint size)
        :Layer(from,start,size)
    {
    }

    virtual void ComputeOutputSensitivity()
    {
        sensitivity = p - targs;
    }
};

/// Input.p pointer is variable and should point into DataSet
class Input: public Layer {
public:
    Input(std::string name_,uint size_,real lr_,real dc_)
        : Layer(name_,size_,lr_,dc_)
    {
        classname="Input";
        do_bprop=false;
    }

    Input(Layer *from,uint start,uint size)
        : Layer(from,start,size)
    {
    }

    virtual void apply(){};//nothing to do
    virtual void ComputeOutputSensitivity(){}
    virtual void multiplyByDerivative(){}
    virtual real computeCost(real target){return ((real)0);};
};

Layer* LayerFactory(std::string con_name,std::string name,uint size_,real lr_,real dc_);
Layer* FromLayerFactory(std::string con_name,Layer* from,uint start,uint size);
std::ostream& operator<<(std::ostream& out,Layer& layer);
#endif

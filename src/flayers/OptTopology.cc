#include "OptTopology.h"
#include "NNLayer.h"
# include "Layer.h"


using namespace std;


OptHiddenTopology::OptHiddenTopology(Layer *in,Layer *out, uint n_hiddens, string hidden_cost_type,real lr,real dc, bool sim_std):SimTopology(in,out,sim_std)
{
	Layer* hidden_i=NULL;
  		for(uint i=0;i<n_hiddens;i++){
			//create hidden Layers
			hidden_i=LayerFactory(hidden_cost_type,"hiddens"+tostring(i),1,lr,dc);
			//add hidden Layer
			addToBack(hidden_i);
			//create Connection
			new Connector(inl,hidden_i,lr,dc);
			new Connector(hidden_i,outl,lr,dc);
		}
}


void OptHiddenTopology::bprop(real target,bool stochastic)
{
        // back propagate on one hidden at a time
	for(LI i=llayers.begin();i!=llayers.end();i++){
		// bprop output
		outl->ComputeOutputSensitivity(target);
        outl->updateOnFly();
		// bprop layer i connections  (bypass layer)
		(*((*i)->lcup.begin()))->bprop(stochastic);
		(*i)->ComputeHiddenSensitivity();
		(*((*i)->lcdown.begin()))->bprop(stochastic);
		this->fprop();
	}
}

OptOutputTopology::OptOutputTopology(Layer *in,Layer *out, uint n_hiddens, string hidden_cost_type,real lr,real dc, bool sim_std, char bprop_type):SimTopology(in,out,sim_std,bprop_type)
{
	// create hidden layer and connect it to input layer
	hiddens=LayerFactory(hidden_cost_type,"hiddens",n_hiddens,lr,dc);
	new Connector(inl,hiddens,lr,dc);
    //add hidden Layer
	addToBack(hiddens);

	// create output uniq layer (reference to real output layer)
	output_i=(Layer**)malloc(out->n*(sizeof(Layer*)));
	for(uint i=0;i<out->n;i++){
		//create Connector
		output_i[i]=new Layer(out,i,1);
		outputs.push_back(output_i[i]);
		new Connector(hiddens,output_i[i],lr,dc);
	  }
}
/*
void OptOutputTopology::_bprop(real target, bool stochastic)
{
	for(LI i=outputs.begin();i!=outputs.end();i++){
		outl->ComputeOutputSensitivity(target);
		(*i)->bpropDeepFirst(stochastic);
		if (sim_std==false)
				fprop();
	}
}*/

// TODO: reintegrate this principe

//bprop 1 output at a time: need only on hidden layer
void OptOutputTopology::bprop(real target,bool stochastic)
{
	// bprop on each output and apply deep first bprop
    if (bprop_type=='1')
		bprop(target,stochastic);

    else if (bprop_type=='s')
    	sortSensitivityBprop(target, stochastic);

	// bprop on expected and error classification output if not expected classification + sorted sensitivity
    else if (bprop_type=='S'){
		if (result!=target)
			sortSensitivityBprop(target, stochastic);
	}
	// bprop on expected and error classification output if not expected classification
	else if (bprop_type=='e'){
		if(result!=target){
			outl->ComputeOutputSensitivity(target);
			output_i[(int)result]->bpropDeepFirst(stochastic, target);
			output_i[(int)target]->bpropDeepFirst(stochastic, target);
		}
	}
}

OptTopology::OptTopology(Layer *in,Layer *out, uint n_hiddens, string hidden_cost_type,real lr,real dc, bool sim_std,char bprop_type):OptOutputTopology(in,out,n_hiddens,hidden_cost_type,lr,dc,sim_std,bprop_type)
{
	// create hidden layer and connect it to input layer
	hiddens=LayerFactory(hidden_cost_type,"hiddens",n_hiddens,lr,dc);
    // create hidden uniq layers (reference to real hidden layer)
	hidden_i=(Layer**)malloc(n_hiddens*(sizeof(Layer*)));
	// create output uniq layer (reference to real output layer)
	output_i=(Layer**)malloc(out->n*(sizeof(Layer*)));

    // create output_i neurones
	for(uint o=0;o<out->n;o++)
		output_i[o]=new Layer(out,o,1);

	// connect everything
	for(uint h=0;h<n_hiddens;h++){
		//create Connector
		hidden_i[h]=new Layer(hiddens,h,1);
		outputs.push_back(output_i[h]);
        new Connector(inl,hidden_i[h],lr,dc);
		for(uint o=0;o<out->n;o++){
			new Connector(hidden_i[h],output_i[o],lr,dc);
		}
	}
}

real OptTopology::fprop(bool apply)
{
	//clean all outputs connections
	outl->clearLayer();
	hiddens->clearLayer();
	inl->fprop();
	hiddens->apply();
	// fprop on all hidden;
	for(uint i=0;i<hiddens->n;i++){
		hidden_i[i]->fprop();
	}
	if(apply)outl->apply();
	result=outl->result();
	return result;
}
/*
void OptTopology::_bprop(real target, bool stochastic)
{
	for(LI i=outputs.begin();i!=outputs.end();i++){
		outl->ComputeOutputSensitivity(target);
		(*i)->_biasbprop(stochastic);
		for(CI conn_it=(*i)->lcdown.begin();conn_it!=(*i)->lcdown.end();conn_it++)
			{
				outl->ComputeOutputSensitivity(target);
				(*conn_it)->bpropDeepFirst(stochastic);
				if (sim_std==false){
					fprop();
				}
			}
	}
}
*/
XTopology::XTopology(Layer *in,Layer *out, uint n_hiddens, string hidden_cost_type,real lr,real dc, bool sim_std):SimTopology(in,out,sim_std)
{
	// create hidden layer and connect it to input layer
	hiddens=LayerFactory(hidden_cost_type,"hiddens",n_hiddens,lr,dc);

	output_i=(Layer**)malloc(out->n*(sizeof(Layer*)));
	for(uint i=0;i<out->n;i++){
		//create Connector
		output_i[i]=new Layer(out,i,1);
	}

	// create hidden uniq layers (reference to real hidden layer)
	hidden_i=(Layer**)malloc(n_hiddens*(sizeof(Layer*)));
	for(uint i=0;i<n_hiddens;i++){
		//create Connector
		hidden_i[i]=new Layer(hiddens,i,1);
	}

	// create input uniq layers (reference to real hidden layer)
	input_i=(Layer**)malloc(in->n*(sizeof(Layer*)));
	for(uint i=0;i<in->n;i++){
		//create Connector
		input_i[i]=new Layer(in,i,1);
	}

	// connect input to hidden
	for(uint i=0;i<in->n;i++){
	    for(uint h=0;h<n_hiddens;h++){
		new Connector(input_i[i],hidden_i[h],lr,dc);
	    }
	}
	// connect hidden to output
	for(uint h=0;h<n_hiddens;h++){
	    for(uint o=0;o<outl->n;o++){
		new Connector(hidden_i[h],output_i[o],lr,dc);
	    }
	}
}

real XTopology::fprop(bool apply)
{
	//clean all outputs connections
	outl->clearLayer();
	hiddens->clearLayer();
	// fprop on all input; conn_ih->fprop();
	for(uint i=0;i<inl->n;i++){
		input_i[i]->fprop();
	}
	hiddens->apply();
	// fprop on all hidden; conn_ho->fprop();
	for(uint i=0;i<hiddens->n;i++){
		hidden_i[i]->fprop();
	}
	if(apply)outl->apply();
	result=outl->result();
	return result;
}
//bprop 1 connection at the time (extremely CPU intensif)
void XTopology::bprop(real target,bool stochastic)
{
    // bprop on each output and apply deep first bprop
	for(uint i=0;i<outl->n;i++){
			output_i[i]->bpropDeepFirst(stochastic, target);
	}
}

// use sim to simulate a std backprop
ZTopology::ZTopology(Layer *in,Layer *out, uint n_hiddens, string hidden_cost_type,real lr,real dc, bool sim_std):SimTopology(in,out,sim_std)
{
	// create hidden layer
	hiddens=LayerFactory(hidden_cost_type,"hiddens",n_hiddens,lr,dc);

	// create output uniq layers (reference to real output layer)
	output_i=(Layer**)malloc(out->n*(sizeof(Layer*)));
	for(uint i=0;i<out->n;i++){
		//create Connector
		output_i[i]=new Layer(out,i,1);
	}

	// create hidden uniq layers (reference to real hidden layer)
	hidden_i=(Layer**)malloc(n_hiddens*(sizeof(Layer*)));
	for(uint i=0;i<n_hiddens;i++){
		//create Connector
		hidden_i[i]=new Layer(hiddens,i,1);
		addToBack(hidden_i[i]);
	}

	// connect input to hidden & hidden and hidden to output
	for(uint h=0;h<n_hiddens;h++){
	    new Connector(inl,hidden_i[h],lr,dc);
	    for(uint o=0;o<outl->n;o++){
	    	new Connector(hidden_i[h],output_i[o],lr,dc);
	    }
	}
}

real ZTopology::fprop(bool apply)
{
	//clean all outputs connections
	outl->clearLayer();
	hiddens->clearLayer();
	inl->fprop();
	hiddens->apply();
	// fprop on all hidden;
	for(uint i=0;i<hiddens->n;i++){
		hidden_i[i]->fprop();
	}
	if(apply)outl->apply();
	result=outl->result();
	return result;
}


//bprop 1 connection at the time (extremely CPU intensif)
void ZTopology::bprop(real target,bool stochastic)
{
  // bprop on each output and apply deep first bprop}
  for(uint i=0;i<outl->n;i++){
		output_i[i]->bpropDeepFirst(stochastic, target);
	}
}

WTopology::WTopology(Layer *in,Layer *out, uint n_hiddens, string hidden_cost_type,real lr,real dc, bool sim_std):SimTopology(in,out,sim_std)
{
      // create a one hidden layer (default)
      // TODO: refactor to allow any type
      Layer *hidden=LayerFactory(hidden_cost_type,"hiddens",n_hiddens,lr,dc);
      //add hidden Layer
      addToBack(hidden);
      //create Connector
      new Connector(this->inl,hidden,lr,dc);
      new Connector(hidden,this->outl,lr,dc);
      fillUp(false);//list_.reverse();
      // allocate memory for weights copy used in bprop
      weights=realMalloc(params_list.size);
}

typedef list<Weights*>::iterator WEIGHTS_LI;

void WTopology::bprop(real target,bool stochastic)
{
	if (sim_std==true)
       Topology::bprop(target, stochastic);
    else
	{
        // temp storage of weight i
        real iw=0;

        for (uint i=0; i<params_list.size; i++){
            fprop();
            // keep all weights
            weights=params_list.copyOfAllWeights(weights);
            Topology::bprop(target, stochastic);
            iw=params_list.get(i);
            // reset all parameters except i
            params_list.copyFrom(weights);
            params_list.set(i,iw);
        }
	}
}

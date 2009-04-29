#include "Topology.h"


using namespace std;


extern bool STDOUT;
Topology::Topology(Layer *in,Layer *out,bool adapt_lr_,
		real inc,real decr,
		BpropType bprop_type,bool force_all_upper_connections)
{
  adapt_lr=adapt_lr_;
  inc_lr_factor=inc;
  decr_lr_factor=decr;
  start_inc=5;
  inl=in;
  outl=out;
  count=0;
  result=-1;
  this->bprop_type = bprop_type;
  this->force_all_upper_connections = force_all_upper_connections;

  if (STDOUT){
	  switch(bprop_type){
		case STD:cout<<"Optimization: Standard"<<endl;break;
		case DEEP_FIRST:cout<<"Optimization: Deep First"<<endl;break;
		case MAX_SENSITIVITY:cout<<"Optimization: MaxSensitivity + Deep First"<<endl;break;
		case MIN_SENSITIVITY:cout<<"Optimization: MinSensitivity + Deep First"<<endl;break;
	  };
  }
  if ((bprop_type!=STD) && (outl->lcdown.size()!=0))
		FERR("output layer shouldn't be connected; only output neurons");
}
Connector* Topology::getConnector(string name)
{
  Connector* con=inl->getConnector(name);
  if(con)return con;
  for(LI i=llayers.begin();i!=llayers.end();i++){
    con=(*i)->getConnector(name);
    if(con)return con;
  }
  return NULL;
}
Layer* Topology::getLayer(string name)
{
  for(LI i=llayers.begin();i!=llayers.end();i++){
    if((*i)->name==name)
      return (*i);
  }
  return NULL;
}
void Topology::setfbprop(bool do_fprop,bool do_bprop)
{
  inl->setfbprop(do_fprop,do_bprop);
  inl->do_bprop=false;
  for(LI i=llayers.begin();i!=llayers.end();i++)
    (*i)->setfbprop(do_fprop,do_bprop);
  for(CI j=outl->lcdown.begin();j!=outl->lcdown.end();j++)
    (*j)->setfbprop(do_fprop,do_bprop);
  outl->setfbprop(do_fprop,do_bprop);
}
void Topology::initEpoch(int i)
{
  //change lr
  if(adapt_lr)
    adaptLr(i);
  if(params_out.is_open())
    params_out<<params_list;
  inl->initEpoch(i);
  for(LI j=llayers.begin();j!=llayers.end();j++)
    (*j)->initEpoch(i);
  outl->initEpoch(i);
}
void Topology::adaptLr(int i)
{
  if (i>1){
    if (cost>last_cost){
      multiplyLrBy(decr_lr_factor);
      cout<<"decrease lr to "<<inl->start_lr;
      cout<<" last ="<<last_cost<<" new ="<<cost<<endl;
      count=0;
    }else{
      count++;
      if (count>start_inc){
        multiplyLrBy(inc_lr_factor);
        cout<<"increase lr to "<<inl->start_lr;
        cout<<" last ="<<last_cost<<" new ="<<cost<<endl;
      }
    }
  }
  last_cost=cost;
}
void Topology::multiplyLrBy(real factor)
{
  inl->multiplyLrBy(factor);
  for(LI j=llayers.begin();j!=llayers.end();j++)
    (*j)->multiplyLrBy(factor);
  outl->multiplyLrBy(factor);
}
void Topology::setLr(real new_lr)
{
  inl->setLr(new_lr);
  for(LI j=llayers.begin();j!=llayers.end();j++)
    (*j)->setLr(new_lr);
  outl->setLr(new_lr);
}
real Topology::fprop(bool apply)
{
  //clean all outputs connections
  outl->clearLayer();
  for(LI i=llayers.begin();i!=llayers.end();i++)
    (*i)->clearLayer();
  //fprop
  inl->fprop();
  for(LI j=llayers.begin();j!=llayers.end();j++){
    (*j)->apply();
    (*j)->fprop();
  }
  if(apply)outl->apply();
  result=outl->result();
  //printVecReal(outl->p,outl->n);
  return result;
}
void Topology::stdBprop(real target, bool stochastic)
{
  outl->bprop(target, stochastic);
  for(RLI i=llayers.rbegin();i!=llayers.rend();i++){
	  (*i)->ComputeHiddenSensitivity();
	  (*i)->bprop(target, stochastic);
  }
}
void Topology::deepFirstBprop(real target, bool stochastic)
{
	for(LI i=outl->lneurons.begin();i!=outl->lneurons.end();i++)
		(*i)->bpropDeepFirst(target, stochastic);
}
void Topology::sortSensitivityBprop(real target, bool stochastic, bool max)
{
	if (max)
		outl->lneurons.sort(compare_output_sensitivity_max);
	else
		outl->lneurons.sort(compare_output_sensitivity_min);
	deepFirstBprop(target, stochastic);
}
void Topology::bprop(real target,bool stochastic)
{
	outl->ComputeOutputSensitivity(target);
	//printVecReal(outl->sensitivity,outl->n);
	switch(bprop_type){
	case DEEP_FIRST:deepFirstBprop(target, stochastic);break;
	//case STD:stdBprop(stochastic);break;
	default: stdBprop(stochastic);
	}
}
void Topology::fillUp(bool in2out)
{
  params_list.cleanup();
  // add inputs bias
  params_list.add(inl);
  for(CI j=inl->lcup.begin();j!=inl->lcup.end();j++)
	// add connections weights
	params_list.add(*j);
  for(LI i=llayers.begin();i!=llayers.end();i++){
    // add hidden bias
    params_list.add(*i);
    for(CI j=(*i)->lcup.begin();j!=(*i)->lcup.end();j++)
	// add hidden up connections weights
	params_list.add(*j);
  }
  params_list.add(outl);
  if (in2out==false)
    params_list.list_.reverse();
}
void Topology::update(uint batch_size)
{
  inl->update(batch_size);
  for(LI i=llayers.begin();i!=llayers.end();i++)
    (*i)->update(batch_size);
  outl->update(batch_size);//update bias
}
void Topology::displayfprop(bool all)
{
  if(all)
	cout<<"WARNING: all mode =don't care about do_fprop and do_bprop"<<endl;
  inl->displayfprop(all);
  //fprop in all connector
  for(LI i=llayers.begin();i!=llayers.end();i++)
    (*i)->displayfprop(all);
  outl->displayfprop(all);
  cout<<endl;
}
void Topology::displaybprop(bool all)
{
  if(all)
	cout<<"WARNING: all mode =don't care about do_fprop and do_bprop"<<endl;
  outl->displaybprop(all);
  //fprop in all connector
  for(RLI i=llayers.rbegin();i!=llayers.rend();i++)
    (*i)->displaybprop(all);
}
void Topology::openParamsFile(string name,uint n_epochs,bool octave)
{
  string fname=name+".params";
  if (octave)fname+=".octave";
  params_out.open(fname.c_str());
  if (octave){
    params_out<<"# Create by FLayer 2"<<endl;
    params_out<<"# name: a"<<endl;
    params_out<<"# type: matrix"<<endl;
    params_out<<"# rows: "<<n_epochs<<endl;
    params_out<<"# columns: "<<params_list.size<<endl;
  }
}

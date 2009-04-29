#include "ClassTopology.h"
#include "NNLayer.h"


using namespace std;


ClassTopology::ClassTopology(Layer *in,Layer *out):Topology(in,out)
{
	bprop_criterion=0.3f;
	bprop_type='?';
	if(outl->classname=="LogSofMax")
		bprop_criterion=log(0.3f);
	class_i=(Layer**)malloc(outl->n*(sizeof(Layer*)));
	for(int i=0;i<(int)outl->n;i++){
		class_i[i]=FromLayerFactory(outl->classname,outl,(uint)i,1);
		//LayerFactory(string con_name,string name,uint size,real lr,real dc)
		//this is not a good idea :addToBack(class_i[i]);
	}
}
void ClassTopology::setfbprop(bool do_fprop,bool do_bprop)
{
  Topology::setfbprop(do_fprop,do_bprop);
  for(int i=0;i<(int)outl->n;i++)
	class_i[i]->setfbprop(do_fprop,do_bprop);
}

void ClassTopology::bprop(real target,bool stochastic)
{
  outl->ComputeOutputSensitivity(target);
  switch(bprop_type)
  {
  case 'c':bpropCriterion(target,stochastic);break;
  case 'o':bpropOneOutputAtaTime(target,stochastic);break;
  case 'h':bpropOneHiddenAtaTime(target,stochastic);break;
  default:Topology::bprop(target,stochastic);break;
  }
}

//bprop 1 output at a time: need only on hidden layer
void ClassTopology::bpropOneOutputAtaTime(real target,bool stochastic)
{

	if(llayers.size()!=1)
	  FERR("ClassTopology::bpropOneOutputAtaTime can't be apply");
	//retreive the only hidden layer
	LI l=llayers.begin();

	// back propagate on each ouput neurone
	for(int i=0;i<(int)outl->n;i++){
		this->fprop();
		outl->ComputeOutputSensitivity(target);
		class_i[i]->bprop(target,stochastic);count++;
		(*l)->ComputeHiddenSensitivity();
		(*l)->bprop(target,stochastic);
	}
}

//bprop 1 hidden at a time
void ClassTopology::bpropOneHiddenAtaTime(real target,bool stochastic)
{
	// back propagate on one hidden at a time
	for(LI i=llayers.begin();i!=llayers.end();i++){
		outl->setfbprop(true,false);
		(*i)->setfbprop(true,true);
		this->fprop();
		outl->ComputeOutputSensitivity(target);
		outl->bprop(target,stochastic);
		(*i)->bprop(target,stochastic);count++;
		(*i)->ComputeHiddenSensitivity();
		(*i)->bprop(target,stochastic);
	}
}
void ClassTopology::bpropCriterion(real target,bool stochastic)
{
  static uint z=1;z++;
  static uint count=0;
  if(llayers.size()!=outl->n)
	  FERR("ClassTopology::bpropCriterion can't be apply");
  //cout<<target<<"-------------------------------------------"<<endl;
  //printVecReal(outl->p,outl->n);
  //outl->bprop(stochastic);//for bias
  // IMPORTANT back propagate on appropriate outputs
  // if wrong class, be sure to backpropagate on it
  // and back propagate on output>bprop_criterion
  LI l=llayers.begin();
  for(int i=0;i<(int)outl->n;i++){
	 if((outl->p[i]>=bprop_criterion)||((i==(int)result)&&(result!=target))){
		//class_i[i]->bpropOfConnector(stochastic);count++;
		class_i[i]->bprop(target,stochastic);count++;
		(*l)->ComputeHiddenSensitivity();
		(*l)->bprop(target,stochastic);
	 }// else be sure to not backproppagate on the this input layer
	 l++;
  }
  //this->displaybprop();
  if((z%20000)==0){z=0;
	  cout<<"average :"<<(real)count/(real)20000<<" ";count=0;
  }
}/*
void ClassTopology::bprop(real target,bool stochastic)
{
  outl->ComputeOutputSensitivity(target);
  outl->bprop(stochastic);//for bias
  for(int i=0;i<outl->n;i++)
	class_i[i]->bpropOfConnector(stochastic);
  for(RLI j=llayers.rbegin();j!=llayers.rend();j++){
    //intermediate connector should only have one connector up
    (*j)->ComputeHiddenSensitivity();
    (*j)->bprop(stochastic);
  }
}*/
void ClassTopology::update(uint batch_size)
{
	Topology::update(batch_size);
	for(int i=0;i<(int)outl->n;i++)
		class_i[i]->update(batch_size);
}
void ClassTopology::displayfprop(bool all)
{
    inl->displayfprop(all);
    //fprop in all connector
    for(LI i=llayers.begin();i!=llayers.end();i++)
        (*i)->displayfprop(all);
    for(int j=0;j<(int)outl->n;j++)
        class_i[j]->displayfprop(all);
    outl->displayfprop(all);
    cout<<endl;
}
void ClassTopology::displaybprop(bool all)
{
    outl->displaybprop(all);
    for(int i=0;i<(int)outl->n;i++)
        class_i[i]->displaybprop(all);
    //fprop in all connector
    for(RLI j=llayers.rbegin();j!=llayers.rend();j++)
        (*j)->displaybprop(all);
}


#include "WeightsList.h"
#include "Layer.h"
#include <cmath>


using namespace std;


bool WeightsList::set(uint i, real val)
{
  WEIGHTS_LI LI=list_.begin();
  uint size=0;
  uint pos=0;
  while(LI !=list_.end()){
    if (i<(size+(*LI)->n)){
      pos=i-size;
      //cout<<"set params #"<<i<<"( "<<(*LI)->name<<") "<<*((*LI)->w+pos)<<"->"<<val<<endl;
      *((*LI)->w+pos)=val;
      return true;
    }
    size+=(*LI)->n;
    LI++;
  }
  return false;
}

real WeightsList::get(uint i)
{
  WEIGHTS_LI LI=list_.begin();
  uint size=0;
  uint pos=0;
  while(LI !=list_.end()){
    if (i<(size+(*LI)->n)){
      pos=i-size;
      //cout<<"set params #"<<i<<"( "<<(*LI)->name<<") "<<*((*LI)->w+pos)<<"->"<<val<<endl;
      return *((*LI)->w+pos);
    }
    size+=(*LI)->n;
    LI++;
  }
  return 0;
}

void WeightsList::cleanup()
{
  WEIGHTS_LI LI=list_.begin();
  while(LI !=list_.end()){
    list_.remove(*LI++);
  }
  size=0;
}

void WeightsList::clearUpdates()
{
  WEIGHTS_LI LI=list_.begin();
  while(LI !=list_.end()){
    (*LI++)->clearUpdates();
  }
}

real* WeightsList::copyOfAllWeights(real* cpy_to)
{
  if (cpy_to==NULL)
    cpy_to=realMalloc(size);
  real* p=cpy_to;
  WEIGHTS_LI LI=list_.begin();
  Weights *pw=NULL;
  while(LI !=list_.end()){
    pw=*LI;
    memcpy(p,pw->w,pw->n*sizeof(real));
    p+=pw->n;
    LI++;
  }
  return cpy_to;
}
ostream& operator<<(ostream& out,WeightsList& wl)
{
  WEIGHTS_LI LI=wl.list_.begin();
  Weights *pw=NULL;
  while(LI !=wl.list_.end()){
    pw=*LI;
    out<<pw->name<<endl;
    printVecReal(pw->w,pw->n,true,10,&out);
    LI++;
  }
  out<<endl;
  return out;
}
istream& operator>>(istream& in,WeightsList& wl)
{
  WEIGHTS_LI LI=wl.list_.begin();
  Weights *pw=NULL;
  while(LI !=wl.list_.end()){
    pw=*LI;
    in>>pw->name;
    readVecReal(pw->w,pw->n,&in);
    LI++;
  }
  return in;
}
void WeightsList::copyFrom(real* source)
{
  real* p=source;
  WEIGHTS_LI LI=list_.begin();
  Weights *pw=NULL;
  while(LI !=list_.end()){
    pw=*LI;
    memcpy(pw->w,p,pw->n*sizeof(real));
    p+=pw->n;
    LI++;
  }
}
real WeightsList::sumAbs()
{
  real sumabs=0;
  real *p=NULL;
  WEIGHTS_LI LI=list_.begin();
  Weights *pw=NULL;
  while(LI !=list_.end()){
    pw=*LI;
    p=pw->w;
    for(uint i=0;i<pw->n;i++)
        sumabs+=std::abs(*p++);
    LI++;
  }
  return sumabs;
}
void WeightsList::randomInit(real max)
{
  WEIGHTS_LI LI=list_.begin();
  while(LI !=list_.end()){
    (*LI++)->randomInit(max);
  }
}
void WeightsList::set(real r)
{
  real *p=NULL;
  WEIGHTS_LI LI=list_.begin();
  Weights *pw=NULL;
  while(LI !=list_.end()){
    pw=*LI;
    p=pw->w;
    for(uint i=0;i<pw->n;i++)
      *p++=r;
    LI++;
  }
}
bool WeightsList::isSame(WeightsList *pw)
{
  if ((size!=pw->size)||(list_.size()!=pw->list_.size()))
    FERR("WeightsList::isSame(WeightsList *pw) ; not the same size");
  real *lp=NULL;//local pointer
  WEIGHTS_LI LLI=list_.begin();//local list iterator
  real *cp=NULL;//compare pointer
  WEIGHTS_LI CLI=list_.begin();//compare list iterator
  Weights *lpw=NULL;Weights *cpw=NULL;
  while(LLI !=list_.end()){
    lpw=*LLI;cpw=*CLI;
    lp=lpw->w;cp=cpw->w;
    if (lpw->n!=cpw->n)
      return false;
    for(uint i=0;i<lpw->n;i++)
      if (*lp++!=*cp++)
	return false;
    LLI++;CLI++;
  }
  return true;
}
void WeightsList::copyFrom(WeightsList *pw)
{
  if ((size!=pw->size)||(list_.size()!=pw->list_.size()))
    FERR("WeightsList::copyFrom(WeightsList *pw) ; not the same size");
  WEIGHTS_LI SLI=list_.begin();//source list iterator
  WEIGHTS_LI DLI=list_.begin();//destination list iterator
  Weights *spw=NULL;
  Weights *dpw=NULL;
  while(SLI !=list_.end()){
    spw=*SLI;
    dpw=*DLI;
    memcpy(dpw->w,spw->w,dpw->n*sizeof(real));
    SLI++;
    DLI++;
  }
}
void WeightsList::perturb(real fraction,bool extremum)
{
  WEIGHTS_LI LI=list_.begin();
  while(LI !=list_.end())
	  (*LI++)->perturb(fraction,extremum);
}
void WeightsList::displayNameList()
{
  cout<<"WeightsList <"<<size<<">"<<endl;
  WEIGHTS_LI LI=list_.begin();
  while(LI !=list_.end())
	  cout<<(*LI)->name<<" <"<<(*LI++)->n<<">"<<endl;
}
//--------------------------------------------------------------------------------
//fillup params list
//--------------------------------------------------------------------------------
void WeightsList::add(Weights *w)
{
	list_.push_back(w);
	size+=w->n;
}

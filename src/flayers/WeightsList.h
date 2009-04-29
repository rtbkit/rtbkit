/*                                                                 -*- C++ -*- */
#ifndef WEIGHTSLIST_H
#define WEIGHTSLIST_H

#include "fLayersGeneral.h"
#include "Weights.h"

typedef std::list<Weights*>::iterator WEIGHTS_LI;
///  \class WeightsList WeightsList.h "Core/WeightsList.h" 
///  \brief WeightsList represent a list of objects Weights. 
class WeightsList
{
 public:
  WeightsList(){size=0;};
  uint size;
  std::list<Weights*> list_;
  bool set(uint i, real val);//set parameter i to val
  real get(uint i);
  void add(Weights *w);
  real sumAbs();
  real* copyOfAllWeights(real* cpy_to=NULL);
  void copyFrom(real* source);
  void copyFrom(WeightsList *pl);
  void displayNameList();
  void perturb(real fraction,bool extremum=false);
  void cleanup();
  void clearUpdates();
  bool isSame(WeightsList *pl);
  void randomInit(real max);
  void set(real r);
  ~WeightsList(){cleanup();};
};    
std::ostream& operator<<(std::ostream& out,WeightsList& pl);
std::istream& operator>>(std::istream& in,WeightsList& pl);
#endif
			     

      
  
  

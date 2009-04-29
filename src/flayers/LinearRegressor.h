#ifndef LINREG
#define LINREG
#include "fLayersGeneral.h"
/// Linear Regressor Class
class LinearRegressor
{
public:
  real X;
  ///sum of x
  real Ex;
  ///mean of x
  real Mx;
  real SSx;
  real Ey; 
  real My;
  real SPxy;
  uint n;
  LinearRegressor(){X=0;Ex=0;Mx=0;SSx=0;Ey=0;My=0;SPxy=0;};
  void init(uint n_history,real extrapolation_factor);
  void linearRegression(real **history,uint i,real* weights);
};
#endif

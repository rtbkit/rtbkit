#ifndef TWODSURFACEDATA
#define TWODSURFACEDATA

#include "PreDefArgvParser.h"
///  \class TwoDSurfaceData TwoDSurfaceData.h "Work/TwoDSurfaceData.h" 
///  \brief TwoDSurfaceData represent a 2 Dimentions DataSetArgvParser. 
///
/// generate 2D Surface DataSet of n X n
/// specifiy nb of examples -e or --n_examples
class TwoDSurfaceData:public DataSetArgvParser{
public:
  DataSetArgvParser* dsp;//use to do same normalisation of train_set
  uint n;//data nXn
  TwoDSurfaceData():DataSetArgvParser()
    {addOpt('D',"dim_x_y","6","dimention of surface NXN",INT);
    addOpt('x',"max_x","6","max value of x",REAL);
    addOpt('y',"max_y","6","max value of y",REAL);dsp=NULL;};
  void setDSP(DataSetArgvParser* dsp_){dsp=dsp_;};
  virtual void parse(int argc,char* argv[]);
  void createData();
  void normalizeDataInputs();
};
#endif

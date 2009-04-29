#ifndef FGENERAL_H
#define FGENERAL_H

#include <fstream>
#include <string>
#include <iostream>
#include <list>
#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include "fTimeMeasurer.h"
#include "limits.h"
#include "float.h"

//#ifdef WIN32
//#define FLT_MAX _I16_MAX
//#endif
//typedef float real;
#ifndef real
typedef float real;
#endif
//typedef double real;
//#define real float
#define uint unsigned int

void FERR(std::string msg);

//#define MISSING_VALUE -999
void printVecReal(real* r,uint size,bool return_line=true,int setw_size=10,std::ostream *out=NULL);
void readVecReal(real* r,uint size,std::istream *in);
std::string tostring(double x);
void set(real* r,uint size,real value);
std::string troncString(std::string name);
std::string argvToString(int argc, char* argv[]);
std::string argvToString(char* argv[],char* stop);
char** stringToArgv(std::string cmd,int &size);
void freeStringToArgv(char** tab,int size);
#endif

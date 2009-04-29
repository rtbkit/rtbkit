#include "fLayersGeneral.h"
#include <iostream>
#include <sstream>
#include <iomanip> 
#include <sstream>
#include <math.h>
#include "fTimeMeasurer.h"
#include "arch/exception.h"


using namespace std;


fTimeMeasurer time_measurer;
void FERR(string msg) { throw ML::Exception(msg); }

void printVecReal(real* r,uint size,bool return_line,int setw_size,ostream *out)
{
  if(setw_size==0){
    if(out)
      for(uint i=0;i<size;i++)
	(*out)<<*r++<<" ";
    else
      for(uint i=0;i<size;i++)
	cout<<*r++<<" ";
  }else{
    if(out)
      for(uint i=0;i<size;i++)
	(*out)<<setw(setw_size)<<*r++<<" ";
    else
      for(uint i=0;i<size;i++)
	cout<<setw(setw_size)<<*r++<<" ";
  }
  if (return_line) {
    if(out)
      (*out)<<endl;
    else
      cout<<endl;
  }
}
void readVecReal(real* r,uint size,istream *in)
{  
  for(uint i=0;i<size;i++)
    (*in)>>*r++;
}

real* realCalloc(uint nelems,uint size)
{
  return ((real*)calloc(nelems,size*sizeof(real)));
}
real* realMalloc(uint nelems)
{
  real *new_mem=((real*)malloc(nelems*sizeof(real)));
  memset(new_mem,0,(nelems*sizeof(real)));
  return new_mem;
}
uint* uintCalloc(uint nelems,uint size)
{
  return ((uint*)calloc(nelems,size*sizeof(uint)));
}
uint* uintMalloc(uint nelems)
{
  return ((uint*)malloc(nelems*sizeof(uint)));
}
string tostring(double x)
{
   ostringstream o;
   if (o << x)
     return o.str();
   // some sort of error handling goes here...
   return "conversion error";
} 
string troncString(string name)
{
  char *c=strrchr(name.c_str(),'/');
  if (c){
    return string(c+1);
  }else{
    return name;
  }
}
string argvToString(int argc, char* argv[])
{
  string s;
  for(int i=0;i<argc;i++){
    s+=argv[i];s+=" ";
  }
  return s;
}
string argvToString(char* argv[],char* stop)
{
  string s;
  int i=0;
  while(argv[i]!='\0'){
    if (strcmp(argv[i],stop)!=0){
      s+=argv[i];s+=" ";i++;
    }else return s;
  }
  return s;
}
char** stringToArgv(string cmd,int &size)
{
  istringstream in(cmd.c_str());
  string str;
  int count=0;
  while(in>>str)
    count++;
  char** argv=(char**)malloc(count*sizeof(char*));
  istringstream in2(cmd.c_str());
  count=0;
  while(in2>>str){
    argv[count]=(char*)malloc(strlen(str.c_str())+1);
    strcpy(argv[count],str.c_str());
    count++;
  }
  size=count;
  return argv;
}
void freeStringToArgv(char**tab,int size)
{
  for(int i=0;i<size;i++)
    free(tab[i]);
  free(tab);
}
void set(real* r,uint size,real value)
{
  real *pr=r;
  for(uint i=0;i<size;i++)
	*pr++=value;
}

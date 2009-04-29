#include "fDataSet.h"

using namespace std;

int Random(real n){return (int)(n*(double(rand())/RAND_MAX));}

void RandomDataSet::initTable()
{
  // init attributs
  table_=(uint*)malloc(size_*sizeof(uint));
  //init table
  for (uint j=0;j<size_;j++)
    table_[j]=j;
  //shuffle table
  int r;//random number
  uint buffer;
  for (uint i=0;i<(size_-1);i++){
    r=Random(size_-i-1)+i;
    buffer=table_[i];
    table_[i]=table_[r];
    table_[r]=buffer;
  }
  if (size_<=20)printTable();
  //init DataSet attributs
}

void RandomDataSet::printTable()
{
  cout<<"RandomData table, size="<<size_<<endl;
  for (uint i=0;i<size_;i++)
    cout<<i<<" "<<table_[i]<<endl;
}

DataSet* SeqDataSet::getSubSet(uint start,uint size,uint iter_size)
{
  if ( (start>size_) || ((start+size)>size_) ){
    FERR("Error in getSetSet(...)");
    return NULL;
  }
  return new SeqDataSet(data_+(start*input_size_),targs_+start,size,iter_size,input_size_,max_input_size_,true);
}

DataSet* RandomDataSet::getSubSet(uint start,uint size,uint iter_size)
{
  if ( (start>size_) || ((start+size)>size_) ){
    FERR("Error in getSetSet(...)");
    return NULL;
  }
  RandomDataSet * rds=new RandomDataSet(data_,targs_,size,iter_size,input_size_,max_input_size_,true);
  rds->table_=table_+start;
  return rds;
}

ostream& operator<<(ostream& out,DataSet& ds)
{
	cout<<"DataSet :"<<endl;
	for(uint z=0;z<ds.size_;z++){
		cout<<ds.input_<<" ";
		printVecReal(ds.input_,ds.input_size_);
		ds.next();
	}
	return out;
}

#include "ArgvParserContainer.h"


using namespace std;


extern int optind;
extern bool STDOUT;
void ArgvParserContainer::usage()
{
  typedef list<ArgvParser*>::const_iterator LI;
  if(STDOUT)cout<<"--------------------------------------------------------------------------------------------------"<<endl;
  if(STDOUT)cout<<"usage of "<<progname_<<endl;
  int k=0;
  for(LI i=parsers_.begin();i!=parsers_.end();++i){
    ArgvParser* p=*i;
    if(STDOUT)cout<<"------------- Argv Block "<<k<<" -------------------------------------------"<<endl;
    p->usage(false);
    k++;
  }
  if(STDOUT)cout<<"--------------------------------------------------------------------------------------------------"<<endl;
}
void ArgvParserContainer::parse(int &argc,char *argv[])
{
  optind=1;
  if(argc&&(argc>1)){
    if(strcmp(argv[1],"--help")==0 || strcmp(argv[1],"-h")==0){
    	usage();
    	exit(0);
    }
  }
  typedef list<ArgvParser*>::const_iterator LI;
  for(LI i=parsers_.begin();i!=parsers_.end();++i){
    ArgvParser* p=*i;
    p->parse(argc,argv);
  }
}
ArgvParser* ArgvParserContainer::getArgvParserOrNull(string name)
{
  ARGV_LI IT=parsers_.begin();
  ArgvParser* ap=NULL;
  while(IT !=parsers_.end()){
    ap=*IT;
    if(ap->name_==name)
      return ap;
    IT++;
  }
  return NULL;
}
void ArgvParserContainer::updateCmdLine()
{
  ARGV_LI IT=parsers_.begin();
  ArgvParser* ap=NULL;
  while(IT !=parsers_.end()){
    ap=*IT;
    ap->updateCmdLine();
    IT++;
  }
}
string ArgvParserContainer::getCmdLine(bool short_rep)
{
  string s="";
  ARGV_LI IT=parsers_.begin();
  ArgvParser* ap=NULL;
  while(IT !=parsers_.end()){
    ap=*IT;
    s+=ap->getCmdLine(short_rep)+" / ";
    IT++;
  }
  return s;
}
string ArgvParserContainer::getStringDescriptor(bool short_descr,ofstream *output)
{
  string s(progname_);
  for(ARGV_LI i=parsers_.begin();i!=parsers_.end();++i){
    ArgvParser* p=*i;
    if (((p->getNbdefaultValModify()>0)&& short_descr)||(!short_descr)){
      s+='_';
      s+=p->getStringDescriptor(short_descr,output);
    }
  }
  return s;
}

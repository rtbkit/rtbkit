#ifndef ARGVPARSERCONTAINER
#define ARGVPARSERCONTAINER

#include "ArgvParser.h"
typedef std::list<ArgvParser*>::const_iterator ARGV_LI;
///
///  \class ArgvParserContainer ArgvParserContainer.h "Utils/ArgvParserContainer.h"
///  \brief ArgvParserContainer is a container of ArgvParser.
///
class ArgvParserContainer
{
 protected:
  std::list<ArgvParser*> parsers_;
  std::string progname_;
 public:
  ArgvParserContainer(std::string progname="Program"){progname_=troncString(progname);};
  void add(ArgvParser* parser){parsers_.push_back(parser);};
  void usage();
  void parse(int &argc,char *argv[]);
  std::string getStringDescriptor(bool short_descr=true,std::ofstream *output=NULL);
  void updateCmdLine();
  std::string getCmdLine(bool short_rep=true);
  ArgvParser* getArgvParser(std::string name){ArgvParser* ap=getArgvParserOrNull(name);if(ap==NULL){FERR(name+" is not include in  ArgvParserContainer");return NULL;}else return ap;};
  ///WARNING if not found, return NULL
  ArgvParser* getArgvParserOrNull(std::string name);
};
#endif

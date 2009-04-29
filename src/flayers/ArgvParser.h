#ifndef ARGVPARSER
#define ARGVPARSER
#include "fLayersGeneral.h"
///
/// ArgvParser/PreDefArgvParser/ArgvParserContainer 
/// -why ? : main idea :    
///          Need simple and easy to use command line arguments parsing.
///          (Need something better then getopt.h)
/// specifics needs:
///          Need reusable parsing components.
///          Need automatic experiment name generator related to parameters.
///          Need automatic usage generator.
///          Need easy default values setting.
///          Need easy interface to get back arguments values.(getBool('c'),getReal("n_epochs") etc.)
///          Need simple automatic coherence parameters checker. 
///          Need simple arguments type setting (int,char,real...).
///  
/// Author : Francis Pieraut   
/// examples: /u/pierautf/fLayers/fLayersLib/Examples/SimpleArgvParser.cpp
///           /u/pierautf/fLayers/fLayersLib/Examples/SimpleArgvParser2.cpp
///           /u/pierautf/fLayers/fLayersLib/Examples/SimpleArgvParserContainer.cpp
///          /u/pierautf/fLayers/fLayersLib/Examples/fexp.cpp
///
enum argv_type{INT,REAL,STRING,CHAR,BOOL,CASE,NOTDEF};

///
///  \class  opt ArgvParser.h "Utils/ArgvParser.h" 
///  \brief define argument option.
///
/// opts is the most important structure you need to define to use ArgvParser
/// IMPORTANT : if you use opt[], last position = opt('>',"","",""), so call no argv constructor -> opt()
/// WARNING don't use empty string ""->strange things with STL !!
///
class opt
{
  public:
  opt(){c='>';name="";value="";description="";type=NOTDEF;pointer=NULL;};
  opt(char c, std::string name,std::string value,std::string description,argv_type type,void *pointer=NULL)
    {this->c=c;this->name=name;this->value=value;this->description=description;this->type=type;this->pointer=pointer;};
  // short option reference character (ex: -c 1)
  char c;           
   // option name
  std::string name;      
  // option default value
  std::string value;      
  // option description
  std::string description;
   // argument type
  argv_type type;   
   // pointer to the associate parameter
  void* pointer;        
  void setPointer(void *p){pointer=p;};//{cout<<"option->pointer "<<name<<" is set "<<endl;pointer=p;}; 
};
///
///  \class noargopt ArgvParser.h "Utils/ArgvParser.h" 
///  \brief define no argument option.
///
/// this structure is optional and define options with no arguments.
/// IMPORTANT : no argument option need to be define first in "opts_" list
/// IMPORTANT : last position = opt(">","","") so call no argv constructor -> noargopt()
///
class noargopt
{
  public:
  noargopt(){name=">";opt_name="";value="";};
  noargopt(std::string name,std::string opt_name,std::string value)
  {this->name=name;this->opt_name=opt_name;this->value=value;};
  std::string name;     /// option name, long or short option can be used
  std::string opt_name; /// this name should be one of the name in shortoptions.name or shortoptions.c
  std::string value;    /// value to assign if this option is use
};
typedef std::list<opt>::iterator OPT_LI;
typedef std::list<noargopt>::iterator NOARGOPT_LI;
class ArgvParser
{
 protected:
  ///list of short options, generate automaticaly in init(..)
  char* shortopts_;    
  ///descrition of ArgvParser 
  std::string description_; 
  std::string examples_;
  ///nb of default values modify
  int n_modif;         
  /// options list
  std::list<opt> opts_;     
  std::list<noargopt> noargopts_;
  void printOpts();
  void printNoArgOpts();
  void printExamples();
public:
  ///name of ArgvParser
  std::string name_;        
  ///all options in string representation
  std::string string_options_;
  ArgvParser(std::string name="",
	     opt *opts=NULL,
	     noargopt *noargopts=NULL,
	     std::string description="",
	     std::string examples="");  
  // Most important functions
  void usage(bool with_presentation=true); 
  virtual void parse(int argc,char* argv[]); 
  /// add options and no argument option
  void addOpt(opt *option);
  void addOpt(char c,std::string name,std::string value,std::string description,argv_type type,void *pointer=NULL){addOpt(new opt(c,name,value,description,type,pointer));};
  void addNoArgOpt(noargopt *option){noargopts_.push_front(*option);};
  void addNoArgOpt(std::string name,std::string opt_name,std::string value){addNoArgOpt(new noargopt(name,opt_name,value));};
  /// set options values
  void set(char key,std::string s);
  void set(char key,int value){set(key,tostring(value));};
  void set(std::string key,std::string s);
  void set(std::string key,int value){set(key,tostring(value));};
  void set(const char* key, const char* s){set(std::string(key),s);};
  /// get options values
  std::string get(std::string key,void* p=NULL); 
  char getChar(std::string key,char* p=NULL);
  real getReal(std::string key,real* p=NULL);
  int getInt(std::string key,int* p=NULL);
  bool getBool(std::string key,bool* p=NULL);
  std::string get(char key,void* p=NULL);
  char getChar(char key,char* p=NULL);
  real getReal(char key,real* p=NULL);
  int getInt(char key,int* p=NULL);
  bool getBool(char key,bool* p=NULL);
  /// find 
  OPT_LI findOpt(char c);
  OPT_LI findOpt(std::string s); 
  NOARGOPT_LI findNoArgOpt(std::string s);
  // others
  std::string toString(argv_type at);
  void updateCmdLine();
  std::string getCmdLine(bool short_rep=true);//short representation else long representation (ex: -a or --algo)
  void printDescription();
  virtual std::string getStringDescriptor(bool short_descr=true,std::ofstream *output=NULL);
  int getNbdefaultValModify(){return n_modif;};
  std::string getName(){return name_;};
  void setDescription(std::string description){description_=description;};
  std::string getDescription(){return description_;};
  ///this fct is call before and after parse()
  ///if you overwrite it call ArgvParser::CoherenceParamsValuesChecker()
  virtual void CoherenceParamsValuesChecker();
  virtual ~ArgvParser(){free(shortopts_);};
 private:
  bool SetNoArgOpt(std::string option);
  int getNextOption(int argc,char* argv[]);
  void init(opt* shortopts,noargopt* noargopts);
  void initShortOpts();
};    

#endif
  

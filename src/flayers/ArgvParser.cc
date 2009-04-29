#include <iomanip>
#include "ArgvParser.h"

using namespace std;

// global variable
int optind;//current position in argv[]
extern bool STDOUT;
ArgvParser::ArgvParser(string name,
		       opt *opts,
		       noargopt *noargopts,
		       string description,
		       string examples
		       )
{
  name_=name;
  description_=description;
  examples_=examples;
  if(STDOUT)cout<<name_<<" :"<<description_<<examples_<<endl;
  shortopts_=NULL;
  init(opts,noargopts);
}

void ArgvParser::printNoArgOpts()
{
  if (noargopts_.size() && STDOUT){
    cout<<"no arguments options : --long_option"<<endl;
    cout.flags(ios::left);
    typedef list<noargopt>::iterator IT;
    for(IT i=noargopts_.begin();i!=noargopts_.end();i++){
      {
	cout<<"\t"<<setw(15)<<i->name.c_str();
	string opt_name="["+i->opt_name+"]";
	cout<<setw(16)<<"  set"<<setw(25)<<opt_name.c_str()<<"  to   "<<i->value<<endl;
      }
    }
  }
}
void ArgvParser::printOpts()
{
  string type;
  cout<<"option : -short_option or --long_option"<<endl;
  cout<<"option list of "<<name_<<" parser "<<'['<<shortopts_<<']'<<endl;
  OPT_LI IT=opts_.begin();
  cout.flags(ios::left);
  while((IT!=opts_.end()) && STDOUT){
    cout.flags(ios::left);
    cout<<"\t"<<IT->c;
    cout<<" || "<<setiosflags(ios::left) << setw(25)<<IT->name.c_str();
    type=" <"+toString(IT->type)+">";
    cout<<setw(9)<<type.c_str();
    string default_val=" ["+IT->value+"]";
    cout<<setw(10)<<default_val.c_str();
    //cout<<setw(10)<<" ["/*<<setw(2)<<setiosflags(ios::right)*/<<so->value.c_str()<<"]";
    cout<<"  -> "<<IT->description<<endl;
    IT++;
  }
}
void ArgvParser::printDescription()
{
  if((description_!="") && STDOUT){
    cout<<"description: "<<description_<<endl;
  }
}
void ArgvParser::printExamples()
{
  if((examples_!="") && STDOUT){
    cout<<"examples : "<<name_<<" parser "<<" "<<examples_<<endl;
  }
}
void ArgvParser::init(opt *opts,noargopt *noargopts)
{
  int i=0;
  n_modif=0;
  optind=1; //start from first option in argv
  int n_opts=0;
  //count nb of option in shortopts_
  if(opts){
    while(opts[i++].c!='>'){
      n_opts++;
    }
  }
  //add option into option list
  for(i=0;i<n_opts;i++){
    opts_.push_front(opts[i]);
  }
  //add no args options into the list
  i=-1;
  if (noargopts){
    while(noargopts[++i].name!=">"){
      noargopts_.push_front(noargopts[i]);
    }
  }
}
void ArgvParser::initShortOpts()
{
  shortopts_=(char*)malloc(opts_.size()*2+1);
  //create string representing all short option used by getopt
  int i=0;
  for(OPT_LI li=opts_.begin();li!=opts_.end();li++,i++){
    shortopts_[i*2]=li->c;
    shortopts_[i*2+1]=':';
  }
  shortopts_[opts_.size()*2]='\0';
}
string ArgvParser::get(char key,void* p)
{
  OPT_LI option=findOpt(key);
  if(option!=opts_.end()){
    if (p)
      option->setPointer(p);
    return option->value;
  }else
    FERR( (string("Parser ")+name_+string(" unknown option :")+tostring(key)).c_str());
  return 0;
}
string ArgvParser::get(string key,void* p)
{
  OPT_LI option=findOpt(key);
  if(option!=opts_.end()){
    if (p)
      option->setPointer(p);
    return option->value;
  }else
    FERR( (string("Parser ")+name_+string(" unknown option :")+key).c_str());
  return 0;
}
real ArgvParser::getReal(string key,real* p)
{
  string s=get(key,(void*)p);
  return atof(s.c_str());
}
int ArgvParser::getInt(string key,int* p)
{
  string s=get(key,(void*)p);
  return atoi(s.c_str());
}
char ArgvParser::getChar(string key,char* p)
{
  return get(key,(void*)p)[0];
}
bool ArgvParser::getBool(string key,bool* p)
{
  string s=get(key,(void*)p);
  int val=atoi(s.c_str());
  if (val==1)
	  return true;
  else if(val==0)
	return false;
  else
	FERR("ArgvParser::getBool(string key)<"+key+">key is'nt a bool");
  return 0;
}
real ArgvParser::getReal(char key,real* p)
{
  string s=get(key,(void*)p);
  return atof(s.c_str());
}
int ArgvParser::getInt(char key,int *p)
{
  string s=get(key,(void*)p);
  return atoi(s.c_str());
}
char ArgvParser::getChar(char key,char* p)
{
  return get(key,(void*)p)[0];
}
bool ArgvParser::getBool(char key,bool* p)
{
  string s=get(key,(void*)p);
  int val=atoi(s.c_str());
  if (val==1)
	  return true;
  else if(val==0)
	return false;
  else
	FERR("ArgvParser::getBool(string key); associate key is'nt a bool");
   return 0;
}
void ArgvParser::set(string key,string s)
{
  OPT_LI option=findOpt(key);
  if(option!=opts_.end()){
    if (STDOUT)cout<<"set "<<option->c<<" or "<<option->name<<" -> "<<s<<endl;
    n_modif++;
    option->value=s;
  }
  else
    FERR( (string("Parser ")+name_+string(" unknown option :")+key).c_str());

}
void ArgvParser::set(char key,string s)
{
  OPT_LI option=findOpt(key);
  if(option!=opts_.end()){
	  if (STDOUT)cout<<"set "<<option->c<<" or "<<option->name<<" -> "<<s<<endl;
    n_modif++;
    option->value=s;
  }
  else
    FERR( (string("Parser ")+name_+string(" unknown option :")+key).c_str());
}
// shortdesc -> to get short/long string descriptor
string ArgvParser::getStringDescriptor(bool short_descr,ofstream *output)
{
  string s(name_);
  if (output && STDOUT){
    (*output)<<"Parser "<<name_<<"--------------------------------"<<endl;
  }
  OPT_LI IT=opts_.begin();
  while(IT!=opts_.end()){
    s+='_';
    if (short_descr)
      s+=IT->c;
    else
      s+=IT->name;
    s+=troncString(IT->value.c_str());
    if (output && STDOUT){
      (*output)<<IT->c<<" = "<<IT->value<<endl;
    }
    IT++;
  }
  return s;
}
void ArgvParser::updateCmdLine()
{
  OPT_LI IT=opts_.begin();
  while(IT!=opts_.end()){
    if(IT->pointer){
      cout<<"set "<<IT->name<<" from "<<IT->value<<" to ";
      switch ( IT->type ) {
      case BOOL:if (*((bool*)IT->pointer))IT->value="1";else IT->value="0";break;
      case REAL:IT->value=tostring(*((double*)IT->pointer));break;
      case INT:IT->value=tostring(*((int*)IT->pointer));break;
      case CHAR:IT->value=*((char*)IT->pointer);break;
      case CASE:case STRING:IT->value=*((string*)IT->pointer);break;
      default:FERR("undefine type conversion in ArgvParser::updateCmdLine()");
      }
      cout<<IT->value<<endl;
    }
    IT++;
  }
}
string ArgvParser::getCmdLine(bool short_rep)
{
  string s="";
  string c;
  OPT_LI IT=opts_.begin();
  while(IT!=opts_.end()){
    if(short_rep){
      c=IT->c;
      s+="-"+c;
    }else
      s+="--"+IT->name;
    s+=" "+IT->value+" ";
    IT++;
  }
  return s;
}
int ArgvParser::getNextOption(int argc,char* argv[])
{
  OPT_LI IT=opts_.end();
  if (optind<argc)
    {
      if (strcmp(argv[optind],"/")==0){
	optind++;
      }else{
	//save options
	string_options_+=argv[optind];
	string_options_+=" ";
	if (argv[optind][0]=='-'){
	  if(argv[optind][1]=='-'){//long options
	    //cout<<optind<<" :"<<argv[optind]<<"-> longoptions"<<endl;
	    int name_size=strlen(argv[optind]);
	    char* longopt=(char*)malloc(name_size);
	    strcpy(longopt,argv[optind]+2);
	    longopt[name_size-1]='\n';
	    if (!SetNoArgOpt(longopt)){
	      set(longopt,string(argv[++optind]));
	    }
	    free(longopt);
	    optind++;
	    return 0;
	  }
	  else{// short options
	    //cout<<optind<<" :"<<argv[optind]<<"-> shortoptions"<<endl;
	    string_options_+=argv[optind+1];
	    string_options_+=" ";
	    char c=argv[optind][1];
	    if (findOpt(c)!=IT){
	      set(c,string(argv[++optind]));
	      optind++;
	      return 0;
	    }
	  }
	}//unknown option
	string msg="option :"+string(argv[optind])+"-> use - for short option and -- for long option";
	FERR(msg.c_str());
      }
    }
  if (STDOUT)cout<<"--------------------------------ArgvParser "<<setw(15)<<name_.c_str()<<" is finish--------------------------------"<<endl;
  return EOF;
}
//-------------------------------------------------------------------------------------------
// usage(...)
//-------------------------------------------------------------------------------------------
void ArgvParser::usage(bool with_presentation)
{
  initShortOpts();
  if (with_presentation){
    cout<<"----------------------------------------------------------------------"<<endl;
    cout<<"usage of "<<setw(15)<<name_.c_str()<<" parser "<<endl<<endl;
  }
  printDescription();
  printOpts();
  printNoArgOpts();
  printExamples();
  if (with_presentation){
    cout<<"----------------------------------------------------------------------"<<endl;
  }
}
void ArgvParser::parse(int argc,char* argv[])
{
  if (argc==1)
    exit(0);
  char option;
  CoherenceParamsValuesChecker();
  while ( ( option=getNextOption( argc,argv) ) != EOF );
  CoherenceParamsValuesChecker();
}
bool ArgvParser::SetNoArgOpt(string option)
{
  list<noargopt>::iterator IT=noargopts_.begin();
  while(IT !=noargopts_.end()){
    {
      if(IT->name==option){
	set(IT->opt_name,IT->value);
	return true;
      }
      IT++;
    }
  }
  return false;
}
OPT_LI ArgvParser::findOpt(char c)
{
  OPT_LI IT=opts_.begin();
  while(IT !=opts_.end()){
    if(IT->c==c)
      return IT;
    IT++;
  }
  return opts_.end();
}
OPT_LI ArgvParser::findOpt(string s)
{
  OPT_LI IT=opts_.begin();
  while(IT !=opts_.end()){
    {
      if(IT->name==s)
	return IT;
      IT++;
    }
  }
  return opts_.end();
}
NOARGOPT_LI ArgvParser::findNoArgOpt(string s)
{
  NOARGOPT_LI IT=noargopts_.begin();
  while(IT !=noargopts_.end()){
    {
      if(IT->name==s)
	return IT;
      IT++;
    }
  }
  return noargopts_.end();
}
void ArgvParser::CoherenceParamsValuesChecker()
{
  typedef list<noargopt>::iterator IT;
  OPT_LI option;
  for(IT i=noargopts_.begin();i!=noargopts_.end();i++){
    // verification of i->name, short of long option name?
    if (strlen(i->name.c_str())==1)
      option=findOpt(i->name[0]);
    else
      option=findOpt(i->name);
    if (option!=opts_.end())
      FERR(("option "+string(i->name)+" is define 2 times").c_str());
    //verification of i->opt_name
    option=findOpt(i->opt_name);
    if (option==opts_.end())
      FERR((string(i->opt_name)+" is undefine in the option list").c_str());
  }
}
void ArgvParser::addOpt(opt *option){
  opts_.push_front(*option);
}
string  ArgvParser::toString(argv_type at)
{
  switch(at){
  case INT:return "int";break;
  case REAL:return "real";break;
  case STRING:return "string";break;
  case CHAR:return "char";break;
  case BOOL:return "bool";break;
  case CASE:return "case";break;
  case NOTDEF:return "not def";break;
  default:FERR("undefine argv_type \n");
  }
  return "";
}




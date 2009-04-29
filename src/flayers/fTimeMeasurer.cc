//#ifndef WIN32

#include "fTimeMeasurer.h"

using namespace std;

long fTimeMeasurer::getRuntime()
{
#ifndef WIN32
  struct tms buffer;
  times(&buffer);
  return((long)(buffer.tms_utime));
#endif
#ifdef WIN32
  time_t t;
  return time(&t);
#endif 
}
//-----------------------------------------------------------------------------------
// TimeMeasurer::startTimer()
// description: initialisation of t_before and cpu_t_before to current time.
//-----------------------------------------------------------------------------------
string fTimeMeasurer::startTimer(bool with_host)
{
  int i=0;
  started=true;
  time(&t_before);
  cpu_t_before =getRuntime();
  char *c_time=ctime(&t_before);
  string s=string("Start at  : ")+string(c_time);
  // host name
  if(with_host){
    string hostname = getenv("HOSTNAME");cout<<i++<<" "<<hostname<<endl;
    string::size_type end_index = hostname.find(".");
    hostname = hostname.substr(0, end_index);
    s+="Host        : "+hostname+"\n";
  }
  //cout<<"Start Timer!"<<endl;
  return s;
}
//-----------------------------------------------------------------------------------
// TimeMeasurer::stoptTimer()
// description: get nb of second from start.
// IMPORTANT set t_after and cpu_t_after 
//-----------------------------------------------------------------------------------
string fTimeMeasurer::stopTimer(bool cpu_time)
{
 if (started) {
   //get t_after and cpu_t_after
   time(&t_after);
   cpu_t_after =getRuntime();
   //return appropriate string 
   if (cpu_time)
     return string(ctime(&t_after));
   else
     return tostring(cpu_t_after);
 }
 FERR("TimeMeasurer::startTime has not been called !");
 return "";
}
//-----------------------------------------------------------------------------------
// TimeMeasurer::getRunningTime(bool only_time=false, bool cpu_time=true)
// only_time=true  -> total time since startTimer() 
// only_time=false -> total time since startTimer()+start time+stop time+host name
// cpu_time        -> if true, returns CPU time instead of aboslute time
//-----------------------------------------------------------------------------------
string fTimeMeasurer::getRunningTime(bool only_time, bool cpu_time)
{
  string output;
  char *c_time=NULL;
  if (!only_time){
    output+="----------------------------------------------\n";
  }
  if (cpu_time)
  {
    if (!only_time)
      output+="CPU Time : ";
    double total_time = (double)(cpu_t_after - cpu_t_before)/((double)CLK);
    output+=tostring(total_time);
  }
  else
  {
    double d_time=difftime(t_after,t_before);
    time_t t_diff=t_after-t_before; 
    struct tm* tm_diff=gmtime(&t_diff);
    // total time since startTimer()
    int n_days=(int)(d_time/(60*60*24));
    if (n_days>0)
      output+=tostring(n_days)+" days ";
    output+=tostring(tm_diff->tm_hour)+":";
    output+=tostring(tm_diff->tm_min)+":";
    output+=tostring(tm_diff->tm_sec)+"sec\n";
  }
  if (!only_time){
    c_time=ctime(&t_before);
    output+=string("Start at    : ")+string(c_time);
    c_time=ctime(&t_after);
    output+=string("Stop  at    : ")+string(c_time);
    // host name
    string hostname = getenv("HOSTNAME");
    string::size_type end_index = hostname.find(".");
    hostname = hostname.substr(0, end_index);
    output+="Host        : "+hostname+"\n";
  }
  return output;
}

string fTimeMeasurer::getStopRunningTime(bool only_time, bool cpu_time)
{
  stopTimer();
  return getRunningTime(only_time,cpu_time);
}
double fTimeMeasurer::getStopTime()
{
  stopTimer();
  return (double)(cpu_t_after - cpu_t_before)/((double)CLK);
}

//#endif

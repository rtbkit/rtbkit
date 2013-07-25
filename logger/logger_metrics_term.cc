#include "logger_metrics_term.h"
#include <sys/types.h>
#include <sys/unistd.h>

namespace Datacratic{

using namespace std;

LoggerMetricsTerm::LoggerMetricsTerm(Json::Value config,
    const string& coll, const string& appName) : ILoggerMetrics(coll)
{
    stringstream ss;
    ss << getpid();
    pid = ss.str();
    cout << "Logger Metrics terminal: app " << appName << " under pid " << pid << endl;
}

void LoggerMetricsTerm::logInCategory(const string& category,
    const Json::Value& json)
{
    cout << pid << "." << coll << "." << category 
         << ": " << json.toStyledString() << endl;
}

void LoggerMetricsTerm
::logInCategory(const std::string& category,
              const std::vector<std::string>& path,
              const NumOrStr& val)
{
    if(path.size() == 0){
        throw new ML::Exception(
            "You need to specify a path where to log the value");
    }
    stringstream ss;
    ss << val;
    stringstream newCat;
    newCat << pid << "." << coll << "." << category;
    for(string part: path){
        newCat << "." << part;
    }
    cout << newCat.str() << ": " << ss.str() << endl;
}

const std::string LoggerMetricsTerm::getProcessId() const{
    return pid;
}


}//namespace Datacratic

/* logger_metrics_term.cc
   François-Michel L'Heureux, 3 June 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
*/

#include <sys/types.h>
#include <sys/unistd.h>

#include "logger_metrics_term.h"

using namespace std;
using namespace Datacratic;


/****************************************************************************/
/* LOGGER METRICS TERM                                                      */
/****************************************************************************/

LoggerMetricsTerm::
LoggerMetricsTerm(Json::Value config,
                  const string & coll, const string & appName)
    : ILoggerMetrics(coll)
{
    stringstream ss;
    ss << getpid();
    pid = ss.str();
    cout << "Logger Metrics terminal: app " << appName << " under pid " << pid << endl;
}

void
LoggerMetricsTerm::
logInCategory(const string & category,
              const Json::Value & json)
{
    cout << pid << "." << coll << "." << category 
         << ": " << json.toStyledString() << endl;
}

void
LoggerMetricsTerm::
logInCategory(const std::string & category,
              const std::vector<std::string> & path,
              const NumOrStr & val)
{
    if (path.size() == 0) {
        throw ML::Exception("You need to specify a path where to log"
                            " the value");
    }
    stringstream ss;
    ss << val;
    stringstream newCat;
    newCat << pid << "." << coll << "." << category;
    for (const string & part: path) {
        newCat << "." << part;
    }
    cout << newCat.str() << ": " << ss.str() << endl;
}

std::string
LoggerMetricsTerm::
getProcessId()
    const
{
    return pid;
}

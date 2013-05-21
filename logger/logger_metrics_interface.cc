#include "soa/logger/logger_metrics_interface.h"
#include "soa/logger/logger_metrics_mongo.h"
#include "soa/jsoncpp/reader.h"

namespace Datacratic{
using namespace std;

std::mutex m;
string ILoggerMetrics::parentObjectId = "";
bool mustSetup = true;
shared_ptr<ILoggerMetrics> logger;

void ILoggerMetrics::setup(const string& configKey, const string& coll,
    const string& appName)
{
    std::lock_guard<std::mutex> lock(m);
    if(mustSetup){
        mustSetup = false;
        char* tmpMetricsParentId = getenv("METRICS_PARENT_ID");
        if(tmpMetricsParentId){
            parentObjectId = string(tmpMetricsParentId);
        }else{
            parentObjectId = "";
        }
        Json::Value config = Json::parseFromFile(getenv("CONFIG"));
        config = config[configKey];
        string loggerType = config["type"].asString();
        if(loggerType == "mongo"){
            logger = shared_ptr<ILoggerMetrics>(
                new LoggerMetricsMongo(config, coll, appName));
        }else{
            throw ML::Exception("Unknown logger type [%s]", loggerType.c_str());
        }
    }else{
        throw ML::Exception("Cannot setup more than once");
    }
}

shared_ptr<ILoggerMetrics> ILoggerMetrics
::getSingleton(){
    if(mustSetup){
        throw ML::Exception("Cannot get singleton within calling setup first");
    }
    return logger;
}
}

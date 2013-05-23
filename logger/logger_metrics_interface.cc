#include "soa/logger/logger_metrics_interface.h"
#include "soa/logger/logger_metrics_mongo.h"
#include "soa/logger/logger_metrics_void.h"
#include "soa/jsoncpp/reader.h"

namespace Datacratic{
using namespace std;

std::mutex m;
string ILoggerMetrics::parentObjectId = "";
bool mustSetup = true;
shared_ptr<ILoggerMetrics> logger;
const string ILoggerMetrics::METRICS = "metrics";
const string ILoggerMetrics::PROCESS = "process";
const string ILoggerMetrics::META = "meta";
bool ILoggerMetrics::failSafe;

shared_ptr<ILoggerMetrics> ILoggerMetrics
::setup(const string& configKey, const string& coll,
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
        failSafe = config["failSafe"].asBool();
        function<void()> fct = [&]{
            if(loggerType == "mongo"){
                logger = shared_ptr<ILoggerMetrics>(
                    new LoggerMetricsMongo(config, coll, appName));
            }else{
                throw ML::Exception("Unknown logger type [%s]", loggerType.c_str());
            }
        };
        if(failSafe){
            try{
                fct(); 
            }catch(const exception& exc){
                cerr << "Logger fail safe caught: " << exc.what() << endl;
                logger = shared_ptr<ILoggerMetrics>(
                    new LoggerMetricsVoid(config, coll, appName));
            }
        }else{
            fct();
        }
    }else{
        throw ML::Exception("Cannot setup more than once");
    }
    return logger;
}

shared_ptr<ILoggerMetrics> ILoggerMetrics
::getSingleton(){
    if(mustSetup){
        throw ML::Exception("Cannot get singleton within calling setup first");
    }
    return logger;
}

void ILoggerMetrics::logMetrics(Json::Value& json){
    vector<string> stack;
    function<void(const Json::Value&)> doit;
    doit = [&](const Json::Value& v){
        for(auto it = v.begin(); it != v.end(); ++it){
            if(v[it.memberName()].isObject()){
                stack.push_back(it.memberName());
                doit(v[it.memberName()]);
                stack.pop_back();
            }else{
                Json::Value current = v[it.memberName()];
                if(!(current.isInt() || current.isUInt() || current.isDouble()
                    || current.isNumeric()))
                {
                    stringstream key;
                    for(string s: stack){
                        key << s << ".";
                    }
                    key << it.memberName();
                    string value = current.toString();
                    cerr << value << endl;
                    value = value.substr(1, value.length() - 3);
                    throw new ML::Exception("logMetrics only accepts numerical "
                                            "values. Key [%s] has value [%s].",
                                            key.str().c_str(), value.c_str());
                }
            }
        }
    };
    function<void()> fct = [&]{
        doit(json);
        logInCategory(METRICS, json);
    };
    failSafeHelper(fct);
}

void ILoggerMetrics::failSafeHelper(std::function<void()> fct){
    if(failSafe){
        try{
            fct();
        }catch(const exception& exc){
            cerr << "Logger fail safe caught: " << exc.what() << endl;
        }
    }else{
        fct();
    }
}
}

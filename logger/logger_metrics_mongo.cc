#include "logger_metrics_mongo.h"
#include "soa/types/date.h"
#include "mongo/bson/bson.h"
#include "mongo/util/net/hostandport.h"

namespace Datacratic{

using namespace std;
using namespace mongo;

void LoggerMetricsMongo::doIt(function<void()>& fct){
    if(failSafe){
        try{
            fct();
        }catch(const exception& e){
            cerr << e.what() << endl;
        }
    }else{
        fct();
    } 
}

LoggerMetricsMongo::LoggerMetricsMongo(Json::Value config,
    const string& coll, const string& appName) : coll(coll)
{
    function<void()> init = [&] (){
        HostAndPort hostAndPort(config["hostAndPort"].asString());
        conn.connect(hostAndPort);
        string err;
        db = config["database"].asString();
        if(!conn.auth(db, config["user"].asString(),
                      config["pwd"].asString(), err))
        {
            throw ML::Exception(
                "MongoDB connection failed with msg [%s]", err.c_str());
        }
        string now = Date::now().printClassic();
        BSONObj obj = BSON(GENOID 
                           << "startTime" << now 
                           << "appName" << appName);
        conn.insert(db + "." + coll, obj);
        objectId = obj["_id"].OID();
    };
    doIt(init);
}

void LoggerMetricsMongo::logMetrics(Json::Value& json){
    logInCategory("metrics", json);
}

void LoggerMetricsMongo::logProcess(Json::Value& json){
    logInCategory("process", json);
}

void LoggerMetricsMongo::logMeta(Json::Value& json){
    logInCategory("meta", json);
}

mongo::BSONObj LoggerMetricsMongo::_fromJson(const Json::Value& json){
    string jsonStr = json.toString();
    if(*jsonStr.rbegin() == '\n'){
        jsonStr = jsonStr.substr(0, jsonStr.length() - 1);
    }
    return fromjson(jsonStr);
}

void LoggerMetricsMongo::logInCategory(const string& category,
    Json::Value& json)
{
    conn.update(db + "." + coll,
                BSON("_id" << objectId),
                BSON("$set" 
                    << BSON(category << _fromJson(json))));

}
}//namespace Datacratic

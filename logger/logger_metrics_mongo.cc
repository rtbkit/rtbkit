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
        setenv("METRICS_PARENT_ID", objectId.toString().c_str(), 1);
    };
    doIt(init);
}

void LoggerMetricsMongo::logInCategory(const string& category,
    Json::Value& json)
{
    BSONObjBuilder bson;
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
                stringstream key;
                key << category;
                for(string s: stack){
                    key << "." << s;
                }
                key << "." << it.memberName();
                string value = current.toString();
                if(current.isInt() || current.isUInt() || current.isDouble()
                    || current.isNumeric())
                {
                    value = value.substr(0, value.length() - 1); 
                }else{
                    value = value.substr(1, value.length() - 3); 
                }
                bson.append(key.str(), value);
            }
        }
    };
    doit(json);

    conn.update(db + "." + coll,
                BSON("_id" << objectId),
                BSON("$set" << bson.obj()),
                    true);

}

void LoggerMetricsMongo
::logInCategory(const std::string& category, const mongo::BSONObj& obj){
    conn.update(db + "." + coll,
                BSON("_id" << objectId),
                BSON("$set" 
                    << BSON(category << obj)),
                true);
}

void LoggerMetricsMongo
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
    newCat << category;
    for(string part: path){
        newCat << "." << part;
    }
    conn.update(db + "." + coll,
                BSON("_id" << objectId),
                BSON("$set" 
                    << BSON(newCat.str() << ss.str())),
                true);
}

}//namespace Datacratic

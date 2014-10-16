#include "logger_metrics_mongo.h"
#include "mongo/bson/bson.h"
#include "mongo/util/net/hostandport.h"

namespace Datacratic{

using namespace std;
using namespace mongo;

LoggerMetricsMongo::LoggerMetricsMongo(Json::Value config,
    const string& coll, const string& appName) : ILoggerMetrics(coll)
{
    for(string s: {"hostAndPort", "database", "user", "pwd"}){
        if(config[s].isNull()){
            throw ML::Exception("Missing LoggerMetricsMongo parameter [%s]",
                                s.c_str());
        }
    }
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
    BSONObj obj = BSON(GENOID);
    conn.insert(db + "." + coll, obj);
    objectId = obj["_id"].OID();
    logToTerm = config["logToTerm"].asBool();
}

void LoggerMetricsMongo::logInCategory(const string& category,
    const Json::Value& json)
{
    BSONObjBuilder bson;
    vector<string> stack;
    function<void(const Json::Value&)> doit;

    auto format = [](const Json::Value& v) -> string{
        string str = v.toString();
        if(v.isInt() || v.isUInt() || v.isDouble() || v.isNumeric()){
            return str.substr(0, str.length() - 1);
        }
        return str.substr(1, str.length() - 3);
    };

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
                if(current.isArray()){
                    BSONArrayBuilder arr;
                    for(const Json::Value el: current){
                        arr.append(format(el));
                    }
                    bson.append(key.str(), arr.arr());
                }else{
                    bson.append(key.str(), format(current));
                }
            }
        }
    };
    doit(json);

    if(logToTerm){
        cout << objectId << "." << coll << "." << category 
             << ": " << json.toStyledString() << endl;
    }

    conn.update(db + "." + coll,
                BSON("_id" << objectId),
                BSON("$set" << bson.obj()),
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
    string newCatStr = newCat.str();
    string str = ss.str();
    
    if(logToTerm){
        cout << newCatStr << ": " << str << endl;
    }
    conn.update(db + "." + coll,
                BSON("_id" << objectId),
                BSON("$set" 
                    << BSON(newCatStr << str)),
                true);
}

const std::string LoggerMetricsMongo::getProcessId() const{
    return objectId.toString(); 
}


}//namespace Datacratic

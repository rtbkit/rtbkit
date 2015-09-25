#include "logger_metrics_mongo.h"
#include "mongo/bson/bson.h"
#include "mongo/util/net/hostandport.h"
#include "jml/utils/string_functions.h"
#include "soa/utils/mongo_init.h"

namespace Datacratic{

using namespace std;
using namespace mongo;
using namespace Datacratic;


/****************************************************************************/
/* LOGGER METRICS MONGO                                                     */
/****************************************************************************/

LoggerMetricsMongo::
LoggerMetricsMongo(Json::Value config, const string & coll,
                   const string & appName)
    : ILoggerMetrics(coll)
{
    for(string s: {"hostAndPort", "database", "user", "pwd"}){
        if(config[s].isNull()){
            throw ML::Exception("Missing LoggerMetricsMongo parameter [%s]",
                                s.c_str());
        }
    }

    vector<string> hapStrs = ML::split(config["hostAndPort"].asString(), ',');
    if (hapStrs.size() > 1) {
        vector<HostAndPort> haps;
        for (const string & hapStr: hapStrs) {
               haps.emplace_back(hapStr);
        }
        conn.reset(new mongo::DBClientReplicaSet(hapStrs[0], haps, 100));
    }
    else {
        std::shared_ptr<DBClientConnection> tmpConn =
            make_shared<DBClientConnection>();
        tmpConn->connect(hapStrs[0]);
        conn = tmpConn;
    }
    db = config["database"].asString();

    auto impl = [&] (string mechanism) {
        BSONObj b = BSON("user" << config["user"].asString()
                  << "pwd" << config["pwd"].asString()
                  << "mechanism" << mechanism
                  << "db" << db);
        try {
            conn->auth(b);
        }
        catch (const UserException & _) {
            return false;
        }
        return true;
    };

    if (!impl("SCRAM-SHA-1")) {
        cerr << "Failed to authenticate with SCRAM-SHA-1, "
                "trying with MONGODB-CR" << endl;
        if (!impl("MONGODB-CR")) {
            cerr << "Failed with MONGODB-CR as well" << endl;
            throw ("Failed to auth");
        }
    }

    BSONObj obj = BSON(GENOID);
    conn->insert(db + "." + coll, obj);
    objectId = obj["_id"].OID();
    logToTerm = config["logToTerm"].asBool();
}

void LoggerMetricsMongo::logInCategory(const string& category,
    const Json::Value& json)
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
                if(current.isArray()){
                    BSONArrayBuilder arr;
                    for(const Json::Value el: current){
                        if (el.isInt()) {
                            arr.append(el.asInt());
                        }
                        else if (el.isUInt()) {
                            arr.append((uint32_t)el.asUInt());
                        }
                        else if (el.isDouble()) {
                            arr.append(el.asDouble());
                        }
                        else {
                            arr.append(el.asString());
                        }
                    }
                    bson.append(key.str(), arr.arr());
                }
                else {
                    if (current.isInt()) {
                        bson.append(key.str(), current.asInt());
                    }
                    else if (current.isUInt()) {
                        bson.append(key.str(), (uint32_t)current.asUInt());
                    }
                    else if (current.isDouble()) {
                        bson.append(key.str(), current.asDouble());
                    }
                    else {
                        bson.append(key.str(), current.asString());
                    }
                }
            }
        }
    };
    doit(json);

    if(logToTerm){
        cout << objectId << "." << coll << "." << category 
             << ": " << json.toStyledString() << endl;
    }

    conn->update(db + "." + coll,
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
    stringstream newCat;
    newCat << category;
    for(string part: path){
        newCat << "." << part;
    }
    string newCatStr = newCat.str();
    
    BSONObj bsonObj;
    //reference
    //typedef boost::variant<int, float, double, size_t, uint32_t, String> NumOrStr;
    int type = val.which();
    if (type == 0) {
        bsonObj = BSON(newCatStr << boost::get<int>(val));
    }
    else if (type == 1) {
        bsonObj = BSON(newCatStr << boost::get<float>(val));
    }
    else if (type == 2) {
        bsonObj = BSON(newCatStr << boost::get<double>(val));
    }
    else if (type == 3) {
        bsonObj = BSON(newCatStr << (int)boost::get<size_t>(val));
    }
    else if (type == 4) {
        bsonObj = BSON(newCatStr << boost::get<uint32_t>(val));
    }
    else {
        stringstream ss;
        ss << val;
        string str = ss.str();
        if (type != 5) {
            cerr << "Unknown type of NumOrStr for value: " << str << endl;
        }
        bsonObj = BSON(newCatStr << str);
    }
    if(logToTerm){
        cerr << bsonObj.toString() << endl;
    }
    conn->update(db + "." + coll,
                BSON("_id" << objectId),
                BSON("$set" << bsonObj),
                true);
}

std::string
LoggerMetricsMongo::
getProcessId()
    const
{
    return objectId.toString();
}


}//namespace Datacratic

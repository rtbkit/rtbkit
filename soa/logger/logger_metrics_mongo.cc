/* logger_metrics_interface.cc
   François-Michel L'Heureux, 21 May 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
*/

#include "mongo/bson/bson.h"
#include "mongo/util/net/hostandport.h"
#include "jml/utils/string_functions.h"
#include "logger_metrics_mongo.h"


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
    for (const string & s: {"hostAndPort", "database", "user", "pwd"}) {
        if (config[s].isNull()) {
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
        auto tmpConn = make_shared<DBClientConnection>();
        tmpConn->connect(hapStrs[0]);
        conn = tmpConn;
    }
    db = config["database"].asString();
    string err;
    if (!conn->auth(db, config["user"].asString(), config["pwd"].asString(),
                    err)) {
        throw ML::Exception("MongoDB connection failed with msg [%s]",
                            err.c_str());
    }
    BSONObj obj = BSON(GENOID);
    conn->insert(db + "." + coll, obj);
    objectId = obj["_id"].OID();
    logToTerm = config["logToTerm"].asBool();
}

void
LoggerMetricsMongo::
logInCategory(const string & category, const Json::Value & json)
{
    BSONObjBuilder bson;
    vector<string> stack;
    function<void(const Json::Value &)> doit;

    doit = [&] (const Json::Value & v) {
        for (auto it = v.begin(); it != v.end(); ++it) {
            string memberName = it.memberName();
            if (v[memberName].isObject()) {
                stack.push_back(memberName);
                doit(v[memberName]);
                stack.pop_back();
            }
            else {
                Json::Value current = v[memberName];
                stringstream key;
                key << category;
                for (const string & s: stack) {
                    key << "." << s;
                }
                key << "." << memberName;
                if (current.isArray()) {
                    BSONArrayBuilder arr;
                    for (const Json::Value el: current) {
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

    if (logToTerm) {
        cout << objectId << "." << coll << "." << category 
             << ": " << json.toStyledString() << endl;
    }

    conn->update(db + "." + coll, BSON("_id" << objectId),
                 BSON("$set" << bson.obj()), true);
}

void
LoggerMetricsMongo::
logInCategory(const std::string & category,
              const std::vector<std::string> & path,
              const NumOrStr & val)
{
    if (path.empty()) {
        throw ML::Exception("You need to specify a path where to log"
                            " the value");
    }
    stringstream newCat;
    newCat << category;
    for (const string & part: path) {
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
    if (logToTerm) {
        cerr << bsonObj.toString() << endl;
    }
    conn->update(db + "." + coll, BSON("_id" << objectId),
                 BSON("$set" << bsonObj), true);
}

std::string
LoggerMetricsMongo::
getProcessId()
    const
{
    return objectId.toString(); 
}

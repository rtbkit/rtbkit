#include "kvp_logger_mongodb.h"
#include "mongo/bson/bson.h"
#include "mongo/util/net/hostandport.h"

namespace Datacratic{

using namespace std;

std::function<void()> KvpLoggerMongoDb
::makeInitFct(const string& hostAndPort, const string& db,
              const string& user, const string& pwd)
{
    function<void()> init = [&] (){
        cerr << hostAndPort << endl;
        conn.connect(hostAndPort);
        string err;
        if(!conn.auth(db, user, pwd, err)){
            throw ML::Exception("MongoDB connection failed with msg [" 
                + err + "]");
        }
    };
    return init;
}

KvpLoggerMongoDb::KvpLoggerMongoDb(const KvpLoggerParams& params) :
    db(params.db), failSafe(params.failSafe)
{
    function<void()> init = makeInitFct(params.hostAndPort,
                                        params.db,
                                        params.user,
                                        params.pwd);
    doIt(init);
}

KvpLoggerMongoDb::KvpLoggerMongoDb(const boost::property_tree::ptree& pt) :
    db(pt.get<string>("db")), failSafe(pt.get<bool>("failSafe"))
{
    function<void()> init = makeInitFct(pt.get<string>("hostAndPort"),
                                        db,
                                        pt.get<string>("user"),
                                        pt.get<string>("pwd"));
    doIt(init);
}

void KvpLoggerMongoDb
::log(const map<string, string>& data, const string& coll){
    function<void()>  _log = [&](){
        map<string, string>::const_iterator it;
        mongo::BSONObjBuilder b;
        b.genOID();
        for(it = data.begin(); it != data.end(); it ++){
            b.append((*it).first, (*it).second);
        }
        mongo::BSONObj p = b.obj();
        conn.insert(db + "." + coll, p);
        cerr << "ID: " << p["_id"].toString() << endl;
    };
    doIt(_log);
}

void KvpLoggerMongoDb
::log(Json::Value& json, const string& coll){
    function<void()>  _log = [&](){
        string jsonStr = json.toString();
        if(*jsonStr.rbegin() == '\n'){
            jsonStr = jsonStr.substr(0, jsonStr.length() - 1);
        }
        mongo::BSONObj o = mongo::fromjson(jsonStr);
        conn.insert(db + "." + coll, o);
        //cerr << "ID: " << o["_id"].toString() << endl;
    };
    doIt(_log);
}

void KvpLoggerMongoDb::doIt(function<void()>& fct){
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

}//namespace Datacratic

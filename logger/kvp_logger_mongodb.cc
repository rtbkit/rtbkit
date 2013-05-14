#include "kvp_logger_mongodb.h"
#include "mongo/bson/bson.h"
#include "mongo/util/net/hostandport.h"

namespace Datacratic{

using namespace std;

KvpLoggerMongoDb::KvpLoggerMongoDb(const KvpLoggerParams& params) :
    params(params)
{
    function<void()> init = [&] (){
        mongo::HostAndPort hostAndPort(params.hostAndPort);
        conn.connect(hostAndPort);
        string err;
        if(!conn.auth(params.db, params.user, params.pwd, err)){
            throw ML::Exception("MongoDB connection failed with msg [" 
                + err + "]");
        }
    };
    doIt(init);
}

void KvpLoggerMongoDb
::log(const map<string, string>& data, const string& coll){
    function<void()>  _log = [&](){
        map<string, string>::const_iterator it;
        mongo::BSONObjBuilder b;
        for(it = data.begin(); it != data.end(); it ++){
            b.append((*it).first, (*it).second);
        }
        mongo::BSONObj p = b.obj();
        conn.insert(params.db + "." + coll, p);
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
        conn.insert(params.db + "." + coll, o);
    };
    doIt(_log);
}

void KvpLoggerMongoDb::doIt(function<void()>& fct){
    if(params.failSafe){
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

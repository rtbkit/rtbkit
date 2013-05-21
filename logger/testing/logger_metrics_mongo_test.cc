
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "soa/logger/kvp_logger_interface.h"
#include "soa/types/date.h"
#include "mongo/client/dbclient.h"
#include <unistd.h>
#include <time.h>
#include "soa/logger/logger_metrics_interface.h"
#include "jml/utils/filter_streams.h"

using namespace std;
using namespace Datacratic;
using namespace mongo;
using namespace bson;


BOOST_AUTO_TEST_CASE( logger_metrics_mongo )
{
    string filename = "logger_metrics_mongo_config.json";
    string configKey = "mongoConfig";
    Json::Value json;
    json["user"]        = "datacratic_test_user";
    json["pwd"]         = "datacratic_test_pwd";
    json["hostAndPort"] = "ds047437.mongolab.com:47437";
    json["database"]    = "datacratic_test";
    json["type"]        = "mongo";
    Json::Value json2;
    json2[configKey] = json;
    ML::filter_ostream out(filename);
    out << json2.toString();
    out.flush();
    out.close();
    setenv("CONFIG", filename.c_str(), 1);

    string coll = "metrics_test";
    ILoggerMetrics::setup(configKey, coll, "metrics_test_app");
    std::shared_ptr<ILoggerMetrics> logger = ILoggerMetrics::getSingleton();
    Json::Value v;
    v["coco"] = "123";
    logger->logMetrics(v);
    v.clear();
    v["octo"] = "sanchez";
    logger->logMeta(v);
    v.clear();
    v["expos"] = "baseball";
    logger->logProcess(v);

    
    sleep(2);
    mongo::HostAndPort hostAndPort(json["hostAndPort"].asString());
    mongo::DBClientConnection conn;
    conn.connect(hostAndPort);
    string err;
    if(!conn.auth(json["database"].asString(), json["user"].asString(),
                  json["pwd"].asString(), err))
    {
        throw ML::Exception(
            "MongoDB connection failed with msg [%s]", err.c_str());
    }
    cerr << "FMLHHHHH" << endl;
    mongo::BSONObj where = BSON("metrics" << BSON("coco" << "123"));
    auto_ptr<DBClientCursor> cursor =
        conn.query(json["database"].asString() + "." + coll, where);
    if(cursor->more()){
        mongo::BSONObj p = cursor->next();
        cerr << p.toString() << endl;
        conn.remove(json["database"].asString() + "." + coll, p, 1);
        BOOST_CHECK_EQUAL(p["appName"].toString(),
            "appName: \"metrics_test_app\"");
        BOOST_CHECK_EQUAL(p["metrics"]["coco"].toString(), "coco: \"123\"");
        BOOST_CHECK_EQUAL(p["meta"]["octo"].toString(), "octo: \"sanchez\"");
        BOOST_CHECK_EQUAL(p["process"]["expos"].toString(), "expos: \"baseball\"");
    }else{
        BOOST_CHECK(false);
    }
}

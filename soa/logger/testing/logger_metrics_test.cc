/**
 * logger_metrics_test.cc
 * Mich, 2014-11-17
 * Copyright (c) 2014 Datacratic Inc. All rights reserved.
 *
 * Manual test for the logger metrics. Provide the proper json config and
 * run.
 **/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/regex.hpp>

#include "mongo/bson/bson.h"
#include "mongo/bson/bsonobj.h"
#include "mongo/client/dbclient.h"
#include "jml/utils/filter_streams.h"
#include "soa/service/testing/mongo_temporary_server.h"
#include "soa/logger/logger_metrics_interface.h"

using namespace ML;
using namespace Datacratic;
using namespace mongo;

BOOST_AUTO_TEST_CASE( test_logger_metrics )
{
    Mongo::MongoTemporaryServer mongo;
    setenv("CONFIG", "logger/testing/logger_metrics_config.json", 1);
    shared_ptr<ILoggerMetrics> logger =
        ILoggerMetrics::setup("metricsLogger", "lalmetrics", "test");

    logger->logMeta({"a", "b"}, "taratapom");

    Json::Value config;
    filter_istream cfgStream("logger/testing/logger_metrics_config.json");
    cfgStream >> config;

    Json::Value metricsLogger = config["metricsLogger"];
    auto conn = std::make_shared<mongo::DBClientConnection>();
    conn->connect(metricsLogger["hostAndPort"].asString());
    string err;
    if (!conn->auth(metricsLogger["database"].asString(),
                    metricsLogger["user"].asString(),
                    metricsLogger["pwd"].asString(), err)) {
        throw ML::Exception("Failed to log to mongo tmp server: %s",
                            err.c_str());
    }

    BOOST_CHECK_EQUAL(conn->count("test.lalmetrics"), 1);
    auto cursor = conn->query("test.lalmetrics", mongo::BSONObj());
    BOOST_CHECK(cursor->more());
    {
        mongo::BSONObj p = cursor->next();
        BOOST_CHECK_EQUAL(p.getFieldDotted("meta.a.b").String(), "taratapom");
    }

    logger->logMetrics({"fooValue"}, 123);
    ML::sleep(1); // Leave time for async write
    cursor = conn->query("test.lalmetrics", mongo::BSONObj());
    BOOST_CHECK(cursor->more());
    {
        mongo::BSONObj p = cursor->next();
        BOOST_CHECK_EQUAL(p["metrics"]["fooValue"].Number(), 123);
    }

    Json::Value block;
    block["alpha"] = 1;
    block["beta"] = 2;
    block["coco"] = Json::objectValue;
    block["coco"]["sanchez"] = 3;
    logger->logMetrics(block);
    ML::sleep(1); // Leave time for async write
    cursor = conn->query("test.lalmetrics", mongo::BSONObj());
    BOOST_CHECK(cursor->more());
    {
        mongo::BSONObj p = cursor->next();
        BOOST_CHECK_EQUAL(p["metrics"]["coco"]["sanchez"].Number(), 3);
    }

    Json::Value v;
    v["coco"] = 123;
    logger->logMetrics(v);
    v.clear();
    v["expos"]["city"] = "baseball";
    v["expos"]["sport"] = "montreal";
    v["expos"]["players"][0] = "pedro";
    v["expos"]["players"][1] = "mario";
    v["expos"]["players"][2] = "octo";
    logger->logProcess(v);

    logger->logMeta({"octo"}, "sanchez");
    logger->close();
    
    ML::sleep(2.0);

    string objectIdStr = getenv("METRICS_PARENT_ID");
    mongo::OID objectId(objectIdStr);
    mongo::BSONObj where = BSON("_id" << objectId);
    cursor = conn->query(database + ".metrics_test", where);

    BOOST_CHECK(cursor->more());
    {
        mongo::BSONObj p = cursor->next();
        cerr << p.toString() << endl;
        // conn->remove(database + ".metrics_test", p, 1);
        BOOST_CHECK_EQUAL(p["process"]["appName"].String(), "test_app");
        BOOST_CHECK_EQUAL(p["metrics"]["coco"].Long(), 123);
        BOOST_CHECK_EQUAL(p["meta"]["octo"].String(), "sanchez");
        BOOST_CHECK_EQUAL(p["process"]["expos"]["city"].String(), "baseball");
        BOOST_CHECK_EQUAL(p["process"]["expos"]["sport"].String(), "montreal");
        //BSONObj bsonPlayers = BSON("process.expos.players" << players);
        auto players = p.getFieldDotted("process.expos.players").Array();
        BOOST_CHECK_EQUAL(players.size(), 3);
        BOOST_CHECK_EQUAL(players[0].String(), "pedro");
        BOOST_CHECK_EQUAL(players[1].String(), "mario");
        BOOST_CHECK_EQUAL(players[2].String(), "octo");
        BOOST_CHECK(p["process"]["endDate"].toString() != "EOO");
        BOOST_CHECK(p["process"]["duration"].toString() != "EOO");
    }

    FILE * pipe = popen("echo -n $METRICS_PARENT_ID", "r");
    ExcAssert(pipe != nullptr);
    char buffer[128];
    std::string result = "";
    while (!feof(pipe)) {
        if (fgets(buffer, 128, pipe) != NULL)
                result += buffer;
    }
    pclose(pipe);

    BOOST_CHECK_EQUAL(objectIdStr, result);
}

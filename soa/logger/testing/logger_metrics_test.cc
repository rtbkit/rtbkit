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
using namespace std;

BOOST_AUTO_TEST_CASE( test_logger_metrics ) {
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
}

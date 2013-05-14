/* kvp_logger_test.cc
   Copyright (c) 2013 Datacratic.  All rights reserved.

*/
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/shared_ptr.hpp>
#include "soa/logger/kvp_logger_interface.h"
#include "soa/types/date.h"
#include "mongo/client/dbclient.h"
#include <unistd.h>
#include <time.h>

using namespace std;
using namespace Datacratic;
using namespace mongo;
using namespace bson;

BOOST_AUTO_TEST_CASE( kvp_logger_mongodb )
{
    IKvpLogger::KvpLoggerParams params;
    params.hostAndPort = "ds047437.mongolab.com:47437";
    params.db = "datacratic_test";
    params.user = "datacratic_test_user";
    params.pwd = "datacratic_test_pwd";
    params.failSafe = false;
    string coll = "test_coll";

    DBClientConnection conn;
    HostAndPort hostAndPort(params.hostAndPort);
    conn.connect(hostAndPort);
    string err;
    BOOST_CHECK(conn.auth(params.db, params.user, params.pwd, err));

    srand (time(NULL));
    string randVal = to_string(rand());
    cout << "rand val:" << randVal << endl;

    std::shared_ptr<IKvpLogger> logger =
        IKvpLogger::getKvpLogger("mongodb", params);
    Datacratic::Date d = Datacratic::Date::now();
    string now = d.printClassic();
    cout << now << endl;
    logger->log({{"test", now}, {"rand", randVal}}, coll);

    sleep(2);//mongodb is assync, wait a bit

    BSONObj obj = BSON("test" << now << "rand" << randVal);
    auto_ptr<DBClientCursor> cursor = conn.query(params.db + "." + coll, obj);
    bool result = false;
    while(cursor->more()){
        result = true;
        BSONObj p = cursor->next();
        cout << p.getStringField("test") << " - "
            << p.getStringField("rand") << endl;
        BOOST_CHECK_EQUAL(p.getStringField("test"), now);
        BOOST_CHECK_EQUAL(p.getStringField("rand"), randVal);
    }
    conn.remove(params.db + "." + coll, obj, 1);
    BOOST_CHECK(result);
    
    struct Coco{
        string randVal;
        Coco(string randVal) : randVal(randVal){}

        Json::Value toJson() const{
            Json::Value v;
            v["test"] = "wololo";
            v["rand"] = randVal;
            return v;
        };
    };
    Coco c(randVal);
    logger->log(c, coll);

    obj = BSON("test" << "wololo" << "rand" << randVal);
    auto_ptr<DBClientCursor> cursor2 = conn.query(params.db + "." + coll, obj);
    result = false;
    while(cursor2->more()){
        result = true;
        BSONObj p = cursor2->next();
        cout << p.getStringField("test") << " - "
            << p.getStringField("rand") << endl;
        BOOST_CHECK_EQUAL(p.getStringField("test"), "wololo");
        BOOST_CHECK_EQUAL(p.getStringField("rand"), randVal);
    }
    conn.remove(params.db + "." + coll, obj, 1);
    BOOST_CHECK(result);
}


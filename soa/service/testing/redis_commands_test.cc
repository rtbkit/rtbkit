/* redis_async_test.cc
   Wolfgang Sourdeau, 11 December 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Test commands supported by our Redis class
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <unordered_map>

#include <boost/test/unit_test.hpp>

#include "soa/service/redis.h"
#include "soa/service/testing/redis_temporary_server.h"

using namespace std;
using namespace Redis;


BOOST_AUTO_TEST_CASE( test_redis_commands )
{
    RedisTemporaryServer redis;
    AsyncConnection connection(redis);

    /* MSET is supported */
    Command mset(MSET);
    mset.addArg("testkey1"); mset.addArg(1234);
    mset.addArg("testkey2"); mset.addArg(9876);
    mset.addArg("testkey3"); mset.addArg("textvalue");

    Result result = connection.exec(mset);
    BOOST_CHECK_EQUAL(result.ok(), true);

    /* MGET is supported and that all slots requested must be returned in the
       form of an array, including empty ones, without errors */
    Command mget(MGET);
    mget.addArg("testkey1");
    mget.addArg("testkey3");
    mget.addArg("poil");

    result = connection.exec(mget);
    BOOST_CHECK_EQUAL(result.ok(), true);
    Reply reply = result.reply();
    BOOST_CHECK_EQUAL(reply.type(), ARRAY);

    Reply subreply = reply[0]; /* testkey1 */
    BOOST_CHECK_EQUAL(subreply.type(), STRING);
    BOOST_CHECK_EQUAL(subreply.asString(), "1234");
    /* FIXME: expected behaviour:
    BOOST_CHECK_EQUAL(subreply.type(), INTEGER);
    BOOST_CHECK_EQUAL(subreply.asInt(), 1234);
    */

    subreply = reply[1]; /* testkey3 */
    BOOST_CHECK_EQUAL(subreply.type(), STRING);
    BOOST_CHECK_EQUAL(string(subreply), "textvalue");

    subreply = reply[2]; /* poil */
    BOOST_CHECK_EQUAL(subreply.type(), NIL);
    

    /* SADD is supported and each slot is used once per key, even when
       specified multiple times */
    Command sadd(SADD);
    sadd.addArg("my-new-set");
    sadd.addArg("testkey1");
    sadd.addArg("testkey2");
    sadd.addArg("testkey2");

    result = connection.exec(sadd);
    BOOST_CHECK_EQUAL(result.ok(), true);

    /* SMEMBERS is supported and does return the values registered above */
    Command smembers(SMEMBERS);
    smembers.addArg("my-new-set");

    result = connection.exec(smembers);
    BOOST_CHECK_EQUAL(result.ok(), true);
    reply = result.reply();
    BOOST_CHECK_EQUAL(reply.type(), ARRAY);
    BOOST_CHECK_EQUAL(reply.length(), 2); /* testkey1, testkey2 */

    /* members are not specifically returned in order */
    unordered_map<string, bool> keyMap;
    subreply = reply[0];
    BOOST_CHECK_EQUAL(subreply.type(), STRING);
    keyMap[subreply.asString()] = true;
    subreply = reply[1];
    BOOST_CHECK_EQUAL(subreply.type(), STRING);
    keyMap[subreply.asString()] = true;

    BOOST_CHECK_EQUAL(keyMap["testkey1"], true);
    BOOST_CHECK_EQUAL(keyMap["testkey2"], true);

    /* SMOVE is supported, it moves a key from one set to another*/
    Command smove(SMOVE);
    smove.addArg("my-new-set");
    smove.addArg("my-destination-set");
    smove.addArg("testkey2");

    result = connection.exec(smove);
    BOOST_CHECK_EQUAL(result.ok(), true);

    /* verify that the key was properly moved */
    Command smembersNewSet(SMEMBERS);
    smembersNewSet.addArg("my-new-set");
    Command smembersDestSet(SMEMBERS);
    smembersDestSet.addArg("my-destination-set");

    result = connection.exec(smembersNewSet);
    BOOST_CHECK_EQUAL(result.ok(), true);
    reply = result.reply();
    BOOST_CHECK_EQUAL(reply.type(), ARRAY);
    BOOST_CHECK_EQUAL(reply.length(), 1);
    BOOST_CHECK_EQUAL(reply[0].type(), STRING);
    BOOST_CHECK_EQUAL(reply[0].asString(), "testkey1");

    result = connection.exec(smembersDestSet);
    BOOST_CHECK_EQUAL(result.ok(), true);
    reply = result.reply();
    BOOST_CHECK_EQUAL(reply.type(), ARRAY);
    BOOST_CHECK_EQUAL(reply.length(), 1);
    BOOST_CHECK_EQUAL(reply[0].type(), STRING);
    BOOST_CHECK_EQUAL(reply[0].asString(), "testkey2");

    /* check a SMOVE failure */
    result = connection.exec(smove);
    BOOST_CHECK_EQUAL(result.ok(), true);
    BOOST_CHECK_EQUAL(result.reply().type(), INTEGER);
    BOOST_CHECK_EQUAL(result.reply().asInt(), 0);

    /* SISMEMBER is supported, it checks for the membership of a
     * key to a set */
    Command sIsMemberTrue(SISMEMBER);
    sIsMemberTrue.addArg("my-new-set");
    sIsMemberTrue.addArg("testkey1");

    result = connection.exec(sIsMemberTrue);
    BOOST_CHECK_EQUAL(result.ok(), true);
    BOOST_CHECK_EQUAL(result.reply().type(), INTEGER);
    BOOST_CHECK_EQUAL(result.reply().asInt(), 1);

    Command sIsMemberFalse(SISMEMBER);
    sIsMemberFalse.addArg("my-new-set");
    sIsMemberFalse.addArg("testkey2");

    result = connection.exec(sIsMemberFalse);
    BOOST_CHECK_EQUAL(result.ok(), true);
    BOOST_CHECK_EQUAL(result.reply().type(), INTEGER);
    BOOST_CHECK_EQUAL(result.reply().asInt(), 0);

    /* EXISTS is supported it checks for the existance of a key */
    Command existsTrue(EXISTS);
    existsTrue.addArg("testkey2");

    result = connection.exec(existsTrue);
    BOOST_CHECK_EQUAL(result.ok(), true);
    BOOST_CHECK_EQUAL(result.reply().type(), INTEGER);
    BOOST_CHECK_EQUAL(result.reply().asInt(), 1);

    Command existsFalse(EXISTS);
    existsFalse.addArg("testkey4");

    result = connection.exec(existsFalse);
    BOOST_CHECK_EQUAL(result.ok(), true);
    BOOST_CHECK_EQUAL(result.reply().type(), INTEGER);
    BOOST_CHECK_EQUAL(result.reply().asInt(), 0);
}

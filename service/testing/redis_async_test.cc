/* redis_async_test.cc
   Jeremy Barnes, 1 December 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Test for our Redis class.
*/


#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "soa/service/redis.h"
#include "soa/service/testing/redis_temporary_server.h"
#include <boost/test/unit_test.hpp>
#include "jml/utils/smart_ptr_utils.h"
#include "jml/utils/string_functions.h"
#include "jml/arch/atomic_ops.h"
#include <boost/thread.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/function.hpp>
#include <iostream>
#include "jml/arch/atomic_ops.h"
#include "jml/arch/timers.h"
#include <linux/futex.h>
#include <unistd.h>
#include <sys/syscall.h>
#include "jml/arch/futex.h"

using namespace std;
using namespace Datacratic;
using namespace ML;

namespace Redis {

extern size_t requestDataCreated;
extern size_t requestDataDestroyed;
extern size_t eventLoopsCreated;
extern size_t eventLoopsDestroyed;

} // namespace Redis

using namespace Redis;


BOOST_AUTO_TEST_CASE( test_redis_async )
{
    RedisTemporaryServer redis;

    Redis::AsyncConnection connection(redis);
    
    boost::mutex m;
    m.lock();

    auto onResult = [&] (const Redis::Result & result)
        {
            if (result) {
                auto reply = result.reply();
                BOOST_CHECK_EQUAL(reply.type(), Redis::STATUS);
                BOOST_CHECK_EQUAL(reply.asString(), "OK");
                m.unlock();
            }
            else {
                BOOST_CHECK(false);
                cerr << "got error " << result.error() << endl;
                m.unlock();
            }
        };

    connection.queue(SET("hello", "world"), onResult);
    
    m.lock();

    BOOST_CHECK_EQUAL(connection.numRequestsPending(), 0);

    auto onResult2 = [&] (const Redis::Result & result)
        {
            if (result) {
                auto reply = result.reply();
                BOOST_CHECK_EQUAL(reply.type(), Redis::STRING);
                BOOST_CHECK_EQUAL(reply.asString(), "world");
                cerr << "got reply " << reply << endl;
                m.unlock();
            }
            else {
                BOOST_CHECK(false);
                cerr << "got error " << result.error() << endl;
                m.unlock();
            }
        };

    connection.queue(GET("hello"), onResult2);
    
    m.lock();

    BOOST_CHECK_EQUAL(connection.numRequestsPending(), 0);
    BOOST_CHECK_EQUAL(requestDataCreated, requestDataDestroyed);

    bool hadTimeout = false;

    auto onResult3 = [&] (const Redis::Result & result)
        {
            if (result) {
                m.unlock();
            }
            else {
                if (result.error() == "timeout")
                    hadTimeout = true;
                m.unlock();
            }
        };

    connection.queue(GET("hello"), onResult3, 0.0);
    
    m.lock();
    BOOST_CHECK(hadTimeout);
    BOOST_CHECK_EQUAL(connection.numRequestsPending(), 0);
    BOOST_CHECK_EQUAL(connection.numTimeoutsPending(), 0);
    BOOST_CHECK_EQUAL(requestDataCreated, requestDataDestroyed);
#if 0
    hadTimeout = false;

    connection.queue(onReply3, onError, 0.000001, onTimeout3, "GET hello");

    m.lock();
    ML::sleep(0.1);

    BOOST_CHECK(hadTimeout);
    BOOST_CHECK_EQUAL(connection.numRequestsPending(), 0);
    BOOST_CHECK_EQUAL(connection.numTimeoutsPending(), 0);
    BOOST_CHECK_EQUAL(requestDataCreated, requestDataDestroyed);
#endif
    hadTimeout = false;
    
    connection.queue(GET("hello"), onResult2, 10.0);

    m.lock();
    
    BOOST_CHECK(!hadTimeout);
    BOOST_CHECK_EQUAL(connection.numRequestsPending(), 0);
    BOOST_CHECK_EQUAL(connection.numTimeoutsPending(), 0);
    BOOST_CHECK_EQUAL(requestDataCreated, requestDataDestroyed);
    unsigned numReplies = 0;

    auto onResult4 = [&] (const Redis::Results & results)
        {
            BOOST_CHECK_EQUAL(results.size(), 5);
            if (!results) {
                cerr << "got reply " << results << endl;
                BOOST_CHECK(false);
                m.unlock();
            }
            else {
                numReplies++;
                cerr << "Got replies from multi command" << results << endl;
                m.unlock();
            }
        };

    vector<Redis::Command> commands = {
        MULTI,
        SET("lazy", "fox"),
        SET("jumped", "over"),
        SET("quick", "brown"),
        EXEC
    };

    connection.queueMulti(commands, onResult4);
    m.lock();
    BOOST_CHECK_EQUAL(numReplies, 1) ;

    commands = {
        GET("lazy"),
        GET("jumped"),
        GET("quick")
    };

    auto onResult5 = [&] (const Redis::Results & results)
        {
            BOOST_CHECK_EQUAL(results.size(), 3);
            if (!results) {
                cerr << "got reply " << results << endl;
                BOOST_CHECK(false);
                m.unlock();
            }
            else {
                cerr << "Got replies from multi command " << results << endl;
                cerr << results.reply(0) << endl;
                cerr << results.reply(1) << endl;
                cerr << results.reply(2) << endl;
                BOOST_CHECK_EQUAL(results.reply(0).asString(), "fox");
                BOOST_CHECK_EQUAL(results.reply(1).asString(), "over");
                BOOST_CHECK_EQUAL(results.reply(2).asString(), "brown");
                m.unlock();
            }
        };

    connection.queueMulti(commands, onResult5);
    m.lock();
    m.unlock();
}

#if 1
BOOST_AUTO_TEST_CASE( test_redis_mt )
{
    using namespace Redis;

    volatile bool finished = false;
    int nthreads = 4;

    RedisTemporaryServer redis;
    Redis::AsyncConnection connection(redis);

    uint64_t numErrors = 0;
    uint64_t numRequests = 0;

    auto doRedisThread = [&] (int threadNum)
        {
            cerr << "doRedisThread" << endl;

            volatile int pending = 0;
            int wait = 0;

            auto finishedRequest = [&] ()
            {
                if (__sync_add_and_fetch(&pending, -1) == 100)
                    futex_wake(wait);
            };
            
            auto onError = [&] (const std::string & error)
            {
                cerr << "error: " << error << endl;
                ML::atomic_inc(numErrors);
                finishedRequest();
            };
            
#if 0
            auto onTimeout = [&] ()
            {
                cerr << "got timeout" << endl;
                finishedRequest();
            };
#endif
            
            while (!finished) {
                //cerr << "doing request" << endl;
                int rand = random() % 100000;

                string key = ML::format("testkey%d", rand);
                Redis::AsyncConnection * redisPtr = &connection;

                auto onReply2 = [=] (const Redis::Result & result)
                {
                    if (!result)
                        onError(result.error());
                    else
                        finishedRequest();
                };

                auto onReply1 = [=] (const Redis::Result & result)
                {
                    if (!result)
                        onError(result.error());
                    else
                        redisPtr->queue(GET(key), onReply2, 5.0);
                };

                ML::atomic_inc(numRequests);
                if (__sync_fetch_and_add(&pending, 1) == 2000)
                    futex_wait(wait, 0);
                
                connection.queue(SET(key, rand), onReply1, 5.0);
            }

            while (pending != 0) ;
        };

    boost::thread_group tg;
        
    for (unsigned i = 0;  i < nthreads;  ++i)
        tg.create_thread(boost::bind<void>(doRedisThread, i));

    ML::sleep(1.0);
    finished = true;
    
    tg.join_all();

    cerr << "numRequests = " << numRequests << endl;

    BOOST_CHECK_EQUAL(connection.numRequestsPending(), 0);
    BOOST_CHECK_EQUAL(connection.numTimeoutsPending(), 0);
    BOOST_CHECK_EQUAL(requestDataCreated, requestDataDestroyed);

}
#endif

BOOST_AUTO_TEST_CASE( test_redis_timeout )
{
    RedisTemporaryServer redis;
    Redis::AsyncConnection connection(redis);

    {
        auto onResult = [&](const Redis::Result &result) {
            if (result) {
                auto reply = result.reply();
                BOOST_CHECK_EQUAL(reply.type(), Redis::STATUS);
                BOOST_CHECK_EQUAL(reply.asString(), "OK");
            }
            else {
                BOOST_CHECK(false);
            }
        };

        connection.queue(SET("Hello", "World"), onResult);
        ML::sleep(2.0);
    }

    std::cerr << "Suspending redis" << std::endl;
    redis.suspend();

    {

        auto result = connection.exec(GET("Hello"), 2.0);
        BOOST_CHECK(result.timedOut());

        auto onResult = [&](const Redis::Result &result) {
            BOOST_CHECK(result.timedOut());
        };

        connection.queue(GET("Hello"), onResult, 2.0);
        ML::sleep(5.0);
    }

    std::cerr << "Resuming redis" << std::endl;
    redis.resume();

    {
        auto onResult = [&](const Redis::Result &result) {
            if (result) {
                auto reply = result.reply();
                BOOST_CHECK_EQUAL(reply.type(), Redis::STRING);
                BOOST_CHECK_EQUAL(reply.asString(), "World");
            }
            else {
                BOOST_CHECK(false);
            }
        };

        connection.queue(GET("Hello"), onResult);
        ML::sleep(2.0);
    }

    redis.shutdown();
}

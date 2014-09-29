/* redis.cc
   Jeremy Barnes, 14 November 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Redis functionality.
*/

#include "soa/service/redis.h"
#include "jml/utils/guard.h"
#include <boost/thread.hpp>
#include <poll.h>
#include <unistd.h>
#include <fcntl.h>
#include "jml/arch/atomic_ops.h"
#include "jml/arch/backtrace.h"
#include "jml/arch/futex.h"
#include "jml/utils/vector_utils.h"


using namespace std;
using namespace Datacratic;
using namespace ML;




namespace Redis {


const Command PING("PING");
const Command HDEL("HDEL");
const Command HGET("HGET");
const Command HGETALL("HGETALL");
const Command HMGET("HMGET");
const Command HMSET("HMSET");
const Command WATCH("WATCH");
const Command MULTI("MULTI");
const Command EXEC("EXEC");
const Command EXISTS("EXISTS");
const Command HSET("HSET");
const Command HINCRBY("HINCRBY");
const Command KEYS("KEYS");             
const Command SET("SET");
const Command MSET("MSET");
const Command GET("GET");
const Command MGET("MGET");
const Command EXPIRE("EXPIRE");
const Command RANDOMKEY("RANDOMKEY");
const Command DEL("DEL");
const Command SADD("SADD");
const Command SMOVE("SMOVE");
const Command SMEMBERS("SMEMBERS");
const Command SISMEMBER("SISMEMBER");
const Command TTL("TTL");
const Command AUTH("AUTH");
const Command SELECT("SELECT");


/*****************************************************************************/
/* REPLY                                                                     */
/*****************************************************************************/


ReplyType
Reply::
type() const
{
    ExcAssert(r_);
    switch (r_->type) {
    case REDIS_REPLY_STATUS: return STATUS;
    case REDIS_REPLY_ERROR: return ERROR;
    case REDIS_REPLY_INTEGER: return INTEGER;
    case REDIS_REPLY_NIL: return NIL;
    case REDIS_REPLY_STRING: return STRING;
    case REDIS_REPLY_ARRAY: return ARRAY;
    default:
        throw ML::Exception("unknown Redis reply type %d", r_->type);
    };
}

std::string
Reply::
getString() const
{
    ExcAssert(r_);
    return std::string(r_->str, r_->str + r_->len);
}

std::string
Reply::
asString() const
{
    ExcAssert(r_);
    switch (r_->type) {
    case REDIS_REPLY_STATUS:
    case REDIS_REPLY_STRING:
    case REDIS_REPLY_ERROR: return getString();
    case REDIS_REPLY_INTEGER: return ML::format("%lli", r_->integer);
    case REDIS_REPLY_NIL: return "";
    case REDIS_REPLY_ARRAY: return asJson().toString();
    default:
        throw ML::Exception("unknown Redis reply type");
    };
}

long long
Reply::
asInt() const
{
    ExcAssert(r_);
    ExcAssertEqual(r_->type, REDIS_REPLY_INTEGER);
    return r_->integer;
}

long long
Reply::
asInt(long long defaultIfNotInteger)
{
    ExcAssert(r_);
    switch (r_->type) {
    case REDIS_REPLY_INTEGER: return r_->integer;
    case REDIS_REPLY_STRING: {
        std::string s = getString();
        char * end = 0;
        long long result = strtoll(s.c_str(), &end, 10);
        if (end != s.c_str() + s.length())
            return defaultIfNotInteger;
        return result;
    }
    case REDIS_REPLY_STATUS:
    case REDIS_REPLY_ERROR:
    case REDIS_REPLY_NIL:
    case REDIS_REPLY_ARRAY: return defaultIfNotInteger;
    default:
        throw ML::Exception("unknown Redis reply type");
    };
    
}

Json::Value
Reply::
asJson() const
{
    ExcAssert(r_);
    Json::Value result;
        
    switch (r_->type) {

    case REDIS_REPLY_STATUS:
        result["status"] = getString();
        return result;

    case REDIS_REPLY_ERROR:
        result["error"] = getString();
        return result;

    case REDIS_REPLY_INTEGER:
        result = (Json::Value::UInt)r_->integer;
        return result;

    case REDIS_REPLY_NIL:
        return result;

    case REDIS_REPLY_STRING:
        result = getString();
        return result;

    case REDIS_REPLY_ARRAY:
        for (unsigned i = 0;  i < r_->elements;  ++i)
            result[i] = (*this)[i].asJson();
        return result;
                
    default:
        throw ML::Exception("unknown Redis reply type ");
    };
            
}

Reply
Reply::
deepCopy() const
{
    return Reply(doDeepCopy(r_.get()), true);
}

redisReply *
Reply::
doDeepCopy(redisReply * r)
{
    redisReply * result = (redisReply *)malloc(sizeof(redisReply));
    memset(result, 0, sizeof(redisReply));
    result->type = r->type;

    try {
        switch (r->type) {

        case REDIS_REPLY_INTEGER:
        case REDIS_REPLY_NIL:
            result->integer = r->integer;
            break;
        case REDIS_REPLY_STATUS:
        case REDIS_REPLY_ERROR:
        case REDIS_REPLY_STRING: {
            // Copy the string
            char * str = (char *)malloc(r->len);
            result->str = str;
            result->len = r->len;
            std::copy(r->str, r->str + r->len, str);
            break;
        }
        case REDIS_REPLY_ARRAY: {
            // Copy the array
            redisReply ** arr
                = (redisReply **)malloc(r->elements * sizeof(redisReply *));
            for (unsigned i = 0;  i < r->elements;  ++i)
                arr[i] = 0;
            result->element = arr;
            result->elements = 0;
            for (unsigned i = 0;  i < r->elements;  ++i) {
                result->element[i] = doDeepCopy(r->element[i]);
                result->elements = i + 1;
            }
            break;
        }
            
        default:
            throw ML::Exception("unknown Redis reply type %d", r->type);
        };
    } catch (...) {
        freeReplyObject(result);
        throw;
    }

    return result;
}

std::ostream & operator << (std::ostream & stream, const Reply & reply)
{
    return stream << reply.asString();
}


/*****************************************************************************/
/* RESULTS                                                                   */
/*****************************************************************************/

const std::string
Result::timeoutError("timeout");

std::ostream & operator << (std::ostream & stream, const Result & result)
{
    if (result) return stream << result.reply().asString();
    else return stream << result.error();
}

std::ostream & operator << (std::ostream & stream, const Results & results)
{
    for (unsigned i = 0;  i < results.size();  ++i) {
        stream << "  " << i << ": " << results[i] << endl;
    }
    return stream;
}


/*****************************************************************************/
/* COMMAND                                                                   */
/*****************************************************************************/

#if 0
Command::
Command()
{
    va_list ap;
    va_start(ap, cmdFormat);
    ML::call_guard([&] () { va_end(ap); });

    return vqueue(onSuccess, onFailure, Date::notADate(), OnTimeout(),
                  cmdFormat, ap);
}
#endif

std::ostream & operator << (std::ostream & stream, const Command & command)
{
    return stream << command.formatStr << command.args;
}

/*****************************************************************************/
/* ADDRESS                                                                   */
/*****************************************************************************/

Address::
Address()
{
}

Address::
Address(const std::string & uri)
    : uri_(uri)
{
}

Address
Address::
tcp(const std::string & host, int port)
{
    if (host.find(':') != string::npos)
        throw ML::Exception("invalid host has a colon");
    if (host.find('/') != string::npos)
        throw ML::Exception("invalid host has a slash");

    return Address(host + ":" + to_string(port));
}

Address
Address::
unix(const std::string & path)
{
    if (path.find(':') != string::npos)
        throw ML::Exception("invalid path has a colon");
    return Address(path);
}

bool 
Address::
isUnix() const
{
    return !uri_.empty()
        && uri_.find(':') == string::npos;
}

bool
Address::
isTcp() const
{
    return !uri_.empty()
        && uri_.find(':') != string::npos;
    
}

std::string
Address::
unixPath() const
{
    if (!isUnix())
        throw ML::Exception("address is not unix");
    return uri_;
}

std::string
Address::
tcpHost() const
{
    if (!isTcp())
        throw ML::Exception("address is not tcp");
    auto pos = uri_.find(':');
    ExcAssertNotEqual(pos, string::npos);
    return string(uri_, 0, pos);
}

int
Address::
tcpPort() const
{
    if (!isTcp())
        throw ML::Exception("address is not tcp");
    auto pos = uri_.find(':');
    ExcAssertNotEqual(pos, string::npos);
    return boost::lexical_cast<int>(string(uri_, pos + 1));
}

std::string
Address::
uri() const
{
    return uri_;
}


/*****************************************************************************/
/* ASYNC CONNECTION                                                          */
/*****************************************************************************/

enum {
    WAITING = 0,
    REPLIED = 1,
    TIMEDOUT = 2
};

size_t requestDataCreated = 0;
size_t requestDataDestroyed = 0;

struct AsyncConnection::RequestData
    : std::enable_shared_from_this<AsyncConnection::RequestData> {
    RequestData()
    {
        ML::atomic_inc(requestDataCreated);
    }

    ~RequestData()
    {
        ML::atomic_inc(requestDataDestroyed);
    }

    OnResult onResult;
    std::string command;
    Date timeout;
    AsyncConnection * connection;
    int64_t id;
    Requests::iterator requestIterator;
    Timeouts::iterator timeoutIterator;
    int state;
};

size_t eventLoopsCreated = 0;
size_t eventLoopsDestroyed = 0;

struct AsyncConnection::EventLoop {

    int wakeupfd[2];
    volatile bool finished;
    AsyncConnection * connection;
    std::shared_ptr<boost::thread> thread;
    pollfd fds[2];
    volatile int disconnected;

    EventLoop(AsyncConnection * connection)
        : finished(false), connection(connection), disconnected(1)
    {
        ML::atomic_inc(eventLoopsCreated);
        
        int res = pipe2(wakeupfd, O_NONBLOCK);
        if (res == -1)
            throw ML::Exception(errno, "pipe2");

        //cerr << "connection on fd " << connection->context_->c.fd << endl;


        fds[0].fd = wakeupfd[0];
        fds[0].events = POLLIN;
        fds[1].fd = connection->context_->c.fd;
        fds[1].events = 0;

        registerMe(connection->context_);

        thread.reset(new boost::thread(boost::bind(&EventLoop::run, this)));

#if 0
        char buf[1];
        res = read(wakeupfd[0], buf, 1);
        if (res == -1)
            throw ML::Exception(errno, "read");
#endif
    }

    ~EventLoop()
    {
        //cerr << "DESTROYING EVENT LOOP" << endl;
        ML::atomic_inc(eventLoopsDestroyed);
        shutdown();
    }

    void shutdown()
    {
        if (!thread) return;

        finished = true;
        wakeup();
        thread->join();
        thread.reset();
        ::close(wakeupfd[0]);
        ::close(wakeupfd[1]);
    }

    void wakeup()
    {
        int res = write(wakeupfd[1], "x", 1);
        if (res == -1)
            throw ML::Exception("error waking up fd %d: %s", wakeupfd[1],
                                strerror(errno));
    }
    
    void registerMe(redisAsyncContext * context)
    {
        //cerr << "called registerMe" << endl;
        redisAsyncSetConnectCallback(context, onConnect);
        redisAsyncSetDisconnectCallback(context, onDisconnect);

        context->ev.data = context->data = this;
        context->ev.addRead = startReading;
        context->ev.delRead = stopReading;
        context->ev.addWrite = startWriting;
        context->ev.delWrite = stopWriting;
        context->ev.cleanup = cleanup;
    }

    void run()
    {
        //wakeup();

        //cerr << this << " starting run loop" << endl;

        while (!finished) {
            //sleep(1);
            Date now = Date::now();

            if (connection->earliestTimeout < now)
                connection->expireTimeouts(now);

            double timeLeft = now.secondsUntil(connection->earliestTimeout);

            //cerr << "timeLeft = " << timeLeft << endl;
            //cerr << "fds[0].events = " << fds[0].events << endl;
            //cerr << "fds[1].events = " << fds[1].events << endl;

            int timeout = std::min(1000.0,
                                   std::max<double>(0, 1000 * timeLeft));

            if (connection->earliestTimeout == Date::positiveInfinity())
                timeout = 1000000;

            //cerr << "looping; fd0 = " << fds[1].fd << " timeout = "
            //     << timeout << endl;

            int res = poll(fds, 2, timeout);
            if (res == -1 && errno != EINTR) {
                cerr << "poll() error: " << strerror(errno) << endl;
            }
            if (res == 0) continue;  // just a timeout; loop around again

            //cerr << "poll() returned " << res << endl;

            if (fds[0].revents & POLLIN) {
                //cerr << "got wakeup" << endl;
                char buf[128];
                int res = read(fds[0].fd, buf, 128);
                if (res == -1)
                    throw ML::Exception(errno, "read from wakeup pipe");
                //cerr << "woken up with " << res << " messages" << endl;
            }
            if ((fds[1].revents & POLLOUT)
                && (fds[1].events & POLLOUT)) {
                //cerr << "got write on " << fds[1].fd << endl;
                boost::unique_lock<Lock> guard(connection->lock);
                redisAsyncHandleWrite(connection->context_);
            }
            if ((fds[1].revents & POLLIN)
                && (fds[1].events & POLLIN)) {
                //cerr << "got read on " << fds[1].fd << endl;
                boost::unique_lock<Lock> guard(connection->lock);
                redisAsyncHandleRead(connection->context_);
            }

            // Now we don't have the lock anymore, do our callbacks
            while (!connection->replyQueue.empty()) {
                try {
                    connection->replyQueue.front()();
                } catch (...) {
                    cerr << "warning: redis callback threw" << endl;
                    //abort();
                }
                connection->replyQueue.pop_front();
            }
        }

        if (!disconnected) {
            // Disconnect
            //cerr << this << " calling redisAsyncDisconnect" << endl;
            redisAsyncDisconnect(connection->context_);
            //redisAsyncFree(connection->context_);
            //cerr << this << " done redisAsyncDisconnect" << endl;
        }
        
        // Wait until we get the callback
        while (!disconnected) {
            futex_wait(disconnected, 0);
        }

        //cerr << this << " now disconnected" << endl;
    }

    static void onConnect(const redisAsyncContext * context, int status)
    {
        EventLoop * eventLoop = reinterpret_cast<EventLoop *>(context->data);

        if (status == REDIS_OK)
            eventLoop->onConnect(status);
        else {
            /* This function will be called with an error status if the
               connection failed.  For us it's like a disconnection, so
               we call into the disconnect code.
            */
            cerr << "onConnect: code = " << status << " err = " << context->err
                 << " errstr = " << context->errstr << " errno = "
                 << strerror(errno) << endl;
            eventLoop->onDisconnect(status);
        }
    }

    void onConnect(int status)
    {
        //cerr << "connection on fd " << connection->context_->c.fd << endl;
        //cerr << "status = " << status << endl;
        fds[1].fd = connection->context_->c.fd;
        wakeup();
        disconnected = 0;
        futex_wake(disconnected);
    }

    static void onDisconnect(const redisAsyncContext * context, int status)
    {
        //cerr << "onDisconnect" << endl;
        EventLoop * eventLoop = reinterpret_cast<EventLoop *>(context->data);
        eventLoop->onDisconnect(status);
    }

    void onDisconnect(int status)
    {
        if (status != REDIS_OK) {
            cerr << "disconnection with status " << status << endl;
            cerr << "onConnect: code = " << status << " err = "
                 << connection->context_->err
                 << " errstr = "
                 << connection->context_->errstr << " errno = "
                 << strerror(errno) << endl;
        }
        disconnected = 1;
        futex_wake(disconnected);
    }

    static void startReading(void * privData)
    {
        EventLoop * eventLoop = reinterpret_cast<EventLoop *>(privData);
        eventLoop->startReading();
    }

    void startReading()
    {
        //cerr << "start reading" << endl;
        //if (fds[1].events & POLLIN) return;  // already reading
        fds[1].events |= POLLIN;
        wakeup();
    }

    static void stopReading(void * privData)
    {
        EventLoop * eventLoop = reinterpret_cast<EventLoop *>(privData);
        eventLoop->stopReading();
    }

    void stopReading()
    {
        //cerr << "stop reading" << endl;
        fds[1].events &= ~POLLIN;
    }

    static void startWriting(void * privData)
    {
        EventLoop * eventLoop = reinterpret_cast<EventLoop *>(privData);
        eventLoop->startWriting();
    }

    void startWriting()
    {
        //cerr << "start writing" << endl;
        //if (fds[1].events & POLLOUT) return;  // already reading
        fds[1].events |= POLLOUT;
        wakeup();
    }

    static void stopWriting(void * privData)
    {
        EventLoop * eventLoop = reinterpret_cast<EventLoop *>(privData);
        eventLoop->stopWriting();
    }

    void stopWriting()
    {
        //cerr << "stop writing" << endl;
        fds[1].events &= ~POLLOUT;
    }

    static void cleanup(void * privData)
    {
        EventLoop * eventLoop = reinterpret_cast<EventLoop *>(privData);
        eventLoop->cleanup();
    }

    void cleanup()
    {
        //cerr << this << " doing cleanup" << endl;
        //backtrace();
    }
};


AsyncConnection::
AsyncConnection()
    : context_(0), idNum(0)
{
}

AsyncConnection::
AsyncConnection(const Address & address)
    : context_(0), idNum(0)
{
    connect(address);
}

AsyncConnection::
~AsyncConnection()
{
    close();
}

void
AsyncConnection::
connect(const Address & address)
{
    //cerr << "connecting to redis " << address.uri() << endl;

    close();

    this->address = address;

    if (address.isTcp()) {
        context_ = redisAsyncConnect(address.tcpHost().c_str(),
                                     address.tcpPort());
    }
    else if (address.isUnix()) {
        context_ = redisAsyncConnectUnix(address.unixPath().c_str());
    }
    else throw ML::Exception("cannot connect to address that is neither tcp "
                             "or unix");
    checkError("connect");

    eventLoop.reset(new EventLoop(this));
}

void
AsyncConnection::
test()
{
    int done = 0;

    string error;
    
    auto onResponse = [&] (const Redis::Result & result)
        {
            if (result) {
                //cerr << "got reply " << result.reply() << endl;
            }
            else error = result.error();
            done = 1;
            futex_wake(done);
        };

    queue(PING, onResponse, 2.0);

    while (!done)
        futex_wait(done, 0);
    
    if (error != "")
        throw ML::Exception("couldn't connect to Redis: " + error);
}

void
AsyncConnection::
auth(std::string password)
{
    int done = 0;

    string error;

    auto onResponse = [&] (const Redis::Result & result)
        {
            if (result) {
                std::cout << "got reply " << result.reply() << endl;
            }
            else error = result.error();
            done = 1;
            futex_wake(done);
        };

    Command authCmd(AUTH);
    authCmd.addArg(password);
    queue(authCmd, onResponse, 2.0);

    while (!done)
        futex_wait(done, 0);

    if (error != "")
        throw ML::Exception("couldn't authenticate redis connection: " + error);
}

void
AsyncConnection::
select(int database)
{
    int done = 0;

    string error;

    auto onResponse = [&] (const Redis::Result & result)
        {
            if (result) {
                std::cout << "got reply " << result.reply() << endl;
            }
            else error = result.error();
            done = 1;
            futex_wake(done);
        };

    Command cmd(SELECT);
    cmd.addArg(database);
    queue(cmd, onResponse, 2.0);

    while (!done)
        futex_wait(done, 0);

    if (error != "")
        throw ML::Exception("couldn't select redis database: " + error);
}



void
AsyncConnection::
close()
{
    if (!context_) return;

    if (eventLoop) {
        eventLoop->shutdown();
        eventLoop.reset();
    }
    
    context_ = 0;
}

void
AsyncConnection::
resultCallback(redisAsyncContext * context, void * reply, void * privData)
{
    //cerr << "resultCallback with reply " << reply << " privData "
    //     << privData << endl;
    //cerr << "context->err = " << context->err << endl;
    //cerr << "context->errstr = " << context->errstr << endl;
    //cerr << "context->c.errstr = " << context->c.errstr << endl;

    ExcAssert(privData);

    // Get a shared pointer to our data so it doesn't go away
    RequestData * dataRawPtr
        = reinterpret_cast<RequestData *>(privData);
    std::shared_ptr<RequestData> data
        = dataRawPtr->shared_from_this();

    //cerr << "command " << data->command << endl;
    //cerr << "reply " << reply << endl;

    // Remove from data structures
    AsyncConnection * c = data->connection;

    {
        boost::unique_lock<Lock> guard(c->lock);
        
        if (data->requestIterator != c->requests.end()) {
            c->requests.erase(data->requestIterator);
            data->requestIterator = c->requests.end();
        }
        
        if (data->timeoutIterator != c->timeouts.end()) {
            c->timeouts.erase(data->timeoutIterator);
            data->timeoutIterator = c->timeouts.end();
            if (c->timeouts.empty())
                c->earliestTimeout = Date::positiveInfinity();
            else c->earliestTimeout = c->timeouts.begin()->first;
        }

        if (data->state != WAITING) return;  // raced; timeout happened
        data->state = REPLIED;
    }


    Result result;

    if (reply) {
        Reply replyObj((redisReply *)reply, false /* take ownership */);
        //cerr << "reply = " << replyObj << endl;
        if (replyObj.type() == ERROR) {
            // Command error, return it
            result = Result(replyObj.asString());
        }
        else {
            // Result (success)
            // We have to take a deep copy since the reply object is owned
            // by the caller
            result = Result(replyObj.deepCopy());
        }
    }
    else {
        // Context encountered an error; return it
        result = Result(data->connection->context_->errstr);
    }

    // Queue up a reply object so it can be called without the lock held.  If
    // we call directly from here, then the lock has to be held and so deadlock
    // is possible.
    c->replyQueue.push_back(std::bind(data->onResult, result));
}

int64_t
AsyncConnection::
queue(const Command & command,
      const OnResult & onResult,
      Timeout timeout)
{
    boost::unique_lock<Lock> guard(lock);

    ExcAssert(context_);
    ExcAssert(!context_->err);
    
    // Check basics
    if (timeout.expiry.isADate() && Date::now() >= timeout.expiry) {
        onResult(Result(Result::timeoutError));
        return -1;
    }

    int64_t id = idNum++;
    
    // Create data structure to be passed around
    std::shared_ptr<RequestData> data(new RequestData);
    data->onResult = onResult;
    data->timeout = timeout.expiry;
    data->command = command.formatStr;
    //data->timeout = Date::notADate();
    data->connection = this;
    data->id = id;
    data->requestIterator = requests.end();
    data->timeoutIterator = timeouts.end();
    data->state = WAITING;

    ExcAssertEqual(requests.count(id), 0);

    //cerr << "data = " << data << endl;

    auto it = requests.insert(make_pair(id, data)).first;
    data->requestIterator = it;

    // Does the servicing thread possibly need to be woken up?
    bool needWakeup = false;
    
    if (timeout.expiry.isADate()) {
        needWakeup = !timeouts.empty()
            && timeout.expiry < timeouts.begin()->first;
        data->timeoutIterator = timeouts.insert(make_pair(timeout.expiry, it));
        earliestTimeout = timeouts.begin()->first;
    }
    
    vector<const char *> argv = command.argv();
    vector<size_t> argl = command.argl();

    int result = redisAsyncCommandArgv(context_, resultCallback, data.get(),
                                       command.argc(),
                                       &argv[0],
                                       &argl[0]);
    
    if (result != REDIS_OK) {
        //cerr << "result not OK" << endl;
        resultCallback(context_, 0, data.get());
        return -1;
    }
    
    if (needWakeup)
        eventLoop->wakeup();
    
    return id;
}

Result
AsyncConnection::
exec(const Command & command, Timeout timeout)
{
    Result result;
    int done = 0;

    auto onResponse = [&] (const Redis::Result & redisResult)
        {
            result = redisResult.deepCopy();
            done = 1;
            futex_wake(done);
        };

    queue(command, onResponse, timeout);

    while (!done)
        futex_wait(done, 0);
 
    return result;
}

struct AsyncConnection::MultiAggregator
    : public Results {

    MultiAggregator(int size,
                    const OnResults & onResults)
        : numDone(0), onResults(onResults)
    {
        resize(size);
    }
    
    // Something succeeded
    void result(int i, const Result & result)
    {
        at(i) = result.deepCopy();
        finish();
    }

    void finish()
    {
        if (__sync_add_and_fetch(&numDone, 1) != size())
            return;
        onResults(*this);
    }
    
    int numDone;
    OnResults onResults;
};

void
AsyncConnection::
queueMulti(const std::vector<Command> & commands,
           const OnResults & onResults,
           Timeout timeout)
{
    if (commands.empty())
        throw ML::Exception("can't call queueMulti with an empty list "
                            "of commands");
    
    auto results
        = std::make_shared<MultiAggregator>(commands.size(), onResults);
    
    // Make sure they all get executed as a block
    boost::unique_lock<Lock> guard(lock);
    
    // Now queue them one at a time
    for (unsigned i = 0;  i < commands.size();  ++i) {
        queue(commands[i],
              std::bind(&MultiAggregator::result, results, i,
                        std::placeholders::_1),
              timeout);
    }
}

Results
AsyncConnection::
execMulti(const std::vector<Command> & commands, Timeout timeout)
{
    Results results;
    int done = 0;

    auto onResponse = [&] (const Redis::Results & redisResults)
        {
            results = redisResults;
            done = 1;
            futex_wake(done);
        };

    queueMulti(commands, onResponse, timeout);

    while (!done)
        futex_wait(done, 0);
 
    return results;
}

void
AsyncConnection::
cancel(int handle)
{
    throw ML::Exception("AsyncConnection::cancel(): not done");
}

void
AsyncConnection::
expireTimeouts(Date now)
{
    boost::unique_lock<Lock> guard(lock);
    
    auto it = timeouts.begin(), end = timeouts.end();
    for (;  it != end;  ++it) {
        if (it->first > now) break;

        auto resultIt = it->second;
        auto data = resultIt->second;

        data->state = TIMEDOUT;
        data->onResult(Result(Result::timeoutError));
        data->timeoutIterator = end;

        // Let it be cleaned up from hiredis once it's finished
    }

    timeouts.erase(timeouts.begin(), it);

    if (timeouts.empty())
        earliestTimeout = Date::positiveInfinity();
    else earliestTimeout = timeouts.begin()->first;
}

} // namespace Redis

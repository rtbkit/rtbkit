/* redis.h                                                         -*- C++ -*-
   Jeremy Barnes, 14 November 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Connection to redis.
*/

#ifndef __redis__redis_h__
#define __redis__redis_h__

#include <hiredis/hiredis.h>
#include <hiredis/async.h>
#include <string>
#include "jml/arch/exception.h"
#include "jml/utils/exc_assert.h"
#include "jml/utils/string_functions.h"
#include "jml/utils/unnamed_bool.h"
#include "jml/utils/ring_buffer.h"
#include "soa/jsoncpp/json.h"
#include "soa/types/date.h"
#include <boost/shared_ptr.hpp>
#include <boost/function.hpp>
#include <boost/thread/thread.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <deque>


namespace Redis {


using Datacratic::Date;


enum ReplyType {
    STATUS,
    ERROR,
    INTEGER,
    NIL,
    STRING,
    ARRAY
};


/*****************************************************************************/
/* REPLY                                                                     */
/*****************************************************************************/

struct Reply {

    static void doDelete(redisReply * reply)
    {
        freeReplyObject(reply);
    }
    
    static void noDelete(redisReply * reply)
    {
    }

    Reply()
    {
    }

    Reply(redisReply * reply, bool needDelete)
        : r_(reply, needDelete ? doDelete : noDelete)
    {
    }

    bool initialized() const { return !!r_; }

    /** Makes a deep copy that we have ownership of. */
    Reply deepCopy() const;

    static redisReply * doDeepCopy(redisReply * r);

    operator std::string () const
    {
        return asString();
    }

    operator long long int () const
    {
        return asInt();
    }

    operator Json::Value () const
    {
        return asJson();
    }

    ReplyType type() const;

    std::string getString() const;

    std::string asString() const;
        
    long long asInt() const;

    long long asInt(long long defaultIfNotInteger);

    Json::Value asJson() const;

    Reply operator [] (size_t index) const
    {
        ExcAssert(r_);
        ExcAssertLess(index, length());
        return Reply(r_->element[index], false);
    }

    ssize_t length() const
    {
        ExcAssert(r_);
        ExcAssertEqual(r_->type, REDIS_REPLY_ARRAY);
        return r_->elements;
    }

private:
    std::shared_ptr<redisReply> r_;
    bool needDelete_;
};

std::ostream & operator << (std::ostream & stream, const Reply & reply);


/** Result of a command, with both a reply and an error code.  If there is
    a timeout, then the error code will be "timeout".
*/
struct Result {
    Result()
    {
    }

    Result(const std::string & error)
        : error_(error)
    {
    }

    Result(const Reply & reply)
        : reply_(reply)
    {
    }

    Result deepCopy() const
    {
        if (ok())
            return Result(reply_.deepCopy());
        else return Result(error_);
    }

    Reply reply_;
    std::string error_;

    /** The error field of a command result will be equal to this if there
        was a timeout.
    */
    static const std::string timeoutError;
    
    bool ok() const { return error_.empty(); }
    
    bool timedOut() const { return error_ == timeoutError; }

    JML_IMPLEMENT_OPERATOR_BOOL(ok());

    const Reply & reply() const
    {
        if (!error_.empty())
            throw ML::Exception("attempt to read reply from Redis result with"
                                "error " + error_);
        ExcAssert(reply_.initialized());
        return reply_;
    }

    const std::string & error() const
    {
        return error_;
    }
};

std::ostream & operator << (std::ostream & stream, const Result & result);


/** Result of multiple commands. */
struct Results : public std::vector<Result> {
    bool ok() const
    {
        for (auto & r: *this)
            if (!r)
                return false;
        return true;
    }

    JML_IMPLEMENT_OPERATOR_BOOL(ok());

    const Reply & reply(int index) const
    {
        return at(index).reply();
    }

    std::string error() const
    {
        for (auto & r: *this)
            if (!r)
                return r.error();
        return "";
    }

    bool timedOut() const
    {
        for (auto & r: *this)
            if (r.timedOut())
                return true;
        return false;
    }
};

std::ostream & operator << (std::ostream & stream, const Results & results);


/*****************************************************************************/
/* COMMAND                                                                   */
/*****************************************************************************/

struct Command {
    Command()
    {
    }

    //explicit Command(const char * args, ...);

    template<typename... Args>
    Command(std::string command, Args &&... args)
        : formatStr(std::move(command))
    {
        addArgs(std::forward<Args>(args)...);
    }

    Command(const std::string & cmd,
            const std::initializer_list<std::string> & args)
        : formatStr(std::move(cmd))
    {
        for (auto arg: args)
            addArg(arg);
    }
    
    std::string formatStr;
    std::vector<std::string> args;

    //std::string formatted() const;

    void addArg(std::string arg)
    {
        args.push_back(std::move(arg));
    }

    void addArg(int64_t arg)
    {
        args.push_back(std::to_string(arg));
    }

    template<typename Arg, typename... Args>
    void addArgs(Arg && head, Args &&... tail)
    {
        addArg(std::forward<Arg>(head));
        addArgs(std::forward<Args>(tail)...);
    }

    void addArgs()
    {
    }

    template<typename... Args>
    Command operator () (Args &&... args) const
    {
        auto result = *this;
        result.addArgs(std::forward<Args>(args)...);
        return result;
    }
    
    int argc() const
    {
        return args.size() + 1;
    }
    
    std::vector<const char *> argv() const
    {
        std::vector<const char *> result;
        result.push_back(formatStr.c_str());
        for (const std::string & arg: args)
            result.push_back(arg.c_str());
        return result;
    }

    std::vector<size_t> argl() const
    {
        std::vector<size_t> result;
        result.push_back(formatStr.length());
        for (const std::string & arg: args)
            result.push_back(arg.length());
        return result;
    }
};

std::ostream & operator << (std::ostream & stream, const Command & command);

// Commands ready to be constructed

extern const Command PING;
extern const Command HDEL;
extern const Command HGET;
extern const Command HGETALL;
extern const Command HMGET;
extern const Command HMSET;
extern const Command WATCH;
extern const Command MULTI;
extern const Command EXEC;
extern const Command EXISTS;
extern const Command HSET;
extern const Command HINCRBY;
extern const Command KEYS;                           
extern const Command SET;
extern const Command MSET;
extern const Command GET;
extern const Command MGET;
extern const Command EXPIRE;
extern const Command RANDOMKEY;
extern const Command DEL;
extern const Command SADD;
extern const Command SMOVE;
extern const Command SMEMBERS;
extern const Command SISMEMBER;
extern const Command TTL;
extern const Command AUTH;
extern const Command SELECT;

/*****************************************************************************/
/* ADDRESS                                                                   */
/*****************************************************************************/

/** Identifies an address to connect to a Redis service. */

struct Address {
    Address();
    Address(const std::string & uri);

    static Address tcp(const std::string & host, int port);
    static Address unix(const std::string & path);

    bool isUnix() const;
    bool isTcp() const;

    std::string unixPath() const;
    std::string tcpHost() const;
    int tcpPort() const;

    std::string uri() const;

    std::string uri_;
};


/*****************************************************************************/
/* ASYNC CONNECTION                                                          */
/*****************************************************************************/

/** Asynchronous connection to Redis. */

struct AsyncConnection {
    
    AsyncConnection();
    
    AsyncConnection(const Address & address);

    ~AsyncConnection();

    void connect(const Address & address);

    /** Test the connection by sending a ping and waiting for the response.
        This is synchronous.  Once this method returns, it is sure that
        the connection works.
    */
    void test();
    void auth(std::string password);
    void select(int database);

    void close();

    // Struct to specify a timeout, either absolute or relative
    struct Timeout {
        Timeout() // no timeout
            : expiry(Date::positiveInfinity())
        {
        }

        Timeout(double relativeTime)
            : expiry(Date::now().plusSeconds(relativeTime))
        {
        }

        Timeout(Date absoluteTime)
            : expiry(absoluteTime)
        {
        }

        Date expiry;
    };

    typedef boost::function<void (const Result &)> OnResult;
    typedef boost::function<void (const Results &)> OnResults;

    /** Queue an asynchronous command with a timeout.  Returns a handle that
        can be used to cancel the command (which means ignore the result).
    */
    int64_t queue(const Command & command,
                  const OnResult & onResult = OnResult(),
                  Timeout timeout = Timeout());

    /** Execute synchronously. */
    Result exec(const Command & command, Timeout timeout = Timeout());

    /** Queue a list of asynchronous commands atomically with a timeout. */
    void queueMulti(const std::vector<Command> & commands,
                    const OnResults & onResults = OnResults(),
                    Timeout timeout = Timeout());
    
    /** Execute multiple commands synchronously. */
    Results execMulti(const std::vector<Command> & command,
                      Timeout timeout = Timeout());
    
    /** Cancel the given command. */
    void cancel(int handle);
    
    size_t numRequestsPending() const
    {
        return requests.size();
    }

    size_t numTimeoutsPending() const
    {
        return timeouts.size();
    }
    
private:
    std::deque<std::function<void ()> > replyQueue;
    
    static void resultCallback(redisAsyncContext * context, void *, void *);

    struct RequestData;

    typedef boost::recursive_mutex Lock;
    Lock lock;

    typedef std::map<uint64_t, std::shared_ptr<RequestData> > Requests;
    Requests requests;
    
    typedef std::multimap<Datacratic::Date, Requests::iterator> Timeouts;
    Timeouts timeouts;

    /** Called when something knows that at least one timeout is expired;
        expire them.
    */
    void expireTimeouts(Datacratic::Date now);

    Datacratic::Date earliestTimeout;

    void checkError(const char * command)
    {
        if (!context_)
            throw ML::Exception("no connection to Redis");

        if (context_->err) {
            throw ML::Exception("Redis command %s returned error %s",
                                command, context_->errstr);
        }
    }

    Address address;
    redisAsyncContext * context_;
    int64_t idNum;

    struct EventLoop;
    std::shared_ptr<EventLoop> eventLoop;

    struct MultiAggregator;
};

} // namespace Datacratic

#endif /* __redis__redis_h__ */

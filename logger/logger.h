/* logger.h                                                        -*- C++ -*-
   Jeremy Barnes, 20 May 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

*/

#pragma once

#include "soa/service/zmq_named_pub_sub.h"
#include "soa/service/zmq_utils.h"
#include "soa/service/socket_per_thread.h"
#include <sstream>
#include "jml/utils/filter_streams.h"
#include <boost/thread/thread.hpp>
#include "jml/utils/smart_ptr_utils.h"
#include "jml/utils/vector_utils.h"
#include "jml/arch/atomic_ops.h"
#include "ace/Synch.h"
#include <boost/function.hpp>
#include <boost/regex.hpp>
#include <boost/shared_ptr.hpp>
#include "soa/jsoncpp/json.h"


namespace Datacratic {


/*****************************************************************************/
/* LOG OUTPUT                                                                */
/*****************************************************************************/

/** Abstract class that takes the log input and does something with it. */

struct LogOutput {
    LogOutput()
    {
    }

    virtual ~LogOutput()
    {
    }

    /** This is the method that will be called whenever we need to log
        a message to something.
    */
    virtual void logMessage(const std::string & channel,
                            const std::string & message) = 0;

    /** Should close whatever resources are being used by the output
        and join any threads that it's created.
    */
    virtual void close() = 0;

    /** Should return a JSON object containing whatever stats have been
        accumulated since last time clearStats() was called.

        Default returns an empty JSON object.
    */
    virtual Json::Value stats() const;

    /** Clears the current value of stats.  Default implementation does
        nothing.
    */
    virtual void clearStats();
};


/*****************************************************************************/
/* LOGGER                                                                    */
/*****************************************************************************/

/** This is a class that:
    1.  Accepts log messages from any thread;
    2.  Forwards the messages to a single destination.

    Everything is entirely thread-safe and in normal operation, logging a
    message will not block.  This allows it to be used in contexts where
    logging happens in a time-critical loop, for example.
*/

struct Logger {

    /** Create a logger with its own zeromq context. */
    Logger();

    /** Create a logger using the given zeromq context. */
    Logger(zmq::context_t & context);

    /** Create a logger using the given shared zeromq context. */
    Logger(std::shared_ptr<zmq::context_t> & context);

    ~Logger();

    void init();

    /** Subscribe to the given stream to get log messages from.  Optionally,
        a filter can also be set up to limit the messages to a given
        subset.

        The identity argument sets the zeromq identity of the connecting socket.
        If this is set, then messages for the logger will be queued until it
        reconnects so that no messages are lost.  If no identity is given then
        messages that are sent while the socket is down will be lost.
    */
    void subscribe(const std::string & uri,
                   const std::vector<std::string> & channels,
                   const std::string & identity = "");

    /** Tell where to log to.  The place it goes depends upon the URI:
        - file://path: log to the filename; if it finishes in "+" then it is
          appended;
        - ipc://path: publish to the given zeromq socket;
        - tcp://hostname: send over tcp/ip
    */
    void logTo(const std::string & uri,
               const boost::regex & allowChannels = boost::regex(),
               const boost::regex & denyChannels = boost::regex(),
               double logProbability = 1.0);

    /** Set the output to the given object.  Note that this is complicated
        by the necessity to stop the logging thread until the operation
        has been completed.

        ONLY ONE THREAD AT A TIME MUST BE IN ADDOUTPUT OR CLEAROUTPUTS. 
        There is no locking to avoid them causing problems for each other,
        just enough locking to stop them interfering with the event loop.
    */
    void addOutput(std::shared_ptr<LogOutput> output,
                   const boost::regex & allowChannels = boost::regex(),
                   const boost::regex & denyChannels = boost::regex(),
                   double logProbability = 1.0);

    /** Set up a callback that will call the given function when a message
        matching the filter is obtained.
    */
    void addCallback(boost::function<void (std::string, std::string)> callback,
                     const boost::regex & allowChannels = boost::regex(),
                     const boost::regex & denyChannels = boost::regex(),
                     double logProbability = 1.0);

    /** Clear all outputs. */
    void clearOutputs();

    /** Log a given message to the given channel.  Each of the arguments will
        be converted to a string and logged like that.
    */
    template<typename... Args>
    void operator () (const std::string & channel, Args... args)
    {
        if (!outputs) return;
        ML::atomic_add(messagesSent, 1);
        sendMessage(logSocket(), channel, Date::now().print(5), args...);
    }

    template<typename... Args>
    void logMessage(const std::string & channel, Args... args)
    {
        if (!outputs) return;
        ML::atomic_add(messagesSent, 1);
        sendMessage(logSocket(), channel, Date::now().print(5), args...);
    }

    template<typename... Args>
    void logMessageNoTimestamp(const std::string & channel, Args... args)
    {
        if (!outputs) return;
        ML::atomic_add(messagesSent, 1);
        sendMessage(logSocket(), channel, args...);
    }

    void logMessageNoTimestamp(const std::vector<std::string> & message)
    {
        if (!outputs) return;

        if (message.empty())
            throw ML::Exception("can't log empty message");
        ML::atomic_add(messagesSent, 1);
        sendAll(logSocket(), message);
    }

    template<typename GetEl>
    void logMessage(const std::string & channel,
                    int numElements,
                    GetEl getElement)
    {
        if (!outputs) return;

        zmq::socket_t & sock = logSocket();
        ML::atomic_add(messagesSent, 1);
        sendMesg(sock, channel, ZMQ_SNDMORE);
        sendMesg(sock, Date::now().print(5), numElements ? ZMQ_SNDMORE : 0);

        for (unsigned i = 0;  i < numElements;  ++i) {
            std::string el = getElement(i);
            sendMesg(sock, el, i < numElements - 1 ? ZMQ_SNDMORE : 0);
        }
    }

    void start(boost::function<void ()> onStop = boost::function<void ()>());

    void waitUntilFinished();

    void shutdown();
    
    std::map<std::string, size_t> getStats();
    void resetStats();

    /// Replay the events in the given filename through the logger
    void replay(const std::string & filename,
                ssize_t maxEvents = -1) const;

    /// Replay directly without going through zmq.  start() cannot have
    /// been called.;
    void replayDirect(const std::string & filename,
                      ssize_t maxEvents = -1) const;
    
    uint64_t numMessagesSent() const { return messagesSent; }
    uint64_t numMessagesDone() const { return messagesDone; }

    void handleListenerMessage(std::vector<zmq::message_t> && message);
    void handleRawListenerMessage(std::vector<zmq::message_t> && message);
    void handleMessage(std::vector<zmq::message_t> && message);

    MessageLoop messageLoop;

private:
    /// Zeromq context that we use
    std::shared_ptr<zmq::context_t> context;

    /// Listening socket for unformatted messages
    zmq::socket_t listener;

    /// Listening socket for formatted messages
    zmq::socket_t rawListener;

    std::map<std::string, size_t> stats;
    
    /// Socket we write things to.  There's one per thread which allows us
    /// to write from multiple threads without blocking.
    SocketPerThread logSocket;

#if 0
    /// Thread to do the logging
    boost::scoped_ptr<boost::thread> logThread;

    /// Thread to run the logging
    void runLogThread(ACE_Semaphore & sem);
#endif

    struct Output;
    struct Outputs;

    /// Current list of outputs.  Must be swapped atomically.
    Outputs * outputs;

    /// Thing we get subscription messages from
    std::vector<std::shared_ptr<zmq::socket_t> > subscriptions;

    /// Thing to run when we stop
    boost::function<void ()> onStop;

    /// Extra flag to make sure that we shutdown
    bool doShutdown;

    /// Number of messages that have been sent to be processed
    mutable uint64_t messagesSent;

    /// Number of messages that have actually been processed
    uint64_t messagesDone;

};


} // namespace Datacratic

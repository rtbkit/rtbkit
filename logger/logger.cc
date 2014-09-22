/* logger.cc
   Jeremy Barnes, 19 May 2011
   Copyright (c) 2011 Datacratic.  All rights reserved.

   Various classes for logging of the RTB data.
*/

#include "logger.h"
#include "jml/utils/vector_utils.h"
#include "jml/arch/atomic_ops.h"
#include "jml/arch/demangle.h"
#include "jml/utils/string_functions.h"
#include "jml/arch/timers.h"
#include "file_output.h"
#include "publish_output.h"
#include "callback_output.h"
#include <boost/make_shared.hpp>


using namespace std;
using namespace ML;


namespace Datacratic {


/*****************************************************************************/
/* LOG OUTPUT                                                                */
/*****************************************************************************/

Json::Value
LogOutput::
stats() const
{
    return Json::Value();
}

void
LogOutput::
clearStats()
{
}


/*****************************************************************************/
/* LOGGER                                                                    */
/*****************************************************************************/

Logger::
Logger(size_t bufferSize)
    : context(std::make_shared<zmq::context_t>(1)),
      messages(bufferSize),
      outputs(0),
      messagesSent(0), messagesDone(0)
{
    doShutdown = false;
}

Logger::
Logger(zmq::context_t & contextRef, size_t bufferSize)
    : context(ML::make_unowned_std_sp(contextRef)),
      messages(bufferSize),
      outputs(0),
      messagesSent(0), messagesDone(0)
{
    doShutdown = false;
}

Logger::
Logger(std::shared_ptr<zmq::context_t> & context, size_t bufferSize)
    : context(context),
      messages(bufferSize),
      outputs(0),
      messagesSent(0), messagesDone(0)
{
    doShutdown = false;
}

Logger::
~Logger()
{
    shutdown();
}

void
Logger::
init()
{
    messageLoop.init();

    messages.onEvent = [=](std::vector<std::string> && message) {
        handleListenerMessage(message);
    };

    messageLoop.addSource("Logger::messages", messages);
}

void
Logger::
subscribe(const std::string & uri,
          const std::vector<std::string> & channels,
          const std::string & identity)
{
#if 0
    if (logThread)
        throw ML::Exception("must subscribe before log thread starts");
#endif

    //using namespace std;
    //cerr << "subscribing to " << uri << " on " << channels.size()
    //     << " channels "
    //     << channels << endl;

    auto subscription = std::make_shared<zmq::socket_t>(*context, ZMQ_SUB);

    setHwm(*subscription, 100000);

    if (identity != "")
        setIdentity(*subscription, identity);
    subscription->connect(uri.c_str());

    if (channels.empty())
        subscribeChannel(*subscription, "");
    else {
        for (auto it = channels.begin(), end = channels.end(); it != end;
             ++it)
            subscribeChannel(*subscription, *it);
    }

    subscriptions.push_back(subscription);

    messageLoop.addSource("Logger::" + identity,
                          std::make_shared<ZmqBinaryEventSource>
                          (*subscription, [=] (std::vector<zmq::message_t> && message)
                           {
                               this->handleMessage(std::move(message));
                           }));
}

    /// Single entry to output to
struct Logger::Output {
    Output()
        : logProbability(1.0)
    {
    }
    
    Output(const boost::regex & allowChannels,
           const boost::regex & denyChannels,
           std::shared_ptr<LogOutput> output,
           double logProbability)
        : allowChannels(allowChannels), denyChannels(denyChannels),
          output(output), logProbability(logProbability)
    {
    }
    
    boost::regex allowChannels;  // channels to match
    boost::regex denyChannels;  // channels to filter out
    std::shared_ptr<LogOutput> output;  // thing to write to
    double logProbability;
};

/// List of entries to output to
struct Logger::Outputs : public std::vector<Output> {
    Outputs()
        : old(0)
    {
    }
    
    Outputs(Outputs * old,
            const Output & toAdd)
        : old(old)
    {
        if (old) {
            reserve(old->size() + 1);
            insert(begin(), old->begin(), old->end());
        }

        push_back(toAdd);
    }

    ~Outputs()
    {
        if (old) delete old;
    }
    
    void logMessage(const std::string & channel,
                    const std::string & message)
    {
        for (auto it = begin(); it != end();  ++it) {
            try {
                //cerr << "channel = " << channel << endl;
                //cerr << "output = " << ML::type_name(*it->output) << endl;
                //cerr << "it->allowChannels = " << it->allowChannels.str() << endl;
                //cerr << "it->denyChannels = " << it->denyChannels.str() << endl;
                if (it->allowChannels.empty()
                    || boost::regex_match(channel, it->allowChannels)) {
                    //cerr << "   allow" << endl;
                    if (it->denyChannels.empty()
                        || !boost::regex_match(channel, it->denyChannels)) {

                        if (it->logProbability == 1.0
                            || ((random() % 100000)
                                < (it->logProbability * 100000))) {
                            it->output->logMessage(channel, message);
                        }
                        //cerr << "  *** log" << endl;
                    }
                }
            } catch (const std::exception & exc) {
                cerr << "error: writing message to channel " << channel
                     << " with output " << ML::type_name(*it->output)
                     << ": " << exc.what() << "; message = "
                     << message << endl;
            }
        }
    }
    
    Outputs * old;   // to allow cleanup
};

bool startsWith(std::string & s,
                const std::string & prefix)
{
    if (s.find(prefix) == 0) {
        s.erase(0, prefix.size());
        return true;
    }
    return false;
}

void
Logger::
logTo(const std::string & uri,
      const boost::regex & allowChannels,
      const boost::regex & denyChannels,
      double logProbability)
{
    string rest = uri;
    if (startsWith(rest, "file://"))
        addOutput(ML::make_std_sp(new FileOutput(rest)),
                  allowChannels, denyChannels, logProbability);
    else if (startsWith(rest, "pub://")) {
        auto output = ML::make_std_sp(new PublishOutput(context));
        output->bind(rest);
        addOutput(output, allowChannels, denyChannels, logProbability);
    }
    else throw Exception("don't know how to interpret output " + uri);
}

void
Logger::
addOutput(std::shared_ptr<LogOutput> output,
          const boost::regex & allowChannels,
          const boost::regex & denyChannels,
          double logProbability)
{
    Outputs * current = outputs;

    for (;;) {
        auto_ptr<Outputs> newOutputs
            (new Outputs(current, Output(allowChannels, denyChannels, output,
                                         logProbability)));
        if (ML::cmp_xchg(outputs, current, newOutputs.get())) {
            newOutputs.release();
            break;
        }
    }
}

void
Logger::
addCallback(boost::function<void (std::string, std::string)> callback,
            const boost::regex & allowChannels,
            const boost::regex & denyChannels,
            double logProbability)
{
    addOutput(std::make_shared<CallbackOutput>(callback),
              allowChannels, denyChannels, logProbability);
}

void
Logger::
clearOutputs()
{
    auto_ptr<Outputs> newOutputs(new Outputs());

    Outputs * current = outputs;

    for (;;) {
        newOutputs->old = current;
        
        if (ML::cmp_xchg(outputs, current, newOutputs.get())) break;
    }

    newOutputs.release();
}

void
Logger::
start(std::function<void ()> onStop)
{
    messagesSent = messagesDone = 0;
    doShutdown = false;

    messageLoop.start(onStop);


#if 0
    ACE_Semaphore sem(0);
        
    // NOTE: we can pass by reference since the log thread never touches
    // sem until this function has exited
    logThread.reset(new boost::thread([&](){ this->runLogThread(sem); }));

    // Wait until we're ready
    sem.acquire();
#endif
}

void
Logger::
waitUntilFinished()
{
    while (messagesDone < messagesSent) {
        //cerr << "sent " << messagesSent << " done "
        //     << messagesDone << endl;
        ML::sleep(0.01);
    }

    //cerr << "finished: sent " << messagesSent << " done "
    //     << messagesDone << endl;
}

void
Logger::
shutdown()
{
    messageLoop.shutdown();

    doShutdown = true;

    delete outputs;  outputs = 0;

    doShutdown = false;
}

void
Logger::
replay(const std::string & filename, ssize_t maxEvents)
{
    if (!outputs) return;

    filter_istream stream(filename);

    for (ssize_t i = 0;  stream && (maxEvents == -1 || i < maxEvents);  ++i) {
        string line;
        getline(stream, line);
        atomic_add(messagesSent, 1);
        messages.push(std::vector<std::string> { line });
    }

    cerr << "replay: sent " << messagesSent << " done: "
         << messagesDone << endl;
}

void
Logger::
replayDirect(const std::string & filename, ssize_t maxEvents) const
{
#if 0
    if (logThread)
        throw ML::Exception("log thread already up for replayDirect");
#endif

    if (!outputs) return;

    filter_istream stream(filename);

    for (ssize_t i = 0;  stream && (maxEvents == -1 || i < maxEvents);  ++i) {
        string line;
        getline(stream, line);
        atomic_add(messagesSent, 1);

        Outputs * current = outputs;
        
        if (!current) continue;
        
        string channel, content;
        string::size_type pos = line.find('\t');

        if (pos != string::npos) {
            channel = string(line, 0, pos);
            content = string(line, pos + 1);
        }
        
        current->logMessage(channel, content);
    }

    cerr << "replay: sent " << messagesSent << " done: "
         << messagesDone << endl;
}

void
Logger::
handleListenerMessage(std::vector<std::string> const & message)
{
    Outputs * current = outputs;
        
    if (!current) return;

    if (current->empty()) {
        current = 0;  // TODO: delete it
    }
    else if (current->old) {
        delete current->old;
        current->old = 0;
    }

    if (message.size() == 1 && message[0] == "SHUTDOWN")
        return;

    atomic_add(messagesDone, 1);

    if (!current) return;

    string const & channel = message[0];

    string toLog;
    toLog.reserve(1024);

    for (unsigned i = 1;  i < message.size();  ++i) {
        string const & strMessage = message[i];
        if (strMessage.find_first_of("\n\t\0\r") != string::npos) {
            cerr << "warning: part " << i << " of message "
                 << channel << " has illegal char: '"
                 << strMessage << "'" << endl;
        }
        if (i > 1) toLog += '\t';
        toLog += strMessage;
    }

    current->logMessage(channel, toLog);
}

void
Logger::
handleRawListenerMessage(std::vector<std::string> const & message)
{
    Outputs * current = outputs;
        
    if (!current) return;

    if (current->empty()) {
        current = 0;  // TODO: delete it
    }
    else if (current->old) {
        delete current->old;
        current->old = 0;
    }
    
    atomic_add(messagesDone, 1);

    if (message.size() == 1) {
        cerr << "ignored message with excessive elements: "
             << message.size()
             << endl;
        return;
    }

    string const & rawMessage = message[0];

    if (!current) return;

    string channel, content;
    string::size_type pos = rawMessage.find('\t');

    if (pos != string::npos) {
        channel = string(rawMessage, 0, pos);
        content = string(rawMessage, pos + 1);
    }

    current->logMessage(channel, content);
}

void
Logger::
handleMessage(std::vector<zmq::message_t> && message)
{
    Outputs * current = outputs;
        
    if (!current) return;

    if (current->empty()) {
        current = 0;  // TODO: delete it
    }
    else if (current->old) {
        delete current->old;
        current->old = 0;
    }
    
    //cerr << "got subscription message " << message << endl;
                
    if (!current) return;

    if (message.size() != 2) {
        vector<string> strMessages;
        for (auto & it: message) {
            strMessages.push_back(it.toString());
        }

        cerr << "ignoring invalid subscription message "
             << strMessages << endl;
        return;
    }

    //cerr << "logging subscription message " << message << endl;

    current->logMessage(message[0].toString(), message[1].toString());
}

#if 0
void
Logger::
runLogThread(ACE_Semaphore & sem)
{
    using namespace std;

    zmq::socket_t sock(*context, ZMQ_PULL);
    sock.bind(ML::format("inproc://logger@%p", this).c_str());

    zmq::socket_t raw_sock(*context, ZMQ_PULL);
    raw_sock.bind(ML::format("inproc://logger@%p-RAW", this).c_str());

    //cerr << "done bind" << endl;

    sem.release();

    int nitems = subscribers.size() + 2;
    zmq_pollitem_t items [nitems];
    zmq_pollitem_t item0 = { sock, 0, ZMQ_POLLIN, 0 };
    zmq_pollitem_t item1 = { raw_sock, 0, ZMQ_POLLIN, 0 };
    items[0] = item0;
    items[1] = item1;
    for (unsigned i = 0;  i < subscribers.size();  ++i) {
        zmq_pollitem_t item = { *subscriptions[i], 0, ZMQ_POLLIN, 0 };
        items[i + 2] = item;
    }

    //bool shutdown = false;

    //struct timeval beforeSleep, afterSleep;
    //gettimeofday(&afterSleep, 0);

    //cerr << "starting logging thread" << endl;
    
    while (!doShutdown) {
        //gettimeofday(&beforeSleep, 0);

        //dutyCycleCurrent.nsProcessing += timeDiff(afterSleep, beforeSleep);
        
        int rc = zmq_poll(items, nitems, 500 /* milliseconds */);

        //cerr << "rc = " << rc << endl;
        
        //gettimeofday(&afterSleep, 0);

        //dutyCycleCurrent.nsSleeping += timeDiff(beforeSleep, afterSleep);
        //dutyCycleCurrent.nEvents += 1;

        if (rc == -1 && zmq_errno() != EINTR) {
            cerr << "zeromq log error: " << zmq_strerror(zmq_errno()) << endl;
        }

        Outputs * current = outputs;
        
        if (!current) continue;

        if (current->empty()) {
            current = 0;  // TODO: delete it
        }

        if (current->old) {
            delete current->old;
            current->old = 0;
        }
        
        if (items[0].revents & ZMQ_POLLIN) {
            vector<string> message = recvAll(sock);

            if (message.size() == 1 && message[0] == "SHUTDOWN")
                return;

            atomic_add(messagesDone, 1);

            if (!current) continue;

            string toLog;
            toLog.reserve(1024);

            for (unsigned i = 1;  i < message.size();  ++i) {
                if (message[i].find_first_of("\n\t\0\r") != string::npos) {
                    cerr << "warning: part " << i << " of message "
                         << message[0] << " has illegal char: '"
                         << message[i] << "'" << endl;
                }
                if (i > 1) toLog += '\t';
                toLog += message[i];
            }
            
            current->logMessage(message[0], toLog);
        }
        if (items[1].revents & ZMQ_POLLIN) {
            string message = recvMesg(raw_sock);

            atomic_add(messagesDone, 1);

            if (!current) continue;

            string channel, content;
            string::size_type pos = message.find('\t');

            if (pos != string::npos) {
                channel = string(message, 0, pos);
                content = string(message, pos + 1);
            }

            current->logMessage(channel, content);
        }
        for (unsigned i = 0;  i < subscriptions.size();  ++i) {
            if (items[i + 2].revents & ZMQ_POLLIN) {
                vector<string> message = recvAll(*subscriptions[i]);
                
                //cerr << "got subscription message " << message << endl;
                
                if (!current) continue;
                
                if (message.size() != 2) {
                    cerr << "ignoring invalid subscription message "
                         << message << endl;
                }
                
                //cerr << "logging subscription message " << message << endl;
                
                current->logMessage(message[0], message[1]);
            }
        }
    }
}
#endif

} // namespace Datacratic

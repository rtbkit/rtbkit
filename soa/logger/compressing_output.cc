/* compressing_output.cc
   Jeremy Barnes, 20 September 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Base for an output source that compresses its data.
*/

#include "compressing_output.h"
#include "jml/utils/parse_context.h"


using namespace std;
using namespace ML;


namespace Datacratic {


/*****************************************************************************/
/* WORKER THREAD OUTPUT                                                      */
/*****************************************************************************/

WorkerThreadOutput::
WorkerThreadOutput(size_t ringBufferSize)
    : ringBuffer(ringBufferSize)
{
}

WorkerThreadOutput::
~WorkerThreadOutput()
{
}

void
WorkerThreadOutput::
startWorkerThread()
{
    if (logThread)
        throw ML::Exception("log thread already up");

    shutdown_ = 0;
    up_ = 0;

    // NOTE: we can pass by reference since the log thread never touches
    // sem until this function has exited
    logThread.reset(new boost::thread([&](){ this->runLogThread(); }));

    // Wait until we're ready
    while (!up_)
        futex_wait(up_, false);
}

void
WorkerThreadOutput::
stopWorkerThread()
{
    if (logThread) {
        shutdown_ = 1;

        Message message;
        message.type = MT_SHUTDOWN;
        ringBuffer.push(message);

        logThread->join();
        logThread.reset();
    }
}

void
WorkerThreadOutput::
logMessage(const std::string & channel,
           const std::string & contents)
{
    Message message;
    message.type    = MT_LOG;
    message.channel = channel;
    message.contents = contents;

    ringBuffer.push(std::move(message));
}

#if 0
void
WorkerThreadOutput::
endRecord()
{
    Message message;
    message.type    = MT_END;

    ringBuffer.push(std::move(message));
}
#endif

void
WorkerThreadOutput::
pushOperation(const std::function<void ()> & op)
{
    Message message;
    message.type = MT_OP;
    message.op = op;

    ringBuffer.push(std::move(message));
}

Json::Value
WorkerThreadOutput::
stats() const
{
    Json::Value result;

    ML::Duty_Cycle_Timer::Stats dutyStats = duty.stats();

    result["duty"]["dutyCycle"] = dutyStats.duty_cycle();
    result["duty"]["usAwake"] = (int)dutyStats.usAwake;
    result["duty"]["usAsleep"] = (int)dutyStats.usAsleep;
    result["duty"]["wakeups"] = (int)dutyStats.numWakeups;

    return result;
}

void
WorkerThreadOutput::
clearStats()
{
    duty.clear();
}

void
WorkerThreadOutput::
runLogThread()
{
    using namespace std;

    duty.clear();

    up_ = 1;
    futex_wake(up_);

    while (!shutdown_) {

        duty.notifyBeforeSleep();
        Message msg;
        bool found = ringBuffer.tryPop(msg, 0.5);
        duty.notifyAfterSleep();

        if (!found)
            continue;

        switch (msg.type) {

        case MT_LOG:
            implementLogMessage(msg.channel, msg.contents);
            break;
            
        case MT_END:
            //implementEndRecord();
            break;

        case MT_OP:
            try {
                msg.op();
            } catch (const std::exception & exc) {
                cerr << "warning: log operation threw exception: "
                     << exc.what() << endl;
            }
            break;

        case MT_SHUTDOWN:
            break;
            
        default:
            throw ML::Exception("unknown file logger message type");
        }
    }
}


/*****************************************************************************/
/* COMPRESSING OUTPUT                                                        */
/*****************************************************************************/

/** LogOutput implementation that compresses data before passing it off
    somewhere else to be actually written.
*/

CompressingOutput::
CompressingOutput(size_t ringBufferSize,
                  Compressor::FlushLevel flushLevel)
    : WorkerThreadOutput(ringBufferSize),
      compressorFlushLevel(flushLevel)
{
}

CompressingOutput::
~CompressingOutput()
{
}

void
CompressingOutput::
open(std::shared_ptr<Sink> sink,
     const std::string & compression,
     int compressionLevel)
{
    if (compressor)
        throw ML::Exception("can't open compressor without closing the "
                            "previous one");
    compressor.reset(Compressor::create(compression, compressionLevel));

    this->sink = sink;

    onData = std::bind(&Sink::write,
                       sink,
                       std::placeholders::_1,
                       std::placeholders::_2);
}

void
CompressingOutput::
closeCompressor()
{
    if (!compressor)
        return;
    compressor->finish(onData);
    compressor.reset();
}

void
CompressingOutput::
implementLogMessage(const std::string & channel,
                    const std::string & message)
{
    if (!compressor)
        throw ML::Exception("implementLogMessage without compressor");

    if (onFileWrite) 
        onFileWrite(channel, channel.size() + message.size() + 2);

    char buf[channel.size() + message.size() + 2];
    memcpy(buf, channel.c_str(), channel.size());
    buf[channel.size()] = '\t';
    memcpy(buf + channel.size() + 1, message.c_str(), message.size());
    buf[channel.size() + message.size() + 1] = '\n';

    compressor->compress(buf, channel.size() + message.size() + 2,
                         onData);

    // This should be done elsewhere or accessed via a flag
    compressor->flush(compressorFlushLevel, onData);
}

} // namespace Datacratic

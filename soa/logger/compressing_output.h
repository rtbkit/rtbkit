/* compressing_output.h                                            -*- C++ -*-
   Jeremy Barnes, 20 September 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

*/

#ifndef __logger__compressing_output_h__
#define __logger__compressing_output_h__

#include "logger.h"
#include "compressor.h"
#include "jml/utils/ring_buffer.h"
#include "jml/arch/timers.h"


namespace Datacratic {


enum FileFlushLevel {
    FLUSH_NONE,       ///< No flushing
    FLUSH_TO_OS,      ///< Data makes it to the OS; survives process crash
    FLUSH_TO_DISK     ///< Data makes it to disk; survives disk crash
};


/*****************************************************************************/
/* WORKER THREAD OUTPUT                                                      */
/*****************************************************************************/

/** LogOutput implementation that causes work to be done in a separate
    thread.  Supports both writing of records and rotation in a
    strictly ordered manner.
*/

struct WorkerThreadOutput : public LogOutput {

    WorkerThreadOutput(size_t ringBufferSize = 65536);

    virtual ~WorkerThreadOutput();

    virtual void logMessage(const std::string & channel,
                            const std::string & message);

    virtual Json::Value stats() const;

    virtual void clearStats();

protected:
    void startWorkerThread();
    void stopWorkerThread();

    /** Push the given operation onto a queue of things to be done.  It will
        be done in the worker thread once its time is up.
    */
    void pushOperation(const std::function<void ()> & op);

    enum MessageType {
        MT_LOG,     ///< Log the given thing
        MT_END,     ///< End the record
        MT_OP,      ///< Run the given function in the thread
        MT_SHUTDOWN
    };

    struct Message {
        MessageType type;
        std::string channel;
        std::string contents;
        std::function<void ()> op;
    };

    ML::RingBufferSRMW<Message> ringBuffer;

    virtual void implementLogMessage(const std::string & channel,
                                     const std::string & message) = 0;

    /// Thread to do the logging
    boost::scoped_ptr<boost::thread> logThread;

    /// Thread to run the logging
    void runLogThread();

    /// Get information on duty cycle
    ML::Duty_Cycle_Timer duty;

    volatile int shutdown_;
    volatile int up_;
};


/*****************************************************************************/
/* COMPRESSING OUTPUT                                                        */
/*****************************************************************************/

/** LogOutput implementation that compresses data before passing it off
    somewhere else to be actually written.
*/

struct CompressingOutput : public WorkerThreadOutput {

    CompressingOutput(size_t ringBufferSize = 65536,
                      Compressor::FlushLevel compressorFlushLevel
                          = Compressor::FLUSH_AVAILABLE);

    virtual ~CompressingOutput();

    struct Sink {
        virtual ~Sink()
        {
        }

        /** Function that closes the given sink. */
        virtual void close() = 0;
        
        /** Function that should be overwritten to write data when it's
            output by the compressor.
        */
        virtual size_t write(const char * data, size_t size) = 0;

        /** Function that should be overwritten when the stream needs to be
            flushed to make sure that the data gets to disk.
        */
        virtual size_t flush(FileFlushLevel flushLevel) = 0;

        std::string currentUri;// the uri we are currently writing to
    };

    void open(std::shared_ptr<Sink> sink,
              const std::string & compression,
              int compressionLevel);

    void closeCompressor();

    boost::function<void (std::string, std::size_t)> onFileWrite;

protected:
    Compressor::FlushLevel compressorFlushLevel;
    std::shared_ptr<Sink> sink;
    std::shared_ptr<Compressor> compressor;
    std::function<size_t (const char *, size_t)> onData;

    // Overrides

    virtual void implementLogMessage(const std::string & channel,
                                     const std::string & message);
};


} // namespace Datacratic


#endif /* __logger__compressing_output_h__ */

/* sink.h                                                          -*- C++ -*-
   Wolfgang Sourdeau, September 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.

   A sink mechanism for writing to input or output "pipes".
 */

#pragma once

#include <functional>
#include <string>
#include <vector>

#include "async_event_source.h"
#include "async_writer_source.h"


namespace Datacratic {

/* OUTPUTSINK

   A sink interface that provides a medium-independent interface for sending
   data. Note that "OutputSink" is a pseudo vitual base class.

   The provider is responsible for making the target resource available
   and for closing it. It also handles thread safety and whether the writes
   are blocking or non-blocking. The provider provides the appropriate
   OutputSink for its operations.
 */

struct OutputSink {
    static constexpr int OPEN = 0;
    static constexpr int CLOSING = 1;
    static constexpr int CLOSED = 2;

    /* The client has requested the closing of the connection. */
    typedef std::function<void ()> OnClose;

    OutputSink(const OnClose & onClose = nullptr)
        : state(OPEN), onClose_(onClose)
    {}

    /* Write data to the output. Returns true when successful. */
    virtual bool write(std::string && data) = 0;

    bool write(const std::string & data)
    {
        std::string localData(data);
        return write(std::move(localData));
    }

    /* Request the output to be closed and guarantee that "write" will never
       be invoked anymore. May be invoked by both ends. */
    virtual void requestClose()
    { doClose(); }

    void waitState(int expectedState);

    int state;

    void doClose();

    OnClose onClose_;
};

std::istream & operator >> (std::istream & stream, OutputSink & sink);


/* CALLBACKOUTPUTSINK
 */

struct CallbackOutputSink : public OutputSink {
    typedef std::function<bool(std::string && data)> OnData;

    CallbackOutputSink(const OnData & onData,
                       const OutputSink::OnClose & onClose = nullptr)
        : OutputSink(onClose), onData_(onData)
    {}

    virtual bool write(std::string && data);

private:
    OnData onData_;

};

/* ASYNCOUTPUTSINK

   A non-blocking output sink that sends data to an open file descriptor.
   Opening and closing of the file descriptor is left to the provider. */
struct AsyncFdOutputSink : public AsyncWriterSource,
                           public OutputSink {
    /* The file-descriptor was hung up by the receiving end. */
    typedef std::function<void ()> OnHangup;

    AsyncFdOutputSink(const OnHangup & onHangup,
                      const OnClose & onClose,
                      int bufferSize = 32);
    ~AsyncFdOutputSink();

    void init(int outputFd);

    /* AsyncWriterSource */
    void onClosed(bool fromPeer, const std::vector<std::string> & msgs);

    /* OutputSink interface */
    virtual bool write(std::string && data);
    virtual void requestClose();

private:
    OnHangup onHangup_;
};


/* INPUTSINK

   A sink that provides a medium-independent interface for receiving data.

   The client is responsible for the resource management. The provider
   returns the appropriate InputSink for its operations.

   An InputSink may write to an OutputSink when piping data between 2 threads
   or file descriptors.
 */

struct InputSink {
    /* Notify that data has been received and transfers it. */
    virtual void notifyReceived(std::string && data) = 0;

    /* Notify that the input has been closed and that data will not be
       received anymore. */
    virtual void notifyClosed(void) = 0;
};


/* NULLINPUTSINK

   An InputSink that discards everything.
 */

struct NullInputSink : public InputSink {
    virtual void notifyReceived(std::string && data);
    virtual void notifyClosed();
};


/* CALLBACKINPUTSINK

   An InputSink invoking a callback upon data reception.
 */

struct CallbackInputSink : public InputSink {
    typedef std::function<void(std::string && data)> OnData;
    typedef std::function<void()> OnClose;

    CallbackInputSink(const OnData & onData,
                      const OnClose & onClose = nullptr)
        : onData_(onData), onClose_(onClose)
    {}

    virtual void notifyReceived(std::string && data);
    virtual void notifyClosed();

private:
    OnData onData_;
    OnClose onClose_;
};

/* OSTREAMINPUTSINK

   An InputSink issuing data to an ostream
 */

struct OStreamInputSink : public InputSink {
    OStreamInputSink(std::ostream * stream)
        : stream_(stream)
    {}

    virtual void notifyReceived(std::string && data);
    virtual void notifyClosed();

private:
    std::ostream * stream_;
};


/* CHAININPUTSINK

   An InputSink that chains callbacks to other input sinks
   (not thread-safe)
 */

struct ChainInputSink : public InputSink {
    void appendSink(const std::shared_ptr<InputSink> & newSink);

    virtual void notifyReceived(std::string && data);
    virtual void notifyClosed();

private:
    std::vector<std::shared_ptr<InputSink> > sinks_;
};

} // namespace Datacratic

#pragma once

#include <functional>
#include <string>

#include "async_event_source.h"
#include "typed_message_channel.h"


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
    typedef std::function<void (const OutputSink & closedSink)> OnClosed;

    OutputSink(const OnClosed & onClosed = nullptr)
        : closed_(false),
          onClosed_(onClosed)
    {}

    /* Write data to the output */
    virtual void write(std::string && data)
    { throw ML::Exception("unimplemented"); }

    /* Request the output to be closed and guarantee that "write" will never
       be invoked anymore. May be invoked by both ends. */
    virtual void requestClose()
    { throw ML::Exception("unimplemented"); }

    /* From the provider, corrolary to the "requestClose" method.
       Invoked "onClosed_" if set. */
    void doClose();

    bool closed() const;

private:
    bool closed_;
    OnClosed onClosed_;
};


/* ASYNCOUTPUTSINK

   A non-blocking output sink that sends data to an open file descriptor. */
struct AsyncFdOutputSink : public AsyncEventSource,
                           public OutputSink {
    AsyncFdOutputSink(const OnClosed & onClosed, int bufferSize = 32);
    ~AsyncFdOutputSink();

    void init(int outputFd);

    /* AsyncEventSource interface */
    virtual int selectFd() const
    { return epollFd_; }
    virtual bool processOne();

    /* OutputSink interface */
    virtual void write(std::string && data);
    virtual void requestClose();

private:
    typedef std::function<void (struct epoll_event &)> EpollCallback;

    void addFdOneShot(int fd, EpollCallback & cb, bool writerFd = false);
    void restartFdOneShot(int fd, EpollCallback & cb, bool writerFd = false);
    void removeFd(int fd);
    void close();

    void handleFdEvent(const struct epoll_event & event);
    void handleWakeupEvent(const struct epoll_event & event);
    EpollCallback handleFdEventCb_;
    EpollCallback handleWakeupEventCb_;

    void flushThreadBuffer();
    void flushStdInBuffer();
    void doClose();

    bool closing_;

    int epollFd_;

    int outputFd_;
    int fdReady_;

    ML::Wakeup_Fd wakeup_;
    ML::RingBufferSRMW<std::string> threadBuffer_;

    std::string buffer_;
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
    virtual void notifyReceived(std::string && data)
    { throw ML::Exception("unimplemented"); }

    /* Notify that the input has been closed and that data will not be
       received anymore. */
    virtual void notifyClosed(void)
    { throw ML::Exception("unimplemented"); }
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

    CallbackInputSink(const OnData & onData, const OnClose & onClose)
        : onData_(onData), onClose_(onClose)
    {}

    virtual void notifyReceived(std::string && data);
    virtual void notifyClosed();

private:
    OnData onData_;
    OnClose onClose_;
};

} // namespace Datacratic

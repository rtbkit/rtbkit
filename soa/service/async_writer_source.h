/* async_writer_source.h                                           -*- C++ -*-
   Wolfgang Sourdeau, April 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.

   A base class for handling writable file descriptors.
*/

#pragma once

#include <sys/epoll.h>

#include <atomic>
#include <map>
#include <queue>
#include <string>
#include <vector>
#include <functional>
#include <exception>

#include "jml/utils/ring_buffer.h"

#include "async_event_source.h"
#include "typed_message_channel.h"


namespace Datacratic {

/****************************************************************************/
/* ASYNC WRITE RESULT                                                       */
/****************************************************************************/

/* invoked when a write operation has been performed, where "written" is the
   string that was sent, "writtenSize" is the amount of bytes from it that was
   sent; the latter is always equal to the length of the string when error is
   0 */

struct AsyncWriteResult {
    AsyncWriteResult(int newError,
                     std::string && newWritten,
                     size_t newWrittenSize)
        : error(newError),
          written(std::move(newWritten)),
          writtenSize(newWrittenSize)
    {
    }

    int error;
    std::string written;
    size_t writtenSize;
};


/****************************************************************************/
/* ASYNC WRITER SOURCE                                                      */
/****************************************************************************/

/* A base class enabling the asynchronous and buffered writing of data to a
 * file descriptor. This class currently implements two separate concerns (a
 * read-write "Epoller" and a write queue) and might need to be split at some
 * point. */

struct AsyncWriterSource : public AsyncEventSource
{
    /* type of callback used when the file descriptor has been closed */
    typedef std::function<void(bool,
                               const std::vector<std::string> & msgs)> OnClosed;

    /* type of callback invoked when a string or a message has been written to
       the file descriptor */
    typedef std::function<void(AsyncWriteResult)> OnWriteResult;

    /* type of callback invoked when data has been read from the file
       descriptor */
    typedef std::function<void(const char *, size_t)> OnReceivedData;

    /* type of callback invoked whenever an uncaught exception occurs */
    typedef std::function<void(const std::exception_ptr &)> OnException;

    AsyncWriterSource(const OnClosed & onClosed,
                      const OnReceivedData & onReceivedData,
                      const OnException & onException,
                      /* size of the message queue */
                      size_t maxMessages,
                      /* size of the read/receive buffer */
                      size_t readBufferSize);
    virtual ~AsyncWriterSource();

    /* AsyncEventSource interface */
    virtual int selectFd() const
    { return epollFd_; }
    virtual bool processOne();

    /* enqueue "data" for writing, provided the file descriptor is open or
     * being opened, or throws */
    bool write(std::string data,
               const OnWriteResult & onWriteResult);
    bool write(const char * data, size_t size,
               const OnWriteResult & onWriteResult)
    {
        return write(std::string(data, size), onWriteResult);
    }

    /* returns whether we are ready to accept messages for sending */
    bool queueEnabled()
        const
    {
        return queueEnabled_;
    }

    /* close the file descriptor as soon as all bytes have been sent and
     * received, implying that "write" will never be invoked anymore */
    void requestClose();

    /* invoked when the connection is closed, where "fromPeer" indicates
     * whether the file descriptor was closed due to a call to "requestClose"
     * or due to a pipe reset. In the latter case, "msgs" also contains all
     * the unsent messages. */
    virtual void onClosed(bool fromPeer,
                                const std::vector<std::string> & msgs);

    /* invoked when the data is available for reading */
    virtual void onReceivedData(const char * data, size_t size);

    /* invoked when an exception occurs during the handling of events */
    virtual void onException(const std::exception_ptr & excPtr);

    /* number of bytes actually sent */
    uint64_t bytesSent() const
    { return bytesSent_; }

    uint64_t bytesReceived() const
    { return bytesReceived_; }

    /* number of messages actually sent */
    size_t msgsSent() const
    { return msgsSent_; }

protected:
    typedef std::function<void()> OnUnregistered;

    /* set the "main" file descriptor, for which epoll events are monitored
     * and the onWriteResult, onReceivedData and onClosed callbacks are
     * invoked automatically */
    void setFd(int fd);
    int getFd()
        const
    {
        return fd_;
    }

    /* enable message queueing */
    void enableQueue()
    {
        queueEnabled_ = true;
    }

    void disableQueue()
    {
        queueEnabled_ = false;
    }

    /* close the "main" file descriptor and take care of the surrounding
       operations */
    virtual void closeFd();

    /* type of callback invoked whenever an epoll event is reported for a
     * file descriptor */
    typedef std::function<void (const ::epoll_event &)> EpollCallback;

    /* register a file descriptor into the internal epoll queue for reading
       and/or writing */
    void addFd(int fd, bool readerFd, bool writerFd)
    { performAddFd(fd, readerFd, writerFd, false, false); }

    /* same as addFd, with the EPOLLONESHOT flag */
    void addFdOneShot(int fd, bool readerFd, bool writerFd)
    { performAddFd(fd, readerFd, writerFd, false, true); }

    /* modify a file descriptor in the epoll queue */
    void modifyFd(int fd, bool readerFd, bool writerFd)
    { performAddFd(fd, readerFd, writerFd, true, false); }

    /* same as modifyFd, with the EPOLLONESHOT flag */
    void modifyFdOneShot(int fd, bool readerFd, bool writerFd)
    { performAddFd(fd, readerFd, writerFd, true, true); }

    /* remove a file descriptor from the internal epoll queue */
    void removeFd(int fd);

    /* associate a callback with a file descriptor for future epoll
       operations */
    void registerFdCallback(int fd, const EpollCallback & cb);

    /* dissociate a callback and a file descriptor from the callback registry,
       with the "delayed" parameter indicating whether the operation must
       occur immediately or at the end of the epoll loop */
    void unregisterFdCallback(int fd, bool delayed,
                              const OnUnregistered & onUnregistered
                              = nullptr);

    std::vector<std::string> emptyMessageQueue();

private:
    /* Structure holding a write operation */
    struct AsyncWrite {
        AsyncWrite()
            : sent(0)
        {
        }

        AsyncWrite(std::string && newMessage,
                   const OnWriteResult & newOnWriteResult)
            : message(std::move(newMessage)), sent(0),
              onWriteResult(newOnWriteResult)
        {
        }

        void clear()
        {
            message.clear();
            sent = 0;
            onWriteResult = nullptr;
        }

        std::string message;
        size_t sent;
        OnWriteResult onWriteResult;
    };

    void performAddFd(int fd, bool readerFd, bool writerFd,
                      bool modify, bool oneshot);

    /* epoll operations */
    void closeEpollFd();

    /* fd operations */
    void flush();

    void handleFdEvent(const ::epoll_event & event);
    void handleReadReady();
    void handleWriteReady();
    void handleWriteResult(int error, AsyncWrite && currentWrite);
    void handleClosing(bool fromPeer, bool delayedUnregistration);
    void handleException();

    /* wakeup operations */
    void handleQueueNotification();

    int epollFd_;
    size_t numFds_;

    std::map<int, EpollCallback> fdCallbacks_;
    std::map<int, OnUnregistered> delayedUnregistrations_;

    int fd_;
    std::atomic<bool> closing_;
    size_t readBufferSize_;
    bool writeReady_;

    bool queueEnabled_;
    TypedMessageQueue<AsyncWrite> queue_;
    AsyncWrite currentWrite_;

    uint64_t bytesSent_;
    uint64_t bytesReceived_;
    size_t msgsSent_;

    OnClosed onClosed_;
    OnWriteResult onWriteResult_;
    OnReceivedData onReceivedData_;
    OnException onException_;
};

}

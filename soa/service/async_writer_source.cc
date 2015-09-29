/* async_writer_source.cc
   Wolfgang Sourdeau, April 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.

   A base class for handling writable file descriptors.
*/

#include <fcntl.h>
#include <sys/epoll.h>
#include <poll.h>
#include <unistd.h>

#include "jml/utils/exc_assert.h"
#include "jml/utils/file_functions.h"

#include "async_writer_source.h"

using namespace std;
using namespace Datacratic;


AsyncWriterSource::
AsyncWriterSource(const OnClosed & onClosed,
                  const OnReceivedData & onReceivedData,
                  const OnException & onException,
                  size_t maxMessages,
                  size_t readBufferSize)
    : AsyncEventSource(),
      epollFd_(-1),
      numFds_(0),
      fd_(-1),
      closing_(false),
      readBufferSize_(readBufferSize),
      writeReady_(false),
      queueEnabled_(false),
      queue_([&] { this->handleQueueNotification(); },
             maxMessages),
      bytesSent_(0),
      bytesReceived_(0),
      msgsSent_(0),
      onClosed_(onClosed),
      onReceivedData_(onReceivedData),
      onException_(onException)
{
    epollFd_ = ::epoll_create(666);
    if (epollFd_ == -1)
        throw ML::Exception(errno, "epoll_create");

    auto handleQueueEventCb = [&] (const ::epoll_event & event) {
        queue_.processOne();
    };
    registerFdCallback(queue_.selectFd(), handleQueueEventCb);
}

AsyncWriterSource::
~AsyncWriterSource()
{
    if (fd_ != -1) {
        closeFd();
    }
    closeEpollFd();
}

void
AsyncWriterSource::
setFd(int newFd)
{
    if (fd_ != -1) {
        throw ML::Exception("fd already set: %d", fd_);
    }
    if (!ML::is_file_flag_set(newFd, O_NONBLOCK)) {
        throw ML::Exception("file decriptor is blocking");
    }

    addFd(queue_.selectFd(), true, false);

    auto handleFdEventCb = [&] (const ::epoll_event & event) {
        this->handleFdEvent(event);
    };
    registerFdCallback(newFd, handleFdEventCb);
    addFd(newFd, readBufferSize_ > 0, true);
    fd_ = newFd;
    closing_ = false;
    enableQueue();
}

void
AsyncWriterSource::
closeFd()
{
    ExcCheck(queue_.size() == 0, "message queue not empty");
    ExcCheck(fd_ != -1, "already closed (fd)");

    handleClosing(false, false);
}

void
AsyncWriterSource::
closeEpollFd()
{
    if (epollFd_ != -1) {
        ::close(epollFd_);
        epollFd_ = -1;
    }
}

bool
AsyncWriterSource::
write(string data, const OnWriteResult & onWriteResult)
{
    ExcAssert(!closing_);
    ExcAssert(queueEnabled_);

    bool result(true);

    if (queueEnabled()) {
        ExcCheck(data.size() > 0, "attempting to write empty data");
        result = queue_.push_back(AsyncWrite(move(data), onWriteResult));
    }
    else {
        throw ML::Exception("cannot write while queue is disabled");
    }

    return result;
}

void
AsyncWriterSource::
handleReadReady()
{
    char buffer[readBufferSize_];

    errno = 0;
    while (1) {
        ssize_t s = ::read(fd_, buffer, readBufferSize_);
        if (s > 0) {
            bytesReceived_ += s;
            onReceivedData(buffer, s);
        }
        else if (s == 0) {
            /* according to read(2), a value of 0 indicates eof. */
            if (closing_) {
                /* When closing_ is set, indicating a manual closing request,
                   we simply ignore that situation and break the loop, as the
                   closing request will likely occur during this round or the
                   next one. */
                break;
            }
            else {
                handleClosing(true, true);
            }
        }
        else {
            if (errno == EWOULDBLOCK) {
                break;
            }
            else if (errno == EBADF || errno == EINVAL) {
                /* This happens when the pipe or socket was closed by the
                   remote process before "read" was called (race
                   condition). */
                break;
            }
            if (s == -1) {
                throw ML::Exception(errno, "read");
            }
            else {
                break;
            }
        }
    }
}

void
AsyncWriterSource::
handleWriteReady()
{
    writeReady_ = true;
    flush();
}

void
AsyncWriterSource::
handleWriteResult(int error, AsyncWrite && currentWrite)
{
    if (currentWrite.onWriteResult) {
        currentWrite.onWriteResult(
            AsyncWriteResult(error,
                             move(currentWrite.message),
                             currentWrite.sent)
        );
    }
    currentWrite.clear();
}

void
AsyncWriterSource::
handleException()
{
    onException(current_exception());
}

void
AsyncWriterSource::
onClosed(bool fromPeer, const vector<string> & msgs)
{
    if (onClosed_) {
        onClosed_(fromPeer, msgs);
    }
}

void
AsyncWriterSource::
onReceivedData(const char * buffer, size_t bufferSize)
{
    if (onReceivedData_) {
        onReceivedData_(buffer, bufferSize);
    }
}

void
AsyncWriterSource::
onException(const exception_ptr & excPtr)
{
    if (onException_) {
        onException(excPtr);
    }
    else {
        rethrow_exception(excPtr);
    }
}

void
AsyncWriterSource::
requestClose()
{
    if (queueEnabled()) {
        disableQueue();
        closing_ = true;
        queue_.push_back(AsyncWrite("", nullptr));
    }
    else {
        throw ML::Exception("already closed/ing\n");
    }
}

/* async event source */
bool
AsyncWriterSource::
processOne()
{
    struct epoll_event events[numFds_];

    if (numFds_ > 0) {
        try {
            int res = epoll_wait(epollFd_, events, numFds_, 0);
            if (res == -1) {
                throw ML::Exception(errno, "epoll_wait");
            }

            for (int i = 0; i < res; i++) {
                auto * fn = static_cast<EpollCallback *>(events[i].data.ptr);
                (*fn)(events[i]);
            }

            for (auto & unreg: delayedUnregistrations_) {
                auto cb = move(unreg.second);
                unregisterFdCallback(unreg.first, false, cb);
            }
        }
        catch (const std::exception & exc) {
            handleException();
        }
    }

    return false;
}

/* wakeup events */

void
AsyncWriterSource::
handleQueueNotification()
{
    if (fd_ != -1) {
        flush();
        if (fd_ != -1 && !writeReady_) {
            modifyFd(fd_, readBufferSize_ > 0, true);
        }
    }
}

void
AsyncWriterSource::
flush()
{
    ExcAssert(fd_ != -1);
    if (!writeReady_) {
        return;
    }

    auto popWrite = [&] () {
        if (queue_.size() == 0) {
            return false;
        }
        auto writes = queue_.pop_front(1);
        ExcAssert(writes.size() > 0);
        currentWrite_ = move(writes[0]);
        return true;
    };

    if (currentWrite_.message.empty()) {
        if (!popWrite()) {
            return;
        }
        if (currentWrite_.message.empty()) {
            ExcAssert(closing_);
            handleClosing(false, true);
            return;
        }
    }

    ssize_t remaining(currentWrite_.message.size() - currentWrite_.sent);

    errno = 0;

    while (true) {
        const char * data = currentWrite_.message.c_str() + currentWrite_.sent;
        ssize_t len = ::write(fd_, data, remaining);
        if (len > 0) {
            currentWrite_.sent += len;
            remaining -= len;
            bytesSent_ += len;
            if (remaining == 0) {
                msgsSent_++;
                handleWriteResult(0, move(currentWrite_));
                if (!popWrite()) {
                    break;
                }
                if (currentWrite_.message.empty()) {
                    ExcAssert(closing_);
                    handleClosing(false, true);
                    break;
                }
                remaining = currentWrite_.message.size();
            }
        }
        else if (len < 0) {
            writeReady_ = false;
            if (errno == EWOULDBLOCK || errno == EAGAIN) {
                break;
            }
            handleWriteResult(errno, move(currentWrite_));
            if (errno == EPIPE || errno == EBADF) {
                handleClosing(true, true);
                break;
            }
            else {
                /* This exception indicates a lack of code in the handling of
                   errno. In a perfect world, it should never ever be
                   thrown. */
                throw ML::Exception(errno, "unhandled write error");
            }
        }
    }
}

/* fd events */

void
AsyncWriterSource::
handleFdEvent(const ::epoll_event & event)
{
    /* fd_ may be -1 here if closed from the queue handler in the same epoll
       loop. We thus need to return to ensure that no operations occur on a
       closed fd. */
    if (fd_ == -1) {
        return;
    }

    if ((event.events & EPOLLOUT) != 0) {
        handleWriteReady();
    }
    if (fd_ != -1 && (event.events & EPOLLIN) != 0) {
        handleReadReady();
    }
    if (fd_ != -1 && (event.events & EPOLLHUP) != 0) {
        handleClosing(true, true);
    }

    if (fd_ != -1) {
        modifyFd(fd_, readBufferSize_ > 0, !writeReady_);
    }
}

void
AsyncWriterSource::
handleClosing(bool fromPeer, bool delayedUnregistration)
{
    if (fd_ != -1) {
        disableQueue();
        removeFd(queue_.selectFd());
        unregisterFdCallback(fd_, delayedUnregistration);
        removeFd(fd_);
        ::close(fd_);
        fd_ = -1;
        writeReady_ = false;

        vector<string> lostMessages = emptyMessageQueue();
        onClosed(fromPeer, lostMessages);
    }
}

/* epoll operations */

void
AsyncWriterSource::
registerFdCallback(int fd, const EpollCallback & cb)
{
    if (delayedUnregistrations_.count(fd) == 0) {
        if (fdCallbacks_.find(fd) != fdCallbacks_.end()) {
            throw ML::Exception("callback already registered for fd");
        }
    }
    else {
        delayedUnregistrations_.erase(fd);
    }
    fdCallbacks_[fd] = cb;
}

void
AsyncWriterSource::
unregisterFdCallback(int fd, bool delayed,
                     const OnUnregistered & onUnregistered)
{
    if (fdCallbacks_.find(fd) == fdCallbacks_.end()) {
        throw ML::Exception("callback not registered for fd");
    }
    if (delayed) {
        ExcAssert(delayedUnregistrations_.count(fd) == 0);
        delayedUnregistrations_[fd] = onUnregistered;
    }
    else {
        delayedUnregistrations_.erase(fd);
        fdCallbacks_.erase(fd);
        if (onUnregistered) {
            onUnregistered();
        }
    }
}

void
AsyncWriterSource::
performAddFd(int fd, bool readerFd, bool writerFd, bool modify, bool oneshot)
{
    if (epollFd_ == -1)
        return;

    struct epoll_event event;
    if (oneshot) {
        event.events = EPOLLONESHOT;
    }
    else {
        event.events = 0;
    }
    if (readerFd) {
        event.events |= EPOLLIN;
    }
    if (writerFd) {
        event.events |= EPOLLOUT;
    }

    EpollCallback & cb = fdCallbacks_.at(fd);
    event.data.ptr = &cb;

    int operation = modify ? EPOLL_CTL_MOD : EPOLL_CTL_ADD;
    int res = epoll_ctl(epollFd_, operation, fd, &event);
    if (res == -1) {
        string message = (string("epoll_ctl:")
                          + " modify=" + to_string(modify)
                          + " fd=" + to_string(fd)
                          + " readerFd=" + to_string(readerFd)
                          + " writerFd=" + to_string(writerFd));
        throw ML::Exception(errno, message);
    }
    if (!modify) {
        numFds_++;
    }
}

void
AsyncWriterSource::
removeFd(int fd)
{
    if (epollFd_ == -1)
        return;

    int res = epoll_ctl(epollFd_, EPOLL_CTL_DEL, fd, 0);
    if (res == -1)
        throw ML::Exception(errno, "epoll_ctl DEL " + to_string(fd));
    if (numFds_ == 0) {
        throw ML::Exception("inconsistent number of fds registered");
    }
    numFds_--;
}

std::vector<std::string>
AsyncWriterSource::
emptyMessageQueue()
{
    std::vector<std::string> messages;

    auto writes = queue_.pop_front(0);
    for (auto & write: writes) {
        messages.emplace_back(move(write.message));
    }

    return messages;
}

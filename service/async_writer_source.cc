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
                  const OnWriteResult & onWriteResult,
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
      currentSent_(0),
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
    ExcCheck(fd_ == -1, "fd already set");
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
    enableQueue();
}

void
AsyncWriterSource::
closeFd()
{
    ExcCheck(queue_.size() == 0, "message queue not empty");
    ExcCheck(fd_ != -1, "already closed (fd)");

    handleClosing(false);
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
write(string data)
{
    ExcAssert(!closing_);
    ExcAssert(queueEnabled_);

    bool result(true);

    if (queueEnabled()) {
        ExcCheck(data.size() > 0, "attempting to write empty data");
        result = queue_.push_back(move(data));
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
handleWriteResult(int error,
                  const string & written, size_t writtenSize)
{
    onWriteResult(error, written, writtenSize);
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
onWriteResult(int error,
              const string & written, size_t writtenSize)
{
    if (onWriteResult_) {
        onWriteResult_(error, written, writtenSize);
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
        queue_.push_back("");
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
        }
        catch (...) {
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

    auto popLine = [&] () {
        if (queue_.size() == 0) {
            return false;
        }
        auto lines = queue_.pop_front(1);
        ExcAssert(lines.size() > 0);
        currentLine_ = move(lines[0]);
        currentSent_ = 0;
        return true;
    };

    if (currentLine_.size() == 0) {
        if (!popLine()) {
            return;
        }
        if (currentLine_.empty()) {
            ExcAssert(closing_);
            closeFd();
            return;
        }
    }

    ssize_t remaining(currentLine_.size() - currentSent_);

    errno = 0;

    while (true) {
        const char * data = currentLine_.c_str() + currentSent_;
        ssize_t len = ::write(fd_, data, remaining);
        if (len > 0) {
            currentSent_ += len;
            remaining -= len;
            bytesSent_ += len;
            if (remaining == 0) {
                msgsSent_++;
                handleWriteResult(0, currentLine_, currentLine_.size());
                if (!popLine()) {
                    currentLine_.clear();
                    break;
                }
                if (currentLine_.empty()) {
                    ExcAssert(closing_);
                    closeFd();
                    break;
                }
                remaining = currentLine_.size();
            }
        }
        else if (len < 0) {
            writeReady_ = false;
            if (errno == EWOULDBLOCK || errno == EAGAIN) {
                break;
            }
            handleWriteResult(errno, currentLine_, currentSent_);
            currentLine_.clear();
            if (errno == EPIPE || errno == EBADF) {
                handleClosing(true);
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
    if ((event.events & EPOLLOUT) != 0) {
        handleWriteReady();
    }
    if ((event.events & EPOLLIN) != 0) {
        handleReadReady();
    }
    if ((event.events & EPOLLHUP) != 0) {
        handleClosing(true);
    }

    if (fd_ != -1) {
        modifyFd(fd_, readBufferSize_ > 0, !writeReady_);
    }
}

void
AsyncWriterSource::
handleClosing(bool fromPeer)
{
    if (fd_ != -1) {
        disableQueue();
        removeFd(queue_.selectFd());
        removeFd(fd_);
        ::close(fd_);
        fd_ = -1;
        writeReady_ = false;

        vector<string> lostMessages = queue_.pop_front(0);
        onClosed(fromPeer, lostMessages);
    }
}

/* epoll operations */

void
AsyncWriterSource::
registerFdCallback(int fd, const EpollCallback & cb)
{
    if (fdCallbacks_.find(fd) != fdCallbacks_.end()) {
        throw ML::Exception("callback already registered for fd");
    }

    fdCallbacks_.insert({fd, cb});
}

void
AsyncWriterSource::
unregisterFdCallback(int fd)
{
    if (fdCallbacks_.find(fd) == fdCallbacks_.end()) {
        throw ML::Exception("callback not registered for fd");
    }
    fdCallbacks_.erase(fd);
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

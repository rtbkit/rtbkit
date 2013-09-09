#include <sys/epoll.h>
#include <poll.h>
#include "jml/arch/exception.h"

#include "sink.h"


using namespace std;
using namespace Datacratic;


/* OUTPUTSINK */

void
OutputSink::
doClose()
{
    state = CLOSING;
    if (onClose_) {
        onClose_();
    }
    state = CLOSED;
}

std::istream &
Datacratic::operator >>
(std::istream & stream, OutputSink & sink)
{
    string newData;

    stream >> newData;
    sink.write(move(newData));

    return stream;
}


/* CALLBACKOUTPUTSINK */

bool
CallbackOutputSink::
write(std::string && data)
{
    return onData_(move(data));
}


/* ASYNCOUTPUTSINK */

AsyncFdOutputSink::
AsyncFdOutputSink(const OnHangup & onHangup,
                  const OnClose & onClose,
                  int bufferSize)
    : AsyncEventSource(),
      OutputSink(onClose),
      onHangup_(onHangup),
      outputFd_(-1),
      fdReady_(false),
      wakeup_(EFD_NONBLOCK | EFD_CLOEXEC),
      threadBuffer_(bufferSize)
{
    epollFd_ = ::epoll_create(2);
    if (epollFd_ == -1)
        throw ML::Exception(errno, "epoll_create");

    handleWakeupEventCb_ = [&] (const struct epoll_event & event) {
        this->handleWakeupEvent(event);
    };
    addFdOneShot(wakeup_.fd(), handleWakeupEventCb_);

    buffer_.reserve(8192);
}

AsyncFdOutputSink::
~AsyncFdOutputSink()
{
    close();
}

void
AsyncFdOutputSink::
init(int outputFd)
{
    outputFd_ = outputFd;
    handleFdEventCb_ = [&] (const struct epoll_event & event) {
        this->handleFdEvent(event);
    };
    addFdOneShot(outputFd_, handleFdEventCb_, true);
}

bool
AsyncFdOutputSink::
write(std::string && data)
{
    bool result(true);

    if (state == OPEN) {
        if (threadBuffer_.tryPush(data)) {
            wakeup_.signal();
        }
        else {
            result = false;
        }
    }
    else {
        throw ML::Exception("cannot write after closing");
    }

    return result;
}

void
AsyncFdOutputSink::
requestClose()
{
    if (state == OPEN) {
        state = CLOSING;
        wakeup_.signal();
    }
    else {
        throw ML::Exception("cannot close twice");
    }
}

bool
AsyncFdOutputSink::
processOne()
{
    struct epoll_event events[3];

    int res = epoll_wait(epollFd_, events, sizeof(events), 0);
    if (res == -1) {
        throw ML::Exception(errno, "epoll_wait");
    }

    for (int i = 0; i < res; i++) {
        auto * fn = static_cast<EpollCallback *>(events[i].data.ptr);
        (*fn)(events[i]);
    }

    return false;
}

void
AsyncFdOutputSink::
addFdOneShot(int fd, EpollCallback & cb, bool writerFd)
{
    //cerr << Date::now().print(4) << "added " << fd << " one-shot" << endl;

    struct epoll_event event;
    event.events = (writerFd ? EPOLLOUT : EPOLLIN) | EPOLLONESHOT;
    event.data.ptr = &cb;
    
    int res = epoll_ctl(epollFd_, EPOLL_CTL_ADD, fd, &event);
        
    if (res == -1)
        throw ML::Exception(errno, "epoll_ctl ADD " + to_string(fd));
}

void
AsyncFdOutputSink::
restartFdOneShot(int fd, EpollCallback & cb, bool writerFd)
{
    //cerr << Date::now().print(4) << "restarted " << fd << " one-shot" << endl;

    struct epoll_event event;
    event.events = (writerFd ? EPOLLOUT : EPOLLIN) | EPOLLONESHOT;
    event.data.ptr = &cb;
    
    int res = epoll_ctl(epollFd_, EPOLL_CTL_MOD, fd, &event);
    
    if (res == -1)
        throw ML::Exception(errno, "epoll_ctl MOD " + to_string(fd));
}

void
AsyncFdOutputSink::
removeFd(int fd)
{
    //cerr << Date::now().print(4) << "removed " << fd << endl;

    int res = epoll_ctl(epollFd_, EPOLL_CTL_DEL, fd, 0);
    if (res == -1)
        throw ML::Exception(errno, "removeFd epoll_ctrl DEL " + to_string(fd));
}

void
AsyncFdOutputSink::
close()
{
    if (epollFd_ > -1) {
        ::close(epollFd_);
        epollFd_ = -1;
    }
}

void
AsyncFdOutputSink::
handleFdEvent(const struct epoll_event & event)
{
    if ((event.events & EPOLLHUP) != 0) {
        removeFd(outputFd_);
        onHangup_();
        outputFd_ = -1;
        state = CLOSED;
    }
    else if ((event.events & EPOLLOUT) != 0) {
        if (state != CLOSED) {
            fdReady_ = true;
            flushStdInBuffer();
            restartFdOneShot(outputFd_, handleFdEventCb_, true);
        }
    }
}

void
AsyncFdOutputSink::
handleWakeupEvent(const struct epoll_event & event)
{
    if ((event.events & EPOLLIN) != 0) {
        wakeup_.read();
        flushThreadBuffer();
        flushStdInBuffer();
    }
    else {
        throw ML::Exception("unhandled event");
    }

    if (state == OPEN) {
        restartFdOneShot(wakeup_.fd(), handleWakeupEventCb_);
    }
    else if (state == CLOSING) {
        doClose();
    }
}

void
AsyncFdOutputSink::
flushThreadBuffer()
{
    string newData;

    while (threadBuffer_.tryPop(newData)) {
        buffer_ += newData;
    }
}

void
AsyncFdOutputSink::
flushStdInBuffer()
{
    ssize_t remaining = buffer_.size();
    if (fdReady_ && remaining > 0) {
        const char * data = buffer_.c_str();
        size_t written(0);
        while (remaining > 0) {
            ssize_t len = ::write(outputFd_, data + written, remaining);
            if (len == 0) {
                buffer_ = buffer_.substr(written);
                break;
            }
            else if (len < 0) {
                if (errno == EWOULDBLOCK) {
                    fdReady_ = false;
                }
                else {
                    throw ML::Exception(errno, "write");
                }
            }
            else if (len > 0) {
                written += len;
                remaining -= len;
                if (remaining == 0) {
                    buffer_ = "";
                }
            }
        }
        buffer_.reserve(8192);
    }
}

void
AsyncFdOutputSink::
doClose()
{
    state = CLOSING;
    if (outputFd_ != -1) {
        removeFd(outputFd_);
        outputFd_ = -1;
    }

    removeFd(wakeup_.fd());
    ::close(wakeup_.fd());

    OutputSink::doClose();
}

/* NULLINPUTSINK */

void
NullInputSink::
notifyReceived(std::string && data)
{}

void
NullInputSink::
notifyClosed()
{}


/* CALLBACKINPUTSINK */

void
CallbackInputSink::
notifyReceived(std::string && data)
{
    onData_(move(data));
}

void
CallbackInputSink::
notifyClosed()
{
    if (onClose_) {
        onClose_();
    }
}


/* OSTREAMINPUTSINK

   An InputSink issuing data to an ostream
 */

void
OStreamInputSink::
notifyReceived(std::string && data)
{
    string newData(data);

    *stream_ << data;
}

void
OStreamInputSink::
notifyClosed()
{
}

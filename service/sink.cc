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
    closed_ = true;
    if (onClosed_) {
        onClosed_(*this);
    }
}

bool
OutputSink::
closed()
    const
{
    return closed_;
}


/* ASYNCOUTPUTSINK */

AsyncFdOutputSink::
AsyncFdOutputSink(const OnClosed & onClosed, int bufferSize)
    : AsyncEventSource(),
      OutputSink(onClosed),
      closing_(false),
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

void
AsyncFdOutputSink::
write(std::string && data)
{
    if (closing_) {
        throw ML::Exception("cannot write after closing");
    }

    if (threadBuffer_.tryPush(data))
        wakeup_.signal();
    else
        throw ML::Exception("the message queue is full");
}

void
AsyncFdOutputSink::
requestClose()
{
    if (closing_)
        return;
    closing_ = true;
    wakeup_.signal();
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
    }
}

void
AsyncFdOutputSink::
handleFdEvent(const struct epoll_event & event)
{
    if ((event.events & EPOLLOUT) != 0) {
        fdReady_ = true;
        flushStdInBuffer();
    }

    if ((event.events & EPOLLHUP) == 0) {
        restartFdOneShot(outputFd_, handleFdEventCb_, true);
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

    if (closing_ || (event.events & EPOLLHUP) != 0) {
        doClose();
    }
    else {
        restartFdOneShot(wakeup_.fd(), handleWakeupEventCb_);
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
            if (len > 0) {
                written += len;
                remaining -= len;
                if (remaining == 0) {
                    buffer_ = "";
                }
            }
            else {
                if (errno == EWOULDBLOCK) {
                    fdReady_ = false;
                    buffer_ = buffer_.substr(written);
                    break;
                }
                else {
                    throw ML::Exception(errno, "write");
                }
            }
        }
    }
    buffer_.reserve(8192);
}

void
AsyncFdOutputSink::
doClose()
{
    removeFd(wakeup_.fd());
    ::close(wakeup_.fd());

    if (outputFd_ != -1) {
        removeFd(outputFd_);
        ::close(outputFd_);
    }

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
    onClose_();
}

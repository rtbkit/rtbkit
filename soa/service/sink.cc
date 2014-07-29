/* sink.cc                                                         -*- C++ -*-
   Wolfgang Sourdeau, September 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.

   A sink mechanism for writing to input or output "pipes".
 */

#include <fcntl.h>
#include <sys/epoll.h>
#include <poll.h>

#include <iostream>

#include "jml/arch/atomic_ops.h"
#include "jml/arch/exception.h"
#include "jml/utils/file_functions.h"

#include "sink.h"


using namespace std;
using namespace Datacratic;


/* OUTPUTSINK */

void
OutputSink::
doClose()
{
    if (state == CLOSED) {
        return;
    }
    state = CLOSING;
    ML::futex_wake(state);
    if (onClose_) {
        onClose_();
    }
    state = CLOSED;
    ML::futex_wake(state);
}

void
OutputSink::
waitState(int expectedState)
{
    while (state != expectedState) {
        int currentState = state;
        ML::futex_wait(state, currentState);
    }
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
      threadBuffer_(bufferSize),
      bytesSent_(0),
      remainingMsgs_(0)
{
    epollFd_ = ::epoll_create(2);
    if (epollFd_ == -1)
        throw ML::Exception(errno, "epoll_create");

    handleWakeupEventCb_ = [&] (const struct epoll_event & event) {
        this->handleWakeupEvent(event);
    };
    addFdOneShot(wakeup_.fd(), handleWakeupEventCb_);
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
    if (!ML::is_file_flag_set(outputFd, O_NONBLOCK)) {
        throw ML::Exception("file decriptor is blocking");
    }

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
            ML::atomic_inc(remainingMsgs_);
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
        ML::futex_wake(state);
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
    if (epollFd_ == -1)
        return;
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
    if (epollFd_ == -1)
        return;
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
    if (epollFd_ == -1)
        return;
    //cerr << Date::now().print(4) << "removed " << fd << endl;

    int res = epoll_ctl(epollFd_, EPOLL_CTL_DEL, fd, 0);
    if (res == -1)
        throw ML::Exception(errno, "removeFd epoll_ctrl DEL " + to_string(fd));
}

void
AsyncFdOutputSink::
close()
{
    if (epollFd_ == -1) 
        return;

    ::close(epollFd_);
    epollFd_ = -1;
}

void
AsyncFdOutputSink::
handleFdEvent(const struct epoll_event & event)
{
    if ((event.events & EPOLLOUT) != 0) {
        if (state != CLOSED) {
            fdReady_ = true;
            flush();
        }
    }
    if ((event.events & EPOLLHUP) != 0) {
        removeFd(outputFd_);
        onHangup_();
        outputFd_ = -1;
        state = CLOSED;
        ML::futex_wake(state);
    }
    else {
        // TODO: this may cause intense looping since we dont currently wait
        // for EWOULDBLOCK to be returned before rearming the epoll fd.
        if (state != CLOSED) {
            restartFdOneShot(outputFd_, handleFdEventCb_, true);
        }
    }
}

void
AsyncFdOutputSink::
handleWakeupEvent(const struct epoll_event & event)
{
    if ((event.events & EPOLLIN) != 0) {
        if (fdReady_) {
            eventfd_t val;
            while (wakeup_.tryRead(val)) {
                flush();
            }
        }
    }
    else {
        throw ML::Exception("unhandled event");
    }

    if (state == OPEN) {
        restartFdOneShot(wakeup_.fd(), handleWakeupEventCb_);
    }
    else if (state == CLOSING) {
        if (remainingMsgs_ > 0 || lastLine_.size() > 0) {
            restartFdOneShot(wakeup_.fd(), handleWakeupEventCb_);
            wakeup_.signal();
        }
        else {
            doClose();
        }
    }
}

void
AsyncFdOutputSink::
flush()
{
    if (!fdReady_)
        return;

    // cerr << "flush1\n";
    if (lastLine_.size() == 0) {
        if (threadBuffer_.tryPop(lastLine_)) {
            ML::atomic_dec(remainingMsgs_);
        }
        else {
            return;
        }
    }

    bool done(false);
    size_t written(0), remaining(lastLine_.size());
    const char * data = lastLine_.c_str();

    while (fdReady_ && !done) {
        ssize_t len = ::write(outputFd_, data + written, remaining);
        if (len > 0) {
            written += len;
            remaining -= len;
            bytesSent_ += len;
            if (remaining == 0) {
                if (threadBuffer_.tryPop(lastLine_)) {
                    ML::atomic_dec(remainingMsgs_);
                    data = lastLine_.c_str();
                    remaining = lastLine_.size();
                    written = 0;
                }
                else {
                    done = true;
                }
            }
        }
        else if (len == 0) {
            done = true;
        }
        else {
            fdReady_ = false;
            if (errno == EPIPE) {
                handleHangup();
            }
            else if (errno != EWOULDBLOCK) {
                throw ML::Exception(errno, "write");
            }
        }
    }

    if (fdReady_ && written == 0) {
        cerr << "written nothing\n";
    }
    if (written > 0) {
        if (remaining == 0) {
            lastLine_ = "";
        }
        else {
            lastLine_ = lastLine_.substr(written);
        }
    }
}

void
AsyncFdOutputSink::
doClose()
{
    if (state == CLOSED) {
        cerr << "already closed\n";
    }
    state = CLOSING;
    ML::futex_wake(state);
    if (outputFd_ != -1) {
        try {
            removeFd(outputFd_);
        }
        catch(const ML::Exception & exc)
        {}
        outputFd_ = -1;
    }

    removeFd(wakeup_.fd());
    ::close(wakeup_.fd());

    OutputSink::doClose();
}

void
AsyncFdOutputSink::
handleHangup()
{
    removeFd(outputFd_);
    onHangup_();
    outputFd_ = -1;
    state = CLOSED;
    ML::futex_wake(state);
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


/* CHAININPUTSINK */

void
ChainInputSink::
appendSink(const std::shared_ptr<InputSink> & newSink)
{
    sinks_.emplace_back(newSink);
}

void
ChainInputSink::
notifyReceived(std::string && data)
{
    for (std::shared_ptr<InputSink> sink: sinks_) {
        string sinkData(data);
        sink->notifyReceived(move(sinkData));
    }
}

void
ChainInputSink::
notifyClosed()
{
    for (std::shared_ptr<InputSink> sink: sinks_) {
        sink->notifyClosed();
    }
}

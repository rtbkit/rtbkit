/* sink.cc                                                         -*- C++ -*-
   Wolfgang Sourdeau, September 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.

   A sink mechanism for writing to input or output "pipes".
 */


#include <iostream>

#include "jml/arch/atomic_ops.h"
#include "jml/arch/exception.h"

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
AsyncFdOutputSink(const OnHangup & onHangup, const OnClose & onClose,
                  int bufferSize)
    : AsyncWriterSource(nullptr, nullptr, nullptr, bufferSize, 0),
      OutputSink(onClose),
      onHangup_(onHangup)
{
}

AsyncFdOutputSink::
~AsyncFdOutputSink()
{
}

void
AsyncFdOutputSink::
init(int outputFd)
{
    setFd(outputFd);
}

bool
AsyncFdOutputSink::
write(std::string && data)
{
    return AsyncWriterSource::write(move(data), nullptr);
}

void
AsyncFdOutputSink::
requestClose()
{
    AsyncWriterSource::requestClose();
}

void
AsyncFdOutputSink::
onClosed(bool fromPeer, const vector<string> & msgs)
{
    if (fromPeer) {
        if (onHangup_) {
            onHangup_();
        }
    }
    else {
        doClose();
    }
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

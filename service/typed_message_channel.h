/* typed_message_channel.h                                         -*- C++ -*-
   Jeremy Barnes, 31 May 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   An internal message channel that keeps a ring of typed objects that
   are being fed between threads.
*/

#pragma once

#include "jml/utils/ring_buffer.h"
#include "jml/arch/wakeup_fd.h"
#include "soa/service/message_loop.h"

namespace Datacratic {


template<typename Message>
struct TypedMessageChannel {
    ML::RingBufferSRMW<Message> buf;
};

template<typename Message>
struct TypedMessageSink: public AsyncEventSource {

    TypedMessageSink(size_t bufferSize)
        : wakeup(EFD_NONBLOCK), buf(bufferSize)
    {
    }

    std::function<void (Message && message)> onEvent;

    void push(const Message & message)
    {
        buf.push(message);
        wakeup.signal();
    }

    void push(Message && message)
    {
        buf.push(message);
        wakeup.signal();
    }

    //protected:
    virtual int selectFd() const
    {
        return wakeup.fd();
    }

    virtual bool poll() const
    {
        return buf.couldPop();
    }

    virtual bool processOne()
    {
        // Try to do one
        Message msg;
        if (!buf.tryPop(msg))
            return false;
        onEvent(std::move(msg));

        // Are there more waiting for us?
        if (buf.couldPop())
            return true;
        
        // Warning: race condition... that's why we need the couldPop from
        // the next instruction to be accurate
        wakeup.tryRead();

        return buf.couldPop();
    }

private:
    ML::Wakeup_Fd wakeup;
    ML::RingBufferSRMW<Message> buf;
};

} // namespace Datacratic

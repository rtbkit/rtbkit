/* typed_message_channel.h                                         -*- C++ -*-
   Jeremy Barnes, 31 May 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   An internal message channel that keeps a ring of typed objects that
   are being fed between threads.
*/

#pragma once

#include <queue>
#include <thread>

#include "jml/utils/ring_buffer.h"
#include "jml/arch/wakeup_fd.h"
#include "soa/service/async_event_source.h"


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

    template<typename MessageT>
    void push(MessageT&& message)
    {
        buf.push(std::forward<MessageT>(message));
        wakeup.signal();
    }

    template<typename MessageT>
    bool tryPush(MessageT&& message)
    {
        bool pushed = buf.tryPush(std::forward<MessageT>(message));
        if (pushed)
            wakeup.signal();

        return pushed;
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
    uint64_t size() const { return buf.ring.size() ; }
private:
    ML::Wakeup_Fd wakeup;
    ML::RingBufferSRMW<Message> buf;
};


/*****************************************************************************
 * TYPED MESSAGE QUEUE                                                       *
 *****************************************************************************/

class test_typed_message_queue;

/* A multiple writer/consumer thread-safe message queue similar to the above
 * but only optionally bounded. When bounded, the advantage over the above is
 * that the limit can be dynamically adjusted. */
template<typename Message>
struct TypedMessageQueue: public AsyncEventSource
{
    friend class test_typed_message_queue;

    /* Type of callback invoked when one or more messages have become
     * available. If the function returns "true", it will continue to be
     * invoked periodically as long as there are elements in the queue,
     * otherwise it will stop after the first invocation and until the queue
     * has been emptied completely. It is the receiver's responsibility to
     * consume the queue using "pop_front". */
    typedef std::function<void ()> OnNotify;

    /* "onNotify": callback used when one or more messages are reported in the
     * queue
     * "maxMessages": maximum size of the queue, 0 for unlimited */
    TypedMessageQueue(const OnNotify & onNotify = nullptr, size_t maxMessages = 0)
        : maxMessages_(maxMessages),
          wakeup_(EFD_NONBLOCK | EFD_CLOEXEC), pending_(false),
          onNotify_(onNotify)
    {
    }

    /* AsyncEventSource interface */
    virtual int selectFd() const
    {
        return wakeup_.fd();
    }

    virtual bool processOne()
    {
        while (wakeup_.tryRead());
        onNotify();
        
        return false;
    }

    virtual void onNotify()
    {
        if (onNotify_) {
            onNotify_();
        }
    }

    /* reset the maximum number of messages */
    void setMaxMessages(size_t count)
    {
        maxMessages_ = count;
    }

    /* push message into the queue */
    bool push_back(Message message)
    {
        Guard guard(queueLock_);

        if (maxMessages_ > 0 && queue_.size() >= maxMessages_) {
            return false;
        }

        queue_.emplace(std::move(message));
        if (!pending_) {
            pending_ = true;
            wakeup_.signal();
        }

        return true;
    }

    /* returns up to "number" messages from the queue or all of them if 0 */
    std::vector<Message> pop_front(size_t number)
    {
        std::vector<Message> messages;
        Guard guard(queueLock_);

        size_t queueSize = queue_.size();
        if (number == 0 || number > queueSize) {
            number = queueSize;
        }
        messages.reserve(number);

        for (size_t i = 0; i < number; i++) {
            messages.emplace_back(std::move(queue_.front()));
            queue_.pop();
        }

        if (queue_.size() == 0) {
            pending_ = false;
        }

        return messages;
    }

    /* number of messages present in the queue */
    uint64_t size()
        const
    {
        return queue_.size();
    }

private:
    typedef std::mutex Mutex;
    typedef std::unique_lock<Mutex> Guard;
    Mutex queueLock_;
    std::queue<Message> queue_;
    size_t maxMessages_;

    ML::Wakeup_Fd wakeup_;

    /* notifications are pending */
    bool pending_;

    /* callback */
    OnNotify onNotify_;
};

} // namespace Datacratic

/* ring_buffer.h                                                   -*- C++ -*-
   Jeremy Barnes, 25 May 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Ring buffer for when there are one or more producers and one consumer
   chasing each other.
*/

#ifndef __jml_utils__ring_buffer_h__
#define __jml_utils__ring_buffer_h__

#include <vector>
#include "jml/arch/futex.h"
#include "jml/arch/spinlock.h"
#include <mutex>
#include <thread>

namespace ML {

template<typename Request>
struct RingBufferBase {
    RingBufferBase(size_t size)
    {
        init(size);
    }

    RingBufferBase(RingBufferBase && other)
        noexcept
    {
        *this = std::move(other);
    }

    RingBufferBase & operator = (RingBufferBase && other)
        noexcept
    {
        ring = std::move(other.ring);
        bufferSize = other.bufferSize;
        other.bufferSize = 0;
        readPosition = other.readPosition;
        other.readPosition = 0;
        writePosition = other.writePosition;
        other.writePosition = 0;

        return *this;
    }

    std::vector<Request> ring;
    int bufferSize;
    int readPosition;
    int writePosition;

    void init(size_t numEntries)
    {
        ring.resize(numEntries);
        bufferSize = numEntries;
        readPosition = 0;
        writePosition = 0;
    }
};

/*****************************************************************************/
/* RING BUFFER SINGLE WRITER MULTIPLE READERS                                */
/*****************************************************************************/

/** Single writer multiple reader ring buffer. */
template<typename Request>
struct RingBufferSWMR : public RingBufferBase<Request> {
    using RingBufferBase<Request>::init;
    using RingBufferBase<Request>::ring;
    using RingBufferBase<Request>::bufferSize;
    using RingBufferBase<Request>::readPosition;
    using RingBufferBase<Request>::writePosition;

    RingBufferSWMR(size_t size)
        : RingBufferBase<Request>(size)
    {
    }

    typedef std::timed_mutex Mutex;
    mutable Mutex readMutex;  // todo: we don't need this... get rid of it
    typedef std::unique_lock<Mutex> Guard;

    void push(const Request & request)
    {
        for (;;) {
            // What position would the read position be in if the buffer was
            // full?  
            unsigned fullReadPosition = (writePosition + 1) % bufferSize;
            if (readPosition == fullReadPosition)
                ML::futex_wait(readPosition, fullReadPosition);
            else break;
        }

        ring[writePosition] = request;
        writePosition = (writePosition + 1) % bufferSize;
        ML::futex_wake(writePosition);
    }

    void push(Request && request)
    {
        for (;;) {
            // What position would the read position be in if the buffer was
            // full?  
            unsigned fullReadPosition = (writePosition + 1) % bufferSize;
            if (readPosition == fullReadPosition)
                ML::futex_wait(readPosition, fullReadPosition);
            else break;
        }
        
        std::swap(ring[writePosition], request);
        writePosition = (writePosition + 1) % bufferSize;
        ML::futex_wake(writePosition);
    }

    void waitUntilEmpty()
    {
        for (;;) {
            int readPos = readPosition;
            if (readPos == writePosition)
                return;
            ML::futex_wait(readPosition, readPos);
        }
    }

    bool tryPush(const Request & request)
    {
        // What position would the read position be in if the buffer was
        // full?  
        unsigned fullReadPosition = (writePosition + 1) % bufferSize;
        if (readPosition == fullReadPosition)
            return false;
                       
        ring[writePosition] = request;
        writePosition = (writePosition + 1) % bufferSize;
        ML::futex_wake(writePosition);
        return true;
    }

    bool tryPush(Request && request)
    {
        // What position would the read position be in if the buffer was
        // full?  
        unsigned fullReadPosition = (writePosition + 1) % bufferSize;
        if (readPosition == fullReadPosition)
            return false;
                       
        std::swap(ring[writePosition], request);
        writePosition = (writePosition + 1) % bufferSize;
        ML::futex_wake(writePosition);
        return true;
    }

    Request pop()
    {
        Request result;

        {
            Guard guard(readMutex);  // todo... can get rid of this one...

            // Wait until write position != read position, ie not full
            for (;;) {
                if (writePosition == readPosition)
                    ML::futex_wait(writePosition, readPosition);
                else break;
            }

            result = ring[readPosition];
            ring[readPosition] = Request();
            readPosition = (readPosition + 1) % bufferSize;
        }
        ML::futex_wake(readPosition);

        return result;
    }

    bool tryPop(Request & result, double maxWaitTime)
    {
        {
            // todo... can get rid of this one...
            Guard guard(readMutex,
                        std::chrono::microseconds(uint64_t(maxWaitTime * 1000000)));
            if (!guard)
                return false;

            // Wait until write position != read position, ie not full
            if (writePosition == readPosition) {
                ML::futex_wait(writePosition, readPosition, maxWaitTime);
                if (writePosition == readPosition) return false;
            }

            //std::swap(result, ring[readPosition]);
            result = std::move(ring[readPosition]);
            //ring[readPosition] = Request();
            readPosition = (readPosition + 1) % bufferSize;
        }
        ML::futex_wake(readPosition);

        return true;
    }

    bool tryPop(Request & result)
    {
        {
            // todo... can get rid of this one...
            Guard guard(readMutex, std::try_to_lock_t());
            if (!guard)
                return false;

            // Wait until write position != read position, ie not full
            if (writePosition == readPosition)
                return false;
            
            //std::swap(result, ring[readPosition]);
            result = std::move(ring[readPosition]);
            //ring[readPosition] = Request();
            readPosition = (readPosition + 1) % bufferSize;
        }
        ML::futex_wake(readPosition);

        return true;
    }

};


/*****************************************************************************/
/* RING BUFFER SINGLE READER MULTIPLE WRITERS                                */
/*****************************************************************************/

template<typename Request>
struct RingBufferSRMW : public RingBufferBase<Request> {
    using RingBufferBase<Request>::init;
    using RingBufferBase<Request>::ring;
    using RingBufferBase<Request>::bufferSize;
    using RingBufferBase<Request>::readPosition;
    using RingBufferBase<Request>::writePosition;

    RingBufferSRMW(size_t size)
        : RingBufferBase<Request>(size)
    {
    }

    RingBufferSRMW(const RingBufferSRMW & other) = delete;
    RingBufferSRMW & operator = (const RingBufferSRMW & other) = delete;

    RingBufferSRMW(RingBufferSRMW && other)
        noexcept
        : RingBufferBase<Request>(std::move(other))
    {
    }

    RingBufferSRMW & operator = (RingBufferSRMW && other)
        noexcept
    {
        Guard guard(other.mutex);

        RingBufferBase<Request>::operator = (std::move(other));

        return *this;
    }

    typedef ML::Spinlock Mutex;
    mutable Mutex mutex; // todo: get rid of...
    typedef std::unique_lock<Mutex> Guard;
    
    void push(const Request & request)
    {
        Guard guard(mutex);

        for (;;) {
            // What position would the read position be in if the buffer was
            // full?  
            unsigned fullReadPosition = (writePosition + 1) % bufferSize;
            if (readPosition == fullReadPosition)
                ML::futex_wait(readPosition, fullReadPosition);
            else break;
        }
                       
        //ring[writePosition] = request;
        ring[writePosition] = request;
        writePosition = (writePosition + 1) % bufferSize;
        ML::futex_wake(writePosition);
    }

    bool tryPush(const Request & request)
    {
        Guard guard(mutex);

        // What position would the read position be in if the buffer was
        // full?  
        unsigned fullReadPosition = (writePosition + 1) % bufferSize;
        if (readPosition == fullReadPosition)
            return false;

        //ring[writePosition] = request;
        ring[writePosition] = request;
        writePosition = (writePosition + 1) % bufferSize;
        ML::futex_wake(writePosition);

        return true;
    }

    bool tryPush(Request && request)
    {
        Guard guard(mutex);

        // What position would the read position be in if the buffer was
        // full?
        unsigned fullReadPosition = (writePosition + 1) % bufferSize;
        if (readPosition == fullReadPosition)
            return false;

        //ring[writePosition] = request;
        ring[writePosition] = std::move(request);
        writePosition = (writePosition + 1) % bufferSize;
        ML::futex_wake(writePosition);

        return true;
    }

    Request pop()
    {
        Request result;

        {
            // Wait until write position != read position, ie not full
            for (;;) {
                if (writePosition == readPosition)
                    ML::futex_wait(writePosition, readPosition);
                else break;
            }

            result = ring[readPosition];
            ring[readPosition] = Request();
            readPosition = (readPosition + 1) % bufferSize;
        }
        ML::futex_wake(readPosition);

        return result;
    }

    bool tryPop(Request & result)
    {
        if (writePosition == readPosition)
            return false;

        result = ring[readPosition];
        ring[readPosition] = Request();
        readPosition = (readPosition + 1) % bufferSize;
        ML::futex_wake(readPosition);
        
        return true;
    }

    bool tryPop(Request & result, double maxWaitTime)
    {
        {
            // Wait until write position != read position, ie not full
            for (;;) {
                if (writePosition == readPosition) {
                    if (ML::futex_wait(writePosition, readPosition, maxWaitTime) == -1
                        && errno == ETIMEDOUT)
                        return false;
                }
                else break;
            }
            
            result = ring[readPosition];
            ring[readPosition] = Request();
            readPosition = (readPosition + 1) % bufferSize;
        }
        ML::futex_wake(readPosition);

        return true;
    }

    std::vector<Request> tryPopMulti(size_t nbrRequests)
    {
        std::vector<Request> result;

        for (size_t i = 0;
             writePosition != readPosition && i < nbrRequests;
             i++) {
            result.emplace_back(std::move(ring[readPosition]));
            ring[readPosition] = Request();
            readPosition = (readPosition + 1) % bufferSize;
        }
        ML::futex_wake(readPosition);
        
        return result;
    }

    bool couldPop() const
    {
        return writePosition != readPosition;
    }
};

} // namespace ML

#endif /* __jml_utils__ring_buffer_h__ */

/** thread_specific.h                                              -*- C++ -*-
    Jeremy Barnes, 13 November 2011
    Copyright (c) 2011 Datacratic.  All rights reserved.

    Placed under the BSD license.

    Contains code to deal with thread specific data in such a way that:
    a) it's deleted (unlike the __thread keyword);
    b) it's not dog slow (unlike the boost::thread_specific_ptr)

    The downside is that each needs to be tagged with a unique type as
    it requires static variables.
*/
#ifndef __arch__thread_specific_h__
#define __arch__thread_specific_h__

#include "exception.h"
#include "spinlock.h"
#include "jml/utils/exc_assert.h"
#include <thread>
#include "spinlock.h"

#include <boost/thread.hpp>
#include <unordered_set>
#include <mutex>

namespace ML {

/*****************************************************************************/
/* THREAD SPECIFIC                                                           */
/*****************************************************************************/

/** A fast thread specific variable. */

template<typename Contained, typename Tag = void>
struct Thread_Specific {

    static Contained * defaultCreateFn ()
    {
        return new Contained();
    }

    Thread_Specific()
        : createFn(defaultCreateFn)
    {
    }

    Thread_Specific(boost::function<Contained * ()> createFn)
        : createFn(createFn)
    {
    }

    boost::function<Contained * ()> createFn;

    static __thread Contained * ptr_;

    void create() const
    {
        ptr_ = createFn();
        if (!ptr_)
            throw Exception("bad pointer");
        assert(!deleter.get());
        deleter.reset(new Deleter(ptr_));
    }

    JML_ALWAYS_INLINE Contained * operator -> () const
    {
        if (!ptr_) create();
        return ptr_;
    }

    JML_ALWAYS_INLINE Contained * get() const
    {
        return operator -> ();
    }

    JML_ALWAYS_INLINE Contained & operator * () const
    {
        return * operator -> ();
    }

    struct Deleter {

        Contained * ptr;

        Deleter(Contained * ptr)
            : ptr(ptr)
        {
        }

        ~Deleter()
        {
            delete ptr;
        }
    };

    mutable boost::thread_specific_ptr<Deleter> deleter;
};

template<typename Contained, typename Tag>
__thread Contained * Thread_Specific<Contained, Tag>::ptr_ = 0;


/*****************************************************************************/
/* THREAD SPECIFIC INSTANCE INFO                                             */
/*****************************************************************************/

/** This structure allows information to be stored per instance of a variable
    per thread.  To do so, include this structure somewhere in the
    class that you want to associate the info with.

    Note that while this class has several locks, they're only grabbed when an
    instance is created, destroyed or first accessed. Past the first access,
    reads equate to a deque probe.

 */
template<typename T, typename Tag>
struct ThreadSpecificInstanceInfo
{
    typedef ML::Spinlock Lock;

    struct Value
    {
        Value() : object(nullptr) {}
        ~Value()
        {
            ThreadSpecificInstanceInfo* oldObject = destruct();
            if (!oldObject) return;

            std::lock_guard<Lock> guard(oldObject->freeSetLock);
            oldObject->freeSet.erase(this);
        }

        ThreadSpecificInstanceInfo* destruct()
        {
            std::lock_guard<Lock> guard(destructLock);
            if (!object) return nullptr;

            storage.value.~T();
            auto oldObject = object;
            object = nullptr;

            return oldObject;
        }

        // This can't raise with either object destruction or thread destruction
        // so no locks are needed.
        void construct(ThreadSpecificInstanceInfo* newObject)
        {
            new (&storage.value) T();
            object = newObject;
        }

        /** The odd setup is to prevent spurious calls to the T constructor and
            destructor when we construct our parent class Value.

            Note that using a union is a well defined type-puning construct in
            gcc while reinterpret_cast<> could cause problems when used with
            strict-aliasing (I think). Feel free to simplify it if I'm wrong.
         */
        union Storage
        {
            Storage() {}
            ~Storage() {}

            T value;
            uint8_t unused[sizeof(T)];
        } storage;


        Lock destructLock;
        ThreadSpecificInstanceInfo* object;
    };

    typedef std::deque<Value> PerThreadInfo;

    ThreadSpecificInstanceInfo()
    {
        std::lock_guard<Lock> guard(freeIndexLock);

        if (!freeIndexes.empty()) {
            index = freeIndexes.front();
            freeIndexes.pop_front();
        }
        else index = ++nextIndex;
    }

    ~ThreadSpecificInstanceInfo()
    {
        // We don't want to be holding the freeSet lock when calling destruct
        // because thread destruction will also attempt to lock our freeSet lock
        // which is a receipie for deadlocks.
        std::unordered_set<Value*> freeSetCopy;
        {
            std::lock_guard<Lock> guard(freeSetLock);
            freeSetCopy = std::move(freeSet);
        }

        for (Value* toFree : freeSetCopy)
            toFree->destruct();

        std::lock_guard<Lock> guard(freeIndexLock);
        freeIndexes.push_back(index);
    }

    static PerThreadInfo * getThisThread()
    {
        return staticInfo.get();
    }

    T * get(PerThreadInfo * & info) const
    {
        if (!info) info = staticInfo.get();
        return load(info);
    }

    T * get(PerThreadInfo * const & info) const
    {
        load(info);
    }

    /** Return the data for this thread for this instance of the class. */
    T * get() const
    {
        PerThreadInfo * info = staticInfo.get();
        return load(info);
    }

private:

    T * load(PerThreadInfo * info) const
    {
        while (info->size() <= index)
            info->emplace_back();

        Value& val = (*info)[index];

        if (JML_UNLIKELY(!val.object)) {
            val.construct(const_cast<ThreadSpecificInstanceInfo*>(this));
            std::lock_guard<Lock> guard(freeSetLock);
            freeSet.insert(&val);
        }

        return &val.storage.value;
    }

    static Thread_Specific<PerThreadInfo> staticInfo;

    static ML::Spinlock freeIndexLock;
    static std::deque<size_t> freeIndexes;
    static unsigned nextIndex;
    int index;

    mutable ML::Spinlock freeSetLock;
    mutable std::unordered_set<Value*> freeSet;
};

template<typename T, typename Tag>
Thread_Specific<typename ThreadSpecificInstanceInfo<T, Tag>::PerThreadInfo>
ThreadSpecificInstanceInfo<T, Tag>::staticInfo;

template<typename T, typename Tag>
ML::Spinlock
ThreadSpecificInstanceInfo<T, Tag>::freeIndexLock;

template<typename T, typename Tag>
std::deque<size_t>
ThreadSpecificInstanceInfo<T, Tag>::freeIndexes;

template<typename T, typename Tag>
unsigned
ThreadSpecificInstanceInfo<T, Tag>::nextIndex = 0;

} // namespace ML


#endif /* __arch__thread_specific_h__ */

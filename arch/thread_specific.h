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

#include <boost/thread.hpp>
#include "exception.h"
#include "jml/utils/exc_assert.h"

namespace ML {

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
*/
template<typename T, typename Tag>
struct ThreadSpecificInstanceInfo {

    struct PerThreadInfo {
        PerThreadInfo()
        {
            threadNum = __sync_fetch_and_add(&totalNumThreads, 1);
        }

        ~PerThreadInfo()
        {
            // TODO: release the thread number so it doesn't grow indefinitely
        }

        int threadNum;
        static int totalNumThreads;

        std::vector<T> info;

        T * get(int index)
        {
            ExcAssertGreaterEqual(index, 0);
            if (info.size() <= index)
                info.resize(index + 1);
            return &info[index];
        }
    };

    ThreadSpecificInstanceInfo()
        : index(__sync_fetch_and_add(&currentIndex, 1))
    {
    }

    int index;
    static Thread_Specific<PerThreadInfo> staticInfo;
    static int currentIndex;

    static PerThreadInfo * getThisThread()
    {
        return staticInfo.get();
    }

    T * get(PerThreadInfo * & info) const
    {
        if (!info) info = staticInfo.get();
        return info->get(index);
    }

    T * get(PerThreadInfo * const & info) const
    {
        return info->get(index);
    }

    /** Return the data for this thread for this instance of the class. */
    T * get() const
    {
        PerThreadInfo * info = staticInfo.get();
        return info->get(index);
    }
};

template<typename T, typename Tag>
Thread_Specific<typename ThreadSpecificInstanceInfo<T, Tag>::PerThreadInfo>
ThreadSpecificInstanceInfo<T, Tag>::staticInfo;

template<typename T, typename Tag>
int
ThreadSpecificInstanceInfo<T, Tag>::currentIndex = 0;

template<typename T, typename Tag>
int
ThreadSpecificInstanceInfo<T, Tag>::PerThreadInfo::
totalNumThreads = 0;

} // namespace ML


#endif /* __arch__thread_specific_h__ */

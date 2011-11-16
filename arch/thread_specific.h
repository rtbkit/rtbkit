/** thread_specific.h                                              -*- C++ -*-
    Jeremy Barnes, 13 November 2011
    Copyright (c) 2011 Recoset.  All rights reserved.

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
        assert(!deleter);
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

} // namespace ML


#endif /* __arch__thread_specific_h__ */

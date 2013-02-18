/* rcu_protected.h                                                 -*- C++ -*-
   Jeremy Barnes, 12 April 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Building blocks for RCU protected data structures.
*/

#ifndef __mmap__rcu_protected_h__
#define __mmap__rcu_protected_h__

#include "gc_lock.h"
#include "jml/utils/unnamed_bool.h"
#include "jml/arch/atomic_ops.h"

namespace Datacratic {

template<typename T>
struct RcuLocked {
    RcuLocked(T * ptr = 0, GcLock * lock = 0)
        : ptr(ptr), lock(lock)
    {
        if (lock)
            lock->lockShared();
    }

    /// Transfer from another lock
    template<typename T2>
    RcuLocked(T * ptr, RcuLocked<T2> && other)
        : ptr(ptr), lock(other.lock)
    {
        other.lock = 0;
    }

    /// Copy from another lock
    template<typename T2>
    RcuLocked(T * ptr, const RcuLocked<T2> & other)
        : ptr(ptr), lock(other.lock)
    {
        if (lock)
            lock->lockShared();
    }

    template<typename T2>
    RcuLocked(RcuLocked<T2> && other)
        : ptr(other.ptr), lock(other.lock)
    {
        other.lock = 0;
    }

    RcuLocked & operator = (RcuLocked && other)
    {
        unlock();
        lock = other.lock;
        ptr = other.ptr;
        return *this;
    }

    template<typename T2>
    RcuLocked & operator = (RcuLocked<T2> && other)
    {
        unlock();
        lock = other.lock;
        ptr = other.ptr;
        return *this;
    }

    ~RcuLocked()
    {
        unlock();
    }

    void unlock()
    {
        if (lock) {
            lock->unlockShared();
            lock = 0;
        }
    }

    T * ptr;
    GcLock * lock;

    operator T * () const
    {
        return ptr;
    }

    T * operator -> () const
    {
        if (!ptr)
            throw ML::Exception("dereferencing null RCUResult");
        return ptr;
    }

    T & operator * () const
    {
        if (!ptr)
            throw ML::Exception("dereferencing null RCUResult");
        return *ptr;
    }
};

template<typename T>
struct RcuProtected {
    T * val;
    GcLock * lock;

    template<typename... Args>
    RcuProtected(GcLock & lock, Args&&... args)
        : val(new T(std::forward<Args>(args)...)), lock(&lock)
    {
        //ExcAssert(this->lock);
    }

    RcuProtected(T * val, GcLock & lock)
        : val(val), lock(&lock)
    {
        //ExcAssert(this->lock);
    }

    RcuProtected(RcuProtected && other)
        : val(other.val), lock(other.lock)
    {
        //ExcAssert(this->lock);
        other.val = 0;
    }

    RcuProtected & operator = (RcuProtected && other)
    {
        auto toDelete = val;
        val = other.val;
        lock = other.lock;
        other.val = 0;
        lock->deferDelete(toDelete);
        //ExcAssert(lock);
        return *this;
    }

    ~RcuProtected()
    {
        lock->deferDelete(val);
        val = 0;
    }

    JML_IMPLEMENT_OPERATOR_BOOL(val);

    RcuLocked<T> operator () ()
    {
        //ExcAssert(lock);
        return RcuLocked<T>(val, lock);
    }

    RcuLocked<const T> operator () () const
    {
        //ExcAssert(lock);
        return RcuLocked<const T>(val, lock);
    }

    void replace(T * newVal, bool defer = true)
    {
        T * toDelete = ML::atomic_xchg(val, newVal);
        if (toDelete) {
            ExcAssertNotEqual(toDelete, val);
            if (defer) lock->deferDelete(toDelete);
            else {
                lock->visibleBarrier();
                delete toDelete;
            }
        }
    }
    
    bool cmp_xchg(RcuLocked<T> & current, std::auto_ptr<T> & newValue,
                  bool defer = true)
    {
        T * currentVal = current.ptr;
        if (ML::cmp_xchg(val, currentVal, newValue.get())) {
            ExcAssertNotEqual(currentVal, val);
            if (currentVal) {
                if (defer) 
                    lock->deferDelete(currentVal);
                else {
                    lock->visibleBarrier();
                    delete currentVal;
                }
            }
            newValue.release();
            current.ptr = val;
            return true;
        }
        return false;
    }

private:
    // Don't allow copy semantics (use RcuProtectedCopyable for that).  Just
    // move semantics are OK.
    RcuProtected();
    RcuProtected(const RcuProtected & other);
    void operator = (const RcuProtected & other);
};

template<typename T>
struct RcuProtectedCopyable : public RcuProtected<T> {

    using RcuProtected<T>::val;
    using RcuProtected<T>::lock;
    using RcuProtected<T>::operator ();
    using RcuProtected<T>::replace;

    template<typename... Args>
    RcuProtectedCopyable(GcLock & lock, Args&&... args)
        : RcuProtected<T>(lock, std::forward<Args>(args)...)
    {
    }

    RcuProtectedCopyable(const RcuProtectedCopyable & other)
        : RcuProtected<T>(new T(*other()), *other.lock)
    {
    }

    RcuProtectedCopyable(RcuProtectedCopyable && other)
        : RcuProtected<T>(static_cast<RcuProtected<T> &&>(other))
    {
    }

    RcuProtectedCopyable & operator = (const RcuProtectedCopyable & other)
    {
        //ExcAssert(lock);
        auto toDelete = val;
        lock = other.lock;
        val = new T(*other());
        if (toDelete) lock->deferDelete(toDelete);
        return *this;
    }

    RcuProtectedCopyable & operator = (RcuProtectedCopyable && other)
    {
        //ExcAssert(lock);
        auto toDelete = val;
        lock = other.lock;
        val = other.val;
        other.val = 0;
        if (toDelete) lock->deferDelete(toDelete);
        return *this;
    }

};

} // namespace Datacratic

#endif /* __mmap__rcu_protected_h__ */

   

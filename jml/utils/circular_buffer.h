/* circular_buffer.h                                               -*- C++ -*-
   Jeremy Barnes, 7 December 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   Circular buffer structure, that will grow to hold whatever we put in it.
   O(1) insertion and deletion at the front and at the back.  Kind of like
   a deque, but much lighter weight.
*/

#ifndef __jml__utils__circular_buffer_h__
#define __jml__utils__circular_buffer_h__

#include <vector>
#include "jml/arch/exception.h"
#include <boost/iterator/iterator_facade.hpp>
#include <iostream> // debug

namespace ML {

template<typename T, class CircularBuffer>
struct Circular_Buffer_Iterator
    : public boost::iterator_facade<Circular_Buffer_Iterator<T, CircularBuffer>,
                                    T,
                                    boost::random_access_traversal_tag> {
    CircularBuffer * buffer;
    int index;
    bool wrapped;
    
    typedef boost::iterator_facade<Circular_Buffer_Iterator<T, CircularBuffer>,
                                   T,
                                   boost::random_access_traversal_tag> Base;

    enum { safe = CircularBuffer::safe };
public:
    typedef T & reference;
    typedef int difference_type;
    
    Circular_Buffer_Iterator()
        : buffer(0), index(0), wrapped(false)
    {
    }
    
    Circular_Buffer_Iterator(CircularBuffer * buffer, int idx, bool wrapped)
        : buffer(buffer), index(idx), wrapped(wrapped)
    {
    }
    
    template<typename T2, class CircularBuffer2>
    difference_type
    operator - (const Circular_Buffer_Iterator<T2, CircularBuffer2>
                    & other) const
    {
            return distance_to(other);
    }
    
    template<typename T2, class CircularBuffer2>
    bool operator == (const Circular_Buffer_Iterator<T2, CircularBuffer2>
                          & other) const
    {
        return equal(other);
    }
    
    template<typename T2, class CircularBuffer2>
    bool operator != (const Circular_Buffer_Iterator<T2, CircularBuffer2>
                          & other) const
    {
        return ! operator == (other);
    }
    
    using Base::operator -;
    
    std::string print() const
    {
        return format("Circular_Buffer_Iterator: buffer %p (start %d size %d capacity %d) index %d wrapped %d",
                      buffer,
                      (buffer ? buffer->start_ : 0),
                      (buffer ? buffer->size_ : 0),
                      (buffer ? buffer->capacity_ : 0),
                      index, wrapped);
    }
    
private:
    friend class boost::iterator_core_access;
    
    template<typename T2, class CircularBuffer2> friend class Iterator;
    
    template<typename T2, class CircularBuffer2>
    bool equal(const Circular_Buffer_Iterator<T2, CircularBuffer2>
                   & other) const
    {
        if (safe && buffer != other.buffer)
            throw Exception("wrong buffer");
        return index == other.index && wrapped == other.wrapped;
    }
    
    T & dereference() const
    {
        if (safe && !buffer)
            throw Exception("defererencing null iterator");
        return buffer->vals_[index] ;
    }
    
    void increment()
    {
        ++index;
        bool wrap = (index >= buffer->capacity_);
        index *= (!wrap);
        wrapped = wrapped || wrap;
        return;
    }
    
    void decrement()
    {
        if (safe && index == buffer->start_ && !wrapped)
            throw Exception("decrementing off the end");

        //cerr << "decrement: " << print() << endl;

        --index;
        if (index < 0) {
            wrapped = false;
            index += buffer->capacity_;
        }

        //cerr << "after decrement: " << print() << endl;
    }
    
    void advance(int nelements)
    {
        //cerr << "advance: before " << print() << endl;
        //cerr << "nelements = " << nelements << endl;

        index += nelements;
        bool wrap1 = index >= buffer->capacity_;
        bool wrap2 = index < 0;
        index += (wrap2 - wrap1) * buffer->capacity_;
        wrapped = (wrapped || wrap1) && !wrap2;
    }
    
    template<typename T2, class CircularBuffer2>
    int distance_to(const Circular_Buffer_Iterator<T2, CircularBuffer2>
                        & other) const
    {
        if (safe && buffer != other.buffer)
            throw Exception("other buffer wrong");
        
        //cerr << "distance_to:" << endl;
        //cerr << " this:   " << print() << endl;
        //cerr << " other:  " << other.print() << endl;
        
        // What is the distance from the start to this element?
        int dstart1 = index - buffer->start_;
        if (wrapped) dstart1 += buffer->capacity_;

        int dstart2 = other.index - buffer->start_;
        if (other.wrapped) dstart2 += buffer->capacity_;

        //cerr << " dist1:  " << dstart1 << endl;
        //cerr << " dist2:  " << dstart2 << endl;
        //cerr << " result: " << dstart1 - dstart2 << endl;

        return dstart1 - dstart2;
    }
};


template<typename T, bool Safe = false,
         class Allocator = std::allocator<T> >
struct Circular_Buffer {
    enum { safe = Safe };

    Circular_Buffer(int initial_capacity = 0)
        : vals_(0), start_(0), size_(0), capacity_(0)
    {
        if (initial_capacity != 0) reserve(initial_capacity);
    }

    void swap(Circular_Buffer & other)
    {
        std::swap(vals_, other.vals_);
        std::swap(start_, other.start_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
    }

    Circular_Buffer(const Circular_Buffer & other)
        : vals_(0), start_(0), size_(0), capacity_(0)
    {
        reserve(other.size());
        for (unsigned i = 0;  i < other.size();  ++i)
            push_back(other[i]);
    }

    Circular_Buffer & operator = (const Circular_Buffer & other)
    {
        Circular_Buffer new_me(other);
        swap(new_me);
        return *this;
    }

    ~Circular_Buffer()
    {
        destroy();
    }

    bool empty() const { return size_ == 0; }
    size_t size() const { return size_; }
    size_t capacity() const { return capacity_; }
    size_t start() const { return start_; }

    void reserve(int new_capacity)
    {
        //cerr << "reserve: capacity = " << capacity_ << " new_capacity = "
        //     << new_capacity << endl;

        if (new_capacity <= capacity_) return;
        new_capacity = std::max(capacity_ * 2, new_capacity);

        int nfirst_half = std::min(capacity_ - start_, size_);
        int nsecond_half = std::max(0, size_ - nfirst_half);

        T * new_vals = allocator.allocate(new_capacity);

        //cerr << "start_ = " << start_ << " size_ = " << size_ << endl;
        //cerr << "nfirst_half = " << nfirst_half << endl;
        //cerr << "nsecond_half = " << nsecond_half << endl;

        // TODO: exception safety if constructors throw

        unsigned i;
        try {
            for (i = 0;  i < nfirst_half;  ++i)
                new (new_vals + i) T(vals_[start_ + i]);
        } catch (...) {
            for (i -= 1; i >= 0;  --i) {
                try {
                    new_vals[i].~T();
                } catch (...) {}
            }
            try {
                allocator.deallocate(new_vals, new_capacity);
            } catch (...) {}
            throw;
        }

        try {
            for (unsigned i = 0;  i < nsecond_half;  ++i)
                new (new_vals + nfirst_half + i) T(vals_[i]);
        } catch (...) {
            for (i -= 1; i >= 0;  --i) {
                try {
                    new_vals[i + nfirst_half].~T();
                } catch (...) {}
            }
            for (unsigned i = 0;  i < nfirst_half;  ++i) {
                try {
                    vals_[start_ + i].~T();
                } catch (...) {}
            }
            try {
                allocator.deallocate(new_vals, new_capacity);
            } catch (...) {}
            throw;
        }

        size_t sz = size_;

        destroy();

        vals_ = new_vals;
        capacity_ = new_capacity;
        size_ = sz;
    }

    void resize(size_t new_size, const T & el = T())
    {
        while (new_size < size_)
            pop_back();
        while (new_size > size_)
            push_back(el);
    }

    void destroy()
    {
        clear();
        allocator.deallocate(vals_, capacity_);

        vals_ = 0;
        capacity_ = 0;
    }

    void clear(int start = 0)
    {
        if (start < 0 || start > capacity_ || (start == capacity_ && capacity_))
            throw Exception("invalid start");

        int nfirst_half = std::min(capacity_ - start_, size_);
        int nsecond_half = std::max(0, size_ - nfirst_half);

        // Those after start and before capacity
        for (unsigned i = 0;  i < nfirst_half;  ++i) {
            try {
                vals_[start_ + i].~T();
            } catch (...) {}
        }
        // Those from 0 to the end
        for (unsigned i = 0;  i < nsecond_half;  ++i) {
            try {
                vals_[i].~T();
            } catch (...) {}
        }            
        
        size_ = 0;
        start_ = start;
    }

    const T & operator [](int index) const
    {
        if (size_ == 0)
            throw Exception("Circular_Buffer: empty array");
        if (index < -size_ || index >= size_)
            throw Exception("Circular_Buffer: invalid index");
        return *element_at(index);
    }

    T & operator [](int index)
    {
        const Circular_Buffer * cthis = this;
        return const_cast<T &>(cthis->operator [] (index));
    }

    const T & at(int index) const
    {
        if (size_ == 0)
            throw Exception("Circular_Buffer: empty array");
        if (index < -size_ || index >= size_)
            throw Exception("Circular_Buffer: invalid index");
        return *element_at(index);
    }

    T & at(int index)
    {
        const Circular_Buffer * cthis = this;
        return const_cast<T &>(cthis->operator [] (index));
    }

    const T & front() const
    {
        if (empty())
            throw Exception("front() with empty circular array");
        return vals_[start_];
    }

    T & front()
    {
        if (empty())
            throw Exception("front() with empty circular array");
        return vals_[start_];
    }

    const T & back() const
    {
        if (empty())
            throw Exception("back() with empty circular array");
        return *element_at(size_ - 1);
    }

    T & back()
    {
        if (empty())
            throw Exception("back() with empty circular array");
        return *element_at(size_ - 1);
    }

    void push_back(const T & val)
    {
        if (size_ == capacity_) reserve(std::max(1, capacity_ * 2));
        ++size_;
        new (element_at(size_ - 1)) T(val);
    }

    void push_front(const T & val)
    {
        if (size_ == capacity_) reserve(std::max(1, capacity_ * 2));
        new (element_at(-1)) T(val);
        if (start_ == 0) start_ = capacity_ - 1;
        else --start_;
    }

    void pop_back()
    {
        if (empty())
            throw Exception("pop_back with empty circular array");
        element_at(-1)->~T();
        --size_;
    }

    void pop_front()
    {
        if (empty())
            throw Exception("pop_front with empty circular array");
        element_at(0)->~T();
        ++start_;
        --size_;

        // Point back to the start if empty
        if (start_ == capacity_) start_ = 0;
    }

    void erase_element(int el)
    {
        //cerr << "erase_element: el = " << el << " size = " << size()
        //     << endl;

        if (el >= size() || el < -(int)size())
            throw Exception("erase_element(): invalid value");
        if (el < 0) el += size_;

        int offset = (start_ + el);
        offset -= capacity_ * (offset >= capacity_);

        erase_element_at(offset);
    }

    typedef Circular_Buffer_Iterator<T, Circular_Buffer> iterator;
    typedef Circular_Buffer_Iterator<const T, const Circular_Buffer>
        const_iterator;

    iterator begin()
    {
        return iterator(this, start_, capacity_ == 0 /* wrapped */);
    }

    iterator end()
    {
        return begin() + size_;
    }
    
    const_iterator begin() const
    {
        return const_iterator(this, start_, capacity_ == 0 /* wrapped */);
    }

    const_iterator end() const
    {
        return begin() + size_;
    }

    void erase(const iterator & it)
    {
        erase_element_at(it.index);
    }

#if 0
    template<typename OtherIt>
    void insert(iterator where, OtherIt first, OtherIt last)
    {
        int n = std::distance(first, last);
        if (n < 0)
            throw Exception("invalid range to insert");
        int offset = (where - begin());
        if (offset < 0 || offset > size_)
            throw Exception("insert(): invalid offset");
        if (size_ + n > capacity_)
            reserve(size_ + n);

        // iterator may have been invalidated by the reserve; we can't use it
        
        // Two choices: a) we move elements after to the end;
        // b) we move elements before to the start
        int nbefore = offset;
        int nafter = size_ - offset;
        if (nafter < nbefore) {
            // push everything after the insert back
            // Initialize new elements
            for (unsigned i = 0;  i < n;  ++i)
                new (element_at(size_ + i)) T();

            // we move those after to the end
            for (int i = size_;  i > offset;  --i)
                *element_at(i + n) = *element_at(i);
            
            // Copy the new ones in
            for (unsigned i = 0;  i < n;  ++i)
                *element_at(offset + i) = *first++;
            
            if (first != last)
                throw Exception("invalid iterators");
        }
        else {
            // push everything before the insert forwards
        }
    }

    void erase(iterator first, iterator last)
    {
        if (first.buffer != this || second.buffer != this)
            throw Exception("erase with invalid iterator range");
        
        int offset = (where - begin());
        int n = last - first;
    }
#endif    

private:
    template<typename T2, class CB> friend class Circular_Buffer_Iterator;

    T * vals_;
    int start_;
    int size_;
    int capacity_;

    void validate() const
    {
        if (!vals_ && capacity_ != 0)
            throw Exception("null vals but capacity");
        if (start_ < 0)
            throw Exception("negative start");
        if (size_ < 0)
            throw Exception("negative size");
        if (capacity_ < 0)
            throw Exception("negaive capacity");
        if (size_ > capacity)
            throw Exception("capacity too high");
        if (start_ > size_ || (start_ == size_ && size_ != 0))
            throw Exception("start too far");
    }

    T * element_at(int index)
    {
        if (capacity_ == 0)
            throw Exception("empty circular buffer");

        //cerr << "element_at: index " << index << " start_ " << start_
        //     << " capacity_ " << capacity_ << " size_ " << size_;

        if (index < 0) index += size_;

        int offset = (start_ + index);
        offset -= capacity_ * (offset >= capacity_);

        //cerr << "  offset " << offset << endl;

        return vals_ + offset;
    }
    
    const T * element_at(int index) const
    {
        if (capacity_ == 0)
            throw Exception("empty circular buffer");

        //cerr << "element_at: index " << index << " start_ " << start_
        //     << " capacity_ " << capacity_ << " size_ " << size_;

        index += size_ * (index < 0);

        int offset = (start_ + index);
        offset -= capacity_ * (offset >= capacity_);

        //cerr << "  offset " << offset << endl;

        return vals_ + offset;
    }

    void erase_element_at(unsigned offset)
    {
        // TODO: could be done more efficiently

        if (offset == start_) {
            pop_front();
        }
        else if (offset == start_ + size_ - 1
                 || offset == start_ + size_ - capacity_ - 1) {
            pop_back();
        }
        else if (offset < start_) {
            // Move all of the wrapped elements after this one back
            //cerr << "slide from back" << endl;
            // slide everything
            int last_element = start_ + size_;
            last_element -= capacity_ * (last_element >= capacity_);
            std::copy(vals_ + offset + 1, vals_ + last_element,
                      vals_ + offset);
            pop_back();
        }
        else {
            //cerr << "slide from front" << endl;
            for (int i = offset;  i > start_;  --i) {
                //cerr << "setting element " << i << " old value "
                //     << vals_[i] << " from element " << i - 1
                //     << " old value " << vals_[i - 1] << endl;
                vals_[i] = vals_[i - 1];
            }
            pop_front();
        }
    }

    static Allocator allocator;
};

template<typename T, bool S, class Allocator>
Allocator
Circular_Buffer<T, S, Allocator>::allocator;

template<typename T, bool S, class A>
bool
operator == (const Circular_Buffer<T, S, A> & cb1,
             const Circular_Buffer<T, S, A> & cb2)
{
    return cb1.size() == cb2.size()
        && std::equal(cb1.begin(), cb1.end(), cb2.begin());
}

template<typename T, bool S, class A>
bool
operator < (const Circular_Buffer<T, S, A> & cb1,
            const Circular_Buffer<T, S, A> & cb2)
{
    return std::lexicographical_compare(cb1.begin(), cb1.end(),
                                        cb2.begin(), cb2.end());
}

template<typename T, bool S, class A>
std::ostream & operator << (std::ostream & stream,
                            const Circular_Buffer<T, S, A> & buf)
{
    stream << "[";
    for (unsigned i = 0;  i < buf.size();  ++i)
        stream << " " << buf[i];
    return stream << " ]";
}

template<typename T, class CB>
std::ostream &
operator << (std::ostream & stream,
             const Circular_Buffer_Iterator<T, CB> & it)
{
    return stream << it.print();
}


} // namespace ML


#endif /* __jml__utils__circular_buffer_h__ */

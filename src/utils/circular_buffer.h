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

namespace ML {

template<typename T, class Underlying = std::vector<T>, bool Safe=false>
struct Circular_Buffer {
    Circular_Buffer(int initial_capacity = 0)
        : start_(0)
    {
        if (initial_capacity != 0) reserve(initial_capacity);
    }

    Circular_Buffer(const Circular_Buffer & other)
        : start_(0), vals_(other.begin(), other.end())
    {
    }

    Circular_Buffer & operator = (const Circular_Buffer & other)
    {
        Circular_Buffer new_me(other);
        swap(new_me);
        return *this;
    }

    void swap(Circular_Buffer & other)
    {
        vals_.swap(other.vals_);
        std::swap(start_, other.start_);
    }

    bool empty() const { return vals_.empty(); }
    size_t size() const { return vals_.size(); }
    size_t capacity() const { return vals_.capacity(); }
    
    void clear()
    {
        vals_.clear();
        start_ = 0;
    }

    const T & operator [](int index) const
    {
        if (index < 0) index += size();
        if (Safe) {
            if (size_ == 0)
                throw Exception("Circular_Buffer: empty array");
            if (index < -size_ || index >= size_)
                throw Exception("Circular_Buffer: invalid size");
        }
        return vals_[index];
    }

    T & operator [](int index)
    {
        const Circular_Buffer * cthis = this;
        return const_cast<T &>(cthis->operator [] (index));
    }

    void reserve(int new_capacity)
    {
        if (vals.capacity() >= new_capacity) return;

        new_capacity = std::max(vals.capacity() * 2, new_capacity);

        Vals new_vals;
        new_vals.reserve(new_capacity);
        new_vals.insert(new_vals.end(), ...);


        T * new_vals = new T[new_capacity];
        memset(new_vals, 0xaa, sizeof(T) * new_capacity);

        int nfirst_half = std::min(capacity_ - start_, size_);
        int nsecond_half = std::max(0, size_ - nfirst_half);

        //cerr << "start_ = " << start_ << " size_ = " << size_ << endl;
        //cerr << "nfirst_half = " << nfirst_half << endl;
        //cerr << "nsecond_half = " << nsecond_half << endl;

        std::copy(vals_ + start_, vals_ + start_ + nfirst_half,
                  new_vals);
        std::copy(vals_ + start_ - nsecond_half, vals_ + start_,
                  new_vals + nfirst_half);

        memset(vals_, 0xaa, sizeof(T) * capacity_);

        delete[] vals_;

        vals_ = new_vals;
        capacity_ = new_capacity;
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
        *element_at(size_) = val;
    }

    void push_front(const T & val)
    {
        if (size_ == capacity_) reserve(std::max(1, capacity_ * 2));
        *element_at(-1) = val;
        if (start_ == 0) start_ = capacity_ - 1;
        else --start_;
    }

    void pop_back()
    {
        if (empty())
            throw Exception("pop_back with empty circular array");
        memset(element_at(size_ - 1), 0xaa, sizeof(T));
        --size_;
    }

    void pop_front()
    {
        if (empty())
            throw Exception("pop_front with empty circular array");
        memset(vals_ + start_, 0xaa, sizeof(T));
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

        if (capacity_ == 0)
            throw Exception("empty circular buffer");

        int offset = (start_ + el) % capacity_;

        //cerr << "offset = " << offset << " start_ = " << start_
        //     << " size_ = " << size_ << " capacity_ = " << capacity_
        //     << endl;

        // TODO: could be done more efficiently

        if (el == 0) {
            pop_front();
        }
        else if (el == size() - 1) {
            pop_back();
        }
        else if (offset < start_) {
            // slide everything 
            int num_to_do = size_ - el - 1;
            std::copy(vals_ + offset + 1, vals_ + offset + 1 + num_to_do,
                      vals_ + offset);
            --size_;
        }
        else {
            for (int i = offset;  i > 0;  --i)
                vals_[i] = vals_[i - 1];
            --size_;
            ++start_;
        }
    }

private:
    T * vals_;
    int start_;
    int size_;
    int capacity_;

    T * element_at(int index)
    {
        if (capacity_ == 0)
            throw Exception("empty circular buffer");

        //cerr << "element_at: index " << index << " start_ " << start_
        //     << " capacity_ " << capacity_ << " size_ " << size_;

        if (index < 0) index += size_;

        int offset = (start_ + index) % capacity_;

        //cerr << "  offset " << offset << endl;

        return vals_ + offset;
    }
    
    const T * element_at(int index) const
    {
        if (capacity_ == 0)
            throw Exception("empty circular buffer");

        //cerr << "element_at: index " << index << " start_ " << start_
        //     << " capacity_ " << capacity_ << " size_ " << size_;

        if (index < 0) index += size_;

        int offset = (start_ + index) % capacity_;

        //cerr << "  offset " << offset << endl;

        return vals_ + offset;
    }
};


} // namespace ML


#endif /* __jml__utils__circular_buffer_h__ */

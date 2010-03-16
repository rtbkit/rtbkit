/* lightweight_hash.h                                              -*- C++ -*-
   Jeremy Barnes, 8 December 2009
   Copyright (c) 2009 Jeremy Barnes.  All rights reserved.

   A lightweight invasive hash map.
*/

#ifndef __jml__utils__lightweight_hash_h__
#define __jml__utils__lightweight_hash_h__

#include "jml/arch/exception.h"
#include <boost/iterator/iterator_facade.hpp>
#include <iostream> // debug
#include "jml/utils/hash_specializations.h"
#include "jml/utils/string_functions.h"
#include <string>
#include <cassert>

namespace ML {

template<typename Key, typename Value, class Hash>
class Lightweight_Hash_Iterator
    : public boost::iterator_facade<Lightweight_Hash_Iterator<Key, Value, Hash>,
                                    std::pair<const Key, Value>,
                                    boost::bidirectional_traversal_tag> {

    typedef boost::iterator_facade<Lightweight_Hash_Iterator<Key, Value, Hash>,
                                   std::pair<const Key, Value>,
                                   boost::bidirectional_traversal_tag> Base;
public:    
    Lightweight_Hash_Iterator()
        : hash(0), index(0)
    {
    }

    Lightweight_Hash_Iterator(Hash * hash, int index)
        : hash(hash), index(index)
    {
        if (index != hash->capacity_)
            advance_to_valid();
    }

    template<typename K2, typename V2, typename H2>
    Lightweight_Hash_Iterator(const Lightweight_Hash_Iterator<K2, V2, H2> & other)
        : hash(other.hash), index(other.index)
    {
    }

    std::string print() const
    {
        return format("Lightweight_Hash_Iterator: hash %p index %d",
                      hash, index);
    }

private:
    // Index in hash of current entry.  It is allowed to point to any
    // valid bucket of the underlying hash, OR to one-past-the-end of the
    // capacity, which means at the end.
public:
    Hash * hash;
    int index;

    friend class boost::iterator_core_access;
    
    template<typename K2, typename V2, typename H2>
    bool equal(const Lightweight_Hash_Iterator<K2, V2, H2> & other) const
    {
        if (hash != other.hash)
            throw Exception("comparing incompatible iterators");
        return index == other.index;
    }
    
    std::pair<const Key, Value> & dereference() const
    {
        if (!hash)
            throw Exception("dereferencing null iterator");
        if (index < 0 || index > hash->capacity_)
            throw Exception("dereferencing invalid iterator");
        if (!hash->vals_[index].first)
            throw Exception("dereferencing invalid iterator bucket");
        
        return reinterpret_cast<std::pair<const Key, Value> &>(hash->vals_[index]);
    }
    
    void increment()
    {
        if (index == hash->capacity_)
            throw Exception("increment past the end");
        ++index;
        if (index != hash->capacity_) advance_to_valid();
    }
    
    void decrement()
    {
        if (index == 0)
            throw Exception("decrement past the start");
        --index;
        if (index != 0) backup_to_valid();
    }

    void advance_to_valid()
    {
        if (index < 0 || index >= hash->capacity_) {
            hash->dump(std::cerr);
            std::cerr << "index = " << index << std::endl;
            throw Exception("advance_to_valid: already at end");
        }

        // Scan through until we find a valid bucket
        while (index < hash->capacity_ && !hash->vals_[index].first)
            ++index;
    }

    void backup_to_valid()
    {
        if (index < 0 || index >= hash->capacity_)
            throw Exception("backup_to_valid: already outside range");
        
        // Scan through until we find a valid bucket
        while (index >= 0 && !hash->vals_[index].first)
            --index;
        
        if (index < 0)
            throw Exception("backup_to_valid: none found");
    }

    template<typename K2, typename V2, typename H2>
    friend class Lightweight_Hash_Iterator;
};

template<typename Key, typename Value, class Hash>
std::ostream &
operator << (std::ostream & stream,
             const Lightweight_Hash_Iterator<Key, Value, Hash> & it)
{
    return stream << it.print();
}

template<typename Key, typename Value, class Hash = std::hash<Key>,
         class Allocator = std::allocator<std::pair<Key, Value> > >
struct Lightweight_Hash {

    typedef Lightweight_Hash_Iterator<Key, const Value, const Lightweight_Hash>
    const_iterator;
    typedef Lightweight_Hash_Iterator<Key, Value, Lightweight_Hash> iterator;

    Lightweight_Hash()
        : vals_(0), size_(0), capacity_(0)
    {
    }

    template<class Iterator>
    Lightweight_Hash(Iterator first, Iterator last, size_t capacity = 0)
        : vals_(0), size_(0), capacity_(capacity)
    {
        if (capacity_ == 0)
            capacity_ = std::distance(first, last) * 2;

        if (capacity_ == 0) return;

        vals_ = allocator.allocate(capacity_);

        for (unsigned i = 0;  i < capacity_;  ++i)
            new (&vals_[i].first) Key(0);

#if 0
        using namespace std;

        cerr << "first.index = " << first.index << endl;
        cerr << "last.index = " << last.index << endl;
        cerr << "first.hash->size() = " << first.hash->size() << endl;
        cerr << "first.hash->capacity() = " << first.hash->capacity() << endl;
        cerr << "capacity = " << capacity << endl;
        //cerr << "distance = " << std::distance(first, last) << endl;
#endif
        for (; first != last;  ++first) {
            //cerr << "first.index = " << first.index << " last.index = "
            //     << last.index << endl;
            insert(*first);
        }

        //cerr << "finished inserting" << endl;
    }

    Lightweight_Hash(const Lightweight_Hash & other)
        : vals_(0), size_(other.size_), capacity_(other.capacity_)
    {
        if (capacity_ == 0) return;

        vals_ = allocator.allocate(capacity_);

        // TODO: exception cleanup?

        for (unsigned i = 0;  i < capacity_;  ++i) {
            if (other.vals_[i].first)
                new (vals_ + i) std::pair<Key, Value>(other.vals_[i]);
            else new (&vals_[i].first) Key(0);
        }
    }

    ~Lightweight_Hash()
    {
        destroy();
    }

    Lightweight_Hash & operator = (const Lightweight_Hash & other)
    {
        Lightweight_Hash new_me(other);
        swap(new_me);
        return *this;
    }

    void swap(Lightweight_Hash & other)
    {
        std::swap(vals_, other.vals_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
    }

    iterator begin()
    {
        if (empty()) return end();
        return iterator(this, 0);
    }

    iterator end()
    {
        return iterator(this, capacity_);
    }

    const_iterator begin() const
    {
        if (empty()) return end();
        return const_iterator(this, 0);
    }

    const_iterator end() const
    {
        return const_iterator(this, capacity_);
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    size_t capacity() const { return capacity_; }

    void clear()
    {
        // Run destructors
        for (unsigned i = 0;  i < capacity_;  ++i) {
            if (vals_[i].first) {
                try {
                    vals_[i].second.~Value();
                } catch (...) {}
                vals_[i].first = 0;
            }
        }

        size_ = 0;
    }

    void destroy()
    {
        clear();
        try {
            allocator.deallocate(vals_, capacity_);
        } catch (...) {}
        vals_ = 0;
        capacity_ = 0;
    }

    iterator find(const Key & key)
    {
        size_t hashed = Hash()(key);
        int bucket = find_bucket(hashed, key);
        if (bucket == -1 || !vals_[bucket].first) return end();
        if (vals_[bucket].first != key) {
            using namespace std;
            dump(cerr);
            cerr << "bucket = " << bucket << endl;
            cerr << "hashed = " << hashed << endl;
            cerr << "key = " << key << endl;
            cerr << "vals_[bucket].first = " << vals_[bucket].first << endl;
            throw Exception("find_bucket didn't return correct key");
        }
        assert(vals_[bucket].first == key);
        return iterator(this, bucket);
    }

    const_iterator find(const Key & key) const
    {
        size_t hashed = Hash()(key);
        int bucket = find_bucket(hashed, key);
        if (bucket == -1 || !vals_[bucket].first) return end();
        assert(vals_[bucket].first == key);
        return const_iterator(this, bucket);
    }

    Value & operator [] (const Key & key)
    {
        size_t hashed = Hash()(key);
        int bucket = find_bucket(hashed, key);
        if (bucket == -1 || !vals_[bucket].first)
            bucket = insert_new(bucket, key, hashed, Value());
        assert(vals_[bucket].first == key);
        return vals_[bucket].second;
    }

    std::pair<iterator, bool>
    insert(const std::pair<Key, Value> & val)
    {
        size_t hashed = Hash()(val.first);
        int bucket = find_bucket(hashed, val.first);
        if (bucket != -1 && vals_[bucket].first) {
            assert(vals_[bucket].first == val.first);
            return make_pair(iterator(this, bucket), false);
        }
        bucket = insert_new(bucket, val.first, hashed, val.second);
        return make_pair(iterator(this, bucket), true);
    }

    void reserve(size_t new_capacity)
    {
        if (new_capacity <= capacity_) return;

        if (new_capacity < capacity_ * 2)
            new_capacity = capacity_ * 2;

        Lightweight_Hash new_me(begin(), end(), new_capacity);
        swap(new_me);
    }

private:
    template<typename K, typename V, class H>
    friend class Lightweight_Hash_Iterator;

    std::pair<Key, Value> * vals_;
    int size_;
    int capacity_;

    int find_bucket(size_t hash, const Key & key) const
    {
        if (!key)
            throw Exception("searching for or inserting guard value");

        if (capacity_ == 0) return -1;
        int bucket = hash % capacity_;
        bool wrapped = false;
        int i;
        for (i = bucket;  vals_[i].first && (i != bucket || !wrapped);
             /* no inc */) {
            if (vals_[i].first == key) return i;
            ++i;
            if (i == capacity_) { i = 0;  wrapped = true; }
        }

        if (!vals_[i].first) return i;

        // No bucket found; will need to be expanded
        if (size_ != capacity_) {
            dump(std::cerr);
            throw Exception("find_bucket: inconsistency");
        }
        return -1;
    }

    void dump(std::ostream & stream) const
    {
        using namespace std;
        stream << "Lightweight_Hash: size " << size_ << " capacity "
               << capacity_ << endl;
        for (unsigned i = 0;  i < capacity_;  ++i) {
            stream << "  bucket " << i << ": hash "
                   << (Hash()(vals_[i].first) % capacity_)
                   << " key " << vals_[i].first;
            if (vals_[i].first)
                stream << " value " << vals_[i].second;
            stream << endl;
        }
    }

        
    int insert_new(int bucket, const Key & key, size_t hashed,
                   const Value & val)
    {
        if (!key)
            throw Exception("searching for or inserting guard value");

        if (size_ >= 3 * capacity_ / 4) {
            // expand
            reserve(std::max(4, capacity_ * 2));
            bucket = find_bucket(hashed, key);
            if (bucket == -1 || vals_[bucket].first)
                throw Exception("logic error: bucket appeared after reserve");
        }

        new (&vals_[bucket].second) Value(val);
        vals_[bucket].first = key;
        ++size_;

        return bucket;
    }

    static Allocator allocator;
};

template<typename Key, typename Value, class Hash, class Allocator>
Allocator
Lightweight_Hash<Key, Value, Hash, Allocator>::allocator;


} // file scope


#endif /* __jml__utils__lightweight_hash_h__ */

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
#include "jml/utils/pair_utils.h"
#include <string>
#include <cassert>
#include <functional>

namespace ML {


/*****************************************************************************/
/* LIGHTWEIGHT HASH ITERATOR                                                 */
/*****************************************************************************/

template<typename Key, typename Value, class Hash, class ConstKeyBucket>
class Lightweight_Hash_Iterator
    : public boost::iterator_facade<Lightweight_Hash_Iterator<Key, Value, Hash, ConstKeyBucket>,
                                    ConstKeyBucket,
                                    boost::bidirectional_traversal_tag> {

    typedef boost::iterator_facade<Lightweight_Hash_Iterator<Key, Value, Hash, ConstKeyBucket>,
                                   ConstKeyBucket,
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
            this->index = hash->advance_to_valid(index);
    }

    template<typename K2, typename V2, typename H2, typename CB2>
    Lightweight_Hash_Iterator(const Lightweight_Hash_Iterator<K2, V2, H2, CB2> & other)
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
    
    template<typename K2, typename V2, typename H2, typename CB2>
    bool equal(const Lightweight_Hash_Iterator<K2, V2, H2, CB2> & other) const
    {
        if (hash != other.hash)
            throw Exception("comparing incompatible iterators");
        return index == other.index;
    }
    
    ConstKeyBucket & dereference() const
    {
        if (!hash)
            throw Exception("dereferencing null iterator");
        return hash->dereference(index);
    }
    
    void increment()
    {
        if (index == hash->capacity_)
            throw Exception("increment past the end");
        ++index;
        if (index != hash->capacity_) index = hash->advance_to_valid(index);
    }
    
    void decrement()
    {
        if (index == 0)
            throw Exception("decrement past the start");
        --index;
        if (index != 0) index = hash->backup_to_valid(index);
    }

    template<typename K2, typename V2, typename H2, typename CB2>
    friend class Lightweight_Hash_Iterator;
};

template<typename Key, typename Value, class Hash, class CB>
std::ostream &
operator << (std::ostream & stream,
             const Lightweight_Hash_Iterator<Key, Value, Hash, CB> & it)
{
    return stream << it.print();
}


/*****************************************************************************/
/* LIGHTWEIGHT HASH BASE                                                     */
/*****************************************************************************/

template<class Key, class Bucket, class Ops, class Allocator>
struct Lightweight_Hash_Base {

    Lightweight_Hash_Base()
        : vals_(0), size_(0), capacity_(0)
    {
    }

    template<class Iterator>
    Lightweight_Hash_Base(Iterator first, Iterator last, size_t capacity = 0)
        : vals_(0), size_(0), capacity_(capacity)
    {
        if (capacity_ == 0)
            capacity_ = std::distance(first, last) * 2;

        if (capacity_ == 0) return;

        vals_ = allocator.allocate(capacity_);

        for (unsigned i = 0;  i < capacity_;  ++i)
            Ops::initEmptyBucket(vals_ + i);

        for (; first != last;  ++first)
            insert(*first);
    }

    Lightweight_Hash_Base(const Lightweight_Hash_Base & other,
                          size_t capacity)
        : vals_(0), size_(0), capacity_(capacity)
    {
        vals_ = allocator.allocate(capacity_);

        for (unsigned i = 0;  i < capacity_;  ++i)
            Ops::initEmptyBucket(vals_ + i);

        for (unsigned i = 0;  i < other.capacity_;  ++i)
            if (Ops::bucketIsFull(other.vals_ + i))
                must_insert(other.vals_[i]);
    }

    Lightweight_Hash_Base(const Lightweight_Hash_Base & other)
        : vals_(0), size_(other.size_), capacity_(other.capacity_)
    {
        if (capacity_ == 0) return;

        vals_ = allocator.allocate(capacity_);

        // TODO: exception cleanup?

        for (unsigned i = 0;  i < capacity_;  ++i) {
            if (Ops::bucketIsFull(other.vals_ + i))
                Ops::initBucket(vals_ + i, other.vals_[i]);
            else Ops::initEmptyBucket(vals_ + i);
        }
    }

    ~Lightweight_Hash_Base()
    {
        destroy();
    }

    Lightweight_Hash_Base & operator = (const Lightweight_Hash_Base & other)
    {
        Lightweight_Hash_Base new_me(other);
        swap(new_me);
        return *this;
    }

    void swap(Lightweight_Hash_Base & other)
    {
        std::swap(vals_, other.vals_);
        std::swap(size_, other.size_);
        std::swap(capacity_, other.capacity_);
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    size_t capacity() const { return capacity_; }

    void clear()
    {
        // Run destructors
        for (unsigned i = 0;  i < capacity_;  ++i) {
            if (Ops::bucketIsFull(vals_ + i)) {
                try {
                    Ops::emptyBucket(vals_ + i);
                } catch (...) {}
            }
        }

        size_ = 0;
    }

    void destroy()
    {
        // Run destructors
        for (unsigned i = 0;  i < capacity_;  ++i) {
            try {
                Ops::destroyBucket(vals_ + i);
            } catch (...) {}
        }
        
        // Destroy the underlying memory
        try {
            allocator.deallocate(vals_, capacity_);
        } catch (...) {}
        vals_ = 0;
        size_ = 0;
        capacity_ = 0;
    }

    void reserve(size_t new_capacity)
    {
        if (new_capacity <= capacity_) return;

        if (new_capacity < capacity_ * 2)
            new_capacity = capacity_ * 2;

        Lightweight_Hash_Base new_me(*this, new_capacity);
        swap(new_me);
    }

    void dump(std::ostream & stream) const
    {
        using namespace std;
        stream << "Lightweight_Hash: size " << size_ << " capacity "
               << capacity_ << endl;
        for (unsigned i = 0;  i < capacity_;  ++i) {
            stream << "  bucket " << i << ": hash "
                   << Ops::hashKey(vals_[i], capacity_)
                   << " bucket " << vals_[i] << endl;
        }
    }

protected:
    Bucket * vals_;
    int size_;
    int capacity_;

    std::pair<int, bool>
    find_or_insert(const Bucket & toInsert)
    {
        Key key = Ops::getKey(toInsert);
        int bucket = find_bucket(key);
        if (bucket != -1 && Ops::bucketIsFull(vals_ + bucket))
            return std::make_pair(bucket, false);
        return std::make_pair(insert_new(bucket, toInsert), true);
    }

    int must_insert(const Bucket & toInsert)
    {
        Key key = Ops::getKey(toInsert);
        int bucket = find_bucket(key);
        if (bucket != -1 && Ops::bucketIsFull(vals_ + bucket))
            throw ML::Exception("must_insert of value already there");
        return insert_new(bucket, toInsert);
    }

    int find_bucket(const Key & key) const
    {
        if (Ops::isGuardValue(key))
            throw Exception("searching for or inserting guard value");

        if (capacity_ == 0) return -1;
        int bucket = Ops::hashKey(key, capacity_);
        bool wrapped = false;
        int i;
        for (i = bucket;  Ops::bucketIsFull(vals_ + i) && (i != bucket || !wrapped);
             /* no inc */) {
            if (Ops::bucketHasKey(vals_ + i, key)) return i;
            ++i;
            if (i == capacity_) { i = 0;  wrapped = true; }
        }

        if (!Ops::bucketIsFull(vals_ + i)) return i;

        // No bucket found; will need to be expanded
        if (size_ != capacity_) {
            dump(std::cerr);
            throw Exception("find_bucket: inconsistency");
        }
        return -1;
    }

    int find_full_bucket(const Key & key) const
    {
        int bucket = find_bucket(key);
        if (bucket == -1 || !Ops::bucketIsFull(vals_ + bucket)) return -1;
        if (!Ops::bucketHasKey(vals_ + bucket, key)) {
#if 0
            using namespace std;
            dump(cerr);
            cerr << "bucket = " << bucket << endl;
            cerr << "hashed = " << hashed << endl;
            cerr << "key = " << key << endl;
            cerr << "vals_[bucket].first = " << vals_[bucket].first << endl;
#endif
            throw Exception("find_full_bucket didn't return correct key");
        }
        
        return bucket;
    }

    int insert_new(int bucket, const Bucket & toInsert)
    {
        Key key = Ops::getKey(toInsert);
        if (Ops::isGuardValue(key))
            throw Exception("searching for or inserting guard value");
        
        if (size_ >= 3 * capacity_ / 4) {
            // expand
            reserve(std::max(4, capacity_ * 2));
            bucket = find_bucket(key);
            if (bucket == -1 || Ops::bucketIsFull(vals_ + bucket))
                throw Exception("logic error: bucket appeared after reserve");
        }

        Ops::fillBucket(vals_ + bucket, toInsert);
        ++size_;

        return bucket;
    }

    int advance_to_valid(int index) const
    {
        if (index < 0 || index >= capacity_) {
            dump(std::cerr);
            std::cerr << "index = " << index << std::endl;
            throw Exception("advance_to_valid: already at end");
        }

        // Scan through until we find a valid bucket
        while (index < capacity_ && !Ops::bucketIsFull(vals_ + index))
            ++index;

        return index;
    }

    int backup_to_valid(int index) const
    {
        if (index < 0 || index >= capacity_)
            throw Exception("backup_to_valid: already outside range");
        
        // Scan through until we find a valid bucket
        while (index >= 0 && !Ops::bucketIsFull(vals_ + index))
            --index;
        
        if (index < 0)
            throw Exception("backup_to_valid: none found");

        return index;
    }

    const Bucket & dereference(int bucket) const
    {
        if (bucket < 0 || bucket > capacity_)
            throw Exception("dereferencing invalid iterator");
        if (!Ops::bucketIsFull(vals_ + bucket)) {
            using namespace std;
            cerr << "bucket = " << bucket << endl;
            dump(cerr);
            throw Exception("dereferencing invalid iterator bucket");
        }
        return this->vals_[bucket];
    }

    static Allocator allocator;
};


/*****************************************************************************/
/* LIGHTWEIGHT HASH MAP                                                      */
/*****************************************************************************/

template<typename Key, typename Value, typename Hash>
struct PairOps {
    typedef std::pair<Key, Value> Bucket;

    static void initEmptyBucket(Bucket * bucket)
    {
        new (bucket) Bucket();
    }

    static void initBucket(Bucket * bucket, const Bucket & value)
    {
        new (bucket) Bucket(value);
    }

    static void fillBucket(Bucket * bucket, const Bucket & value)
    {
        *bucket = value;
    }
    
    static void emptyBucket(Bucket * bucket)
    {
        bucket->~Bucket();
        initEmptyBucket(bucket);
    }

    static void destroyBucket(Bucket * bucket)
    {
        bucket->~Bucket();
    }
    
    static bool bucketIsFull(Bucket * bucket) JML_PURE_FN
    {
        return bucket->first;
    }

    static bool isGuardValue(Key key) JML_CONST_FN
    {
        return key == 0;
    }

    static bool bucketHasKey(Bucket * bucket, Key key) JML_PURE_FN
    {
        return bucket->first == key;
    }

    static Key getKey(Bucket bucket) JML_CONST_FN
    {
        return bucket.first;
    }

    static size_t hashKey(Bucket bucket, int capacity) JML_CONST_FN
    {
        return hashKey(getKey(bucket), capacity);
    }

    static size_t hashKey(Key key, int capacity) JML_CONST_FN
    {
        return Hash()(key) % capacity;
    }
 };

template<typename Key, class Bucket, class Ops, class Allocator>
Allocator Lightweight_Hash_Base<Key, Bucket, Ops, Allocator>::
allocator;

template<typename Key, typename Value, class Hash = std::hash<Key>,
         class Bucket = std::pair<Key, Value>,
         class ConstKeyBucket = std::pair<const Key, Value>,
         class Ops = PairOps<Key, Value, Hash>,
         class Allocator = std::allocator<Bucket> >
struct Lightweight_Hash
    : public Lightweight_Hash_Base<Key, Bucket, Ops, Allocator> {

    typedef Lightweight_Hash_Iterator<Key, const Value, const Lightweight_Hash,
                                      const Bucket>
    const_iterator;
    typedef Lightweight_Hash_Iterator<Key, Value, Lightweight_Hash,
                                      ConstKeyBucket> iterator;

    typedef Lightweight_Hash_Base<Key, Bucket, Ops, Allocator> Base;

    Lightweight_Hash()
    {
    }

    template<class Iterator>
    Lightweight_Hash(Iterator first, Iterator last, size_t capacity = 0)
        : Base(first, last, capacity)
    {
    }

    Lightweight_Hash(const Lightweight_Hash & other)
        : Base(other)
    {
    }

    Lightweight_Hash & operator = (const Lightweight_Hash & other)
    {
        Lightweight_Hash new_me(other);
        swap(new_me);
        return *this;
    }

    void swap(Lightweight_Hash & other)
    {
        Base::swap(other);
    }

    using Base::size;
    using Base::empty;
    using Base::capacity;
    using Base::clear;

    iterator begin()
    {
        if (empty()) return end();
        return iterator(this, 0);
    }

    iterator end()
    {
        return iterator(this, this->capacity_);
    }

    const_iterator begin() const
    {
        if (empty()) return end();
        return const_iterator(this, 0);
    }

    const_iterator end() const
    {
        return const_iterator(this, this->capacity_);
    }

    iterator find(const Key & key)
    {
        int bucket = find_full_bucket(key);
        if (bucket == -1) return end();
        return iterator(this, bucket);
    }

    const_iterator find(const Key & key) const
    {
        int bucket = find_full_bucket(key);
        if (bucket == -1) return end();
        return const_iterator(this, bucket);
    }

    Value & operator [] (const Key & key)
    {
        int bucket = find_or_insert(Bucket(key, Value())).first;
        assert(this->vals_[bucket].first == key);
        return this->vals_[bucket].second;
    }

    std::pair<iterator, bool>
    insert(const std::pair<Key, Value> & val)
    {
        std::pair<int, bool> r = find_or_insert(val);
        return make_pair(iterator(this, r.first), r.second);
    }

    using Base::reserve;

private:
    template<typename K, typename V, class H, class CB>
    friend class Lightweight_Hash_Iterator;

    using Base::dereference;
    ConstKeyBucket & dereference(int bucket)
    {
        if (bucket < 0 || bucket > this->capacity_)
            throw Exception("dereferencing invalid iterator");
        if (!this->vals_[bucket].first) {
            using namespace std;
            cerr << "bucket = " << bucket << endl;
            dump(cerr);
            throw Exception("dereferencing invalid iterator bucket");
        }
        return reinterpret_cast<ConstKeyBucket &>(this->vals_[bucket]);
    }

    void dump(std::ostream & stream) const
    {
        using namespace std;
        stream << "Lightweight_Hash: size " << this->size_ << " capacity "
               << this->capacity_ << endl;
        for (unsigned i = 0;  i < this->capacity_;  ++i) {
            stream << "  bucket " << i << ": hash "
                   << (Hash()(this->vals_[i].first) % this->capacity_)
                   << " key " << this->vals_[i].first;
            if (this->vals_[i].first)
                stream << " value " << this->vals_[i].second;
            stream << endl;
        }
    }
};


/*****************************************************************************/
/* LIGHTWEIGHT HASH SET                                                      */
/*****************************************************************************/

template<typename Key, typename Hash, Key guard = (Key)-1>
struct ScalarOps {
    typedef Key Bucket;

    static void initEmptyBucket(Bucket * bucket)
    {
        new (bucket) Bucket(guard);
    }

    static void initBucket(Bucket * bucket, const Bucket & value)
    {
        new (bucket) Bucket(value);
    }

    static void fillBucket(Bucket * bucket, const Bucket & value)
    {
        *bucket = value;
    }
    
    static void emptyBucket(Bucket * bucket)
    {
        bucket->~Bucket();
        initEmptyBucket(bucket);
    }

    static void destroyBucket(Bucket * bucket)
    {
        bucket->~Bucket();
    }
    
    static bool bucketIsFull(Bucket * bucket)
    {
        return *bucket != guard;
    }

    static bool isGuardValue(Key key)
    {
        return key == guard;
    }

    static bool bucketHasKey(Bucket * bucket, Key key)
    {
        return *bucket == key;
    }

    static Key getKey(const Bucket & bucket)
    {
        return bucket;
    }

    static size_t hashKey(Key key, int capacity)
    {
        return Hash()(key) % capacity;
    }
};

template<typename Key, class Hash = std::hash<Key>,
         class Bucket = Key,
         class Ops = ScalarOps<Key, Hash>,
         class Allocator = std::allocator<Bucket> >
struct Lightweight_Hash_Set
    : public Lightweight_Hash_Base<Key, Bucket, Ops, Allocator> {

    typedef Lightweight_Hash_Iterator<Key, const Key, const Lightweight_Hash_Set,
                                      const Bucket>
    const_iterator;
    typedef const_iterator iterator;

    typedef Lightweight_Hash_Base<Key, Bucket, Ops, Allocator> Base;

    Lightweight_Hash_Set()
    {
    }

    template<class Iterator>
    Lightweight_Hash_Set(Iterator first, Iterator last, size_t capacity = 0)
        : Base(first, last, capacity)
    {
    }

    Lightweight_Hash_Set(const Lightweight_Hash_Set & other)
        : Base(other)
    {
    }

    Lightweight_Hash_Set & operator = (const Lightweight_Hash_Set & other)
    {
        Lightweight_Hash_Set new_me(other);
        swap(new_me);
        return *this;
    }

    void swap(Lightweight_Hash_Set & other)
    {
        Base::swap(other);
    }

    using Base::size;
    using Base::empty;
    using Base::capacity;
    using Base::clear;

    const_iterator begin() const
    {
        if (empty()) return end();
        return const_iterator(this, 0);
    }

    const_iterator end() const
    {
        return const_iterator(this, this->capacity_);
    }

    const_iterator find(const Key & key) const
    {
        int bucket = this->find_full_bucket(key);
        if (bucket == -1) return end();
        return const_iterator(this, bucket);
    }

    std::pair<const_iterator, bool>
    insert(const Key & val)
    {
        std::pair<int, bool> r = this->find_or_insert(val);
        return make_pair(const_iterator(this, r.first), r.second);
    }

    bool count(const Key & key) const
    {
        int bucket = this->find_full_bucket(key);
        return bucket != -1;
    }

    using Base::reserve;

private:
    template<typename K, typename V, class H, class CB>
    friend class Lightweight_Hash_Iterator;
};


} // file scope


#endif /* __jml__utils__lightweight_hash_h__ */

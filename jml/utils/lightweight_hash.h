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
#include "jml/utils/exc_assert.h"
#include "jml/arch/bitops.h"
#include "jml/arch/atomic_ops.h"
#include <string>
#include <cassert>
#include <functional>

namespace ML {


/*****************************************************************************/
/* LIGHTWEIGHT HASH ITERATOR                                                 */
/*****************************************************************************/

/** Iterator for a lightweight hash.  Note that bucket indexes go from -1 to
    capacity; index -1 is reserved for the bucket containing the element that
    corresponds to the "guard" value in the key.
*/

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
        : hash(0), index(-1)
    {
    }

    Lightweight_Hash_Iterator(Hash * hash, ssize_t index)
        : hash(hash), index(index)
    {
        if (index != hash->capacity())
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
    ssize_t index;

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
        if (index == hash->capacity())
            throw Exception("increment past the end");
        ++index;
        if (index != hash->capacity()) index = hash->advance_to_valid(index);
    }
    
    void decrement()
    {
        if (index == -1)
            throw Exception("decrement past the start");
        --index;
        if (index != -1) index = hash->backup_to_valid(index);
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
/* MEM STORAGE                                                               */
/*****************************************************************************/

template<typename Bucket, typename Allocator = std::allocator<Bucket> >
struct MemStorage {
    MemStorage()
        : capacity_(0), vals_(0), guardBucketIsFull_(0)
    {
    }

    MemStorage(size_t capacity)
        : capacity_(0), vals_(0), guardBucketIsFull_(0)
    {
        reserve(capacity);
    }

    MemStorage(MemStorage && other)
        : capacity_(other.capacity_), vals_(other.vals_),
          guardBucketIsFull_(other.guardBucketIsFull_)
    {
        other.capacity_ = 0;
        other.vals_ = 0;
    }

    MemStorage & operator = (MemStorage && other)
    {
        destroy();
        swap(other);
        return *this;
    }

    ~MemStorage()
    {
        destroy();
    }

    int capacity_;
    Bucket * vals_;
    uint8_t guardBucketIsFull_;

    void swap(MemStorage & other)
    {
        std::swap(capacity_, other.capacity_);
        std::swap(vals_, other.vals_);
        std::swap(guardBucketIsFull_, other.guardBucketIsFull_);
    }

    ssize_t capacity() const { return capacity_; }
    
    void reserve(size_t newCapacity)
    {
        if (vals_)
            throw ML::Exception("can't double initialize storage");
            
        // We allocate one extra bucket and shift things so that index -1 has
        // a valid bucket
        vals_ = allocator.allocate(newCapacity + 1) + 1;
        capacity_ = newCapacity;
    }

    void destroy()
    {
        if (!vals_) return;
        try {
            allocator.deallocate(vals_ - 1, capacity_ + 1);
        } catch (...) {}
        vals_ = 0;
        capacity_ = 0;
    }

    Bucket * operator + (ssize_t index)
    {
        //ExcAssert(vals_);
        //ExcAssertLess(index, capacity_);
        return vals_ + index;
    }
    
    Bucket & operator [] (ssize_t index)
    {
        //ExcAssert(vals_);
        //ExcAssertLess(index, capacity_);
        return vals_[index];
    }

    const Bucket & operator [] (ssize_t index) const
    {
        //ExcAssert(vals_);
        //ExcAssertLess(index, capacity_);
        return vals_[index];
    }

private:
    void operator = (const MemStorage & other);
    MemStorage(const MemStorage & other);
    static Allocator allocator;
};

template<typename Bucket, typename Allocator>
Allocator MemStorage<Bucket, Allocator>::
allocator;


/*****************************************************************************/
/* LOG MEM STORAGE                                                           */
/*****************************************************************************/

template<typename Bucket, typename Allocator = std::allocator<Bucket> >
struct LogMemStorage {
    LogMemStorage()
        : vals_(0), bits_(0), guardBucketIsFull_(0)
    {
    }

    LogMemStorage(size_t capacity)
        : vals_(0), bits_(0), guardBucketIsFull_(0)
    {
        reserve(capacity);
    }

    LogMemStorage(LogMemStorage && other)
        : bits_(other.bits_), vals_(other.vals_),
          guardBucketIsFull_(other.guardBucketIsFull_)
    {
        other.bits_ = 0;
        other.vals_ = 0;
    }

    LogMemStorage & operator = (LogMemStorage && other)
    {
        destroy();
        swap(other);
        return *this;
    }

    ~LogMemStorage()
    {
        destroy();
    }

    Bucket * vals_;
    uint8_t bits_;
    uint8_t guardBucketIsFull_;

    void swap(LogMemStorage & other)
    {
        std::swap(bits_, other.bits_);
        std::swap(vals_, other.vals_);
        std::swap(guardBucketIsFull_, other.guardBucketIsFull_);
    }

    ssize_t capacity() const
    {
        return size_t(bits_ != 0) * (1ULL << (bits_ - 1));
    }
    
    void reserve(size_t newCapacity)
    {
        if (vals_)
            throw ML::Exception("can't double initialize storage");
        
        if (newCapacity == 0) return;

        bits_ = ML::highest_bit((newCapacity - 1), -1) + 2;
        vals_ = allocator.allocate(capacity() + 1) + 1;

        ExcAssertGreaterEqual(capacity(), newCapacity);
    }

    void destroy()
    {
        if (!vals_) return;
        try {
            allocator.deallocate(vals_ - 1, capacity() + 1);
        } catch (...) {}
        vals_ = 0;
        bits_ = 0;
    }

    bool guardBucketIsFull() const
    {
        return guardBucketIsFull_;
    }

    void setGuardBucketIsOccupied()
    {
        ExcAssert(!guardBucketIsFull_);
        guardBucketIsFull_ = true;
    }

    Bucket * operator + (ssize_t index)
    {
        //ExcAssert(vals_);
        //ExcAssertLess(index, capacity_);
        return vals_ + index;
    }
    
    Bucket & operator [] (ssize_t index)
    {
        //ExcAssert(vals_);
        //ExcAssertLess(index, capacity_);
        return vals_[index];
    }

    const Bucket & operator [] (ssize_t index) const
    {
        //ExcAssert(vals_);
        //ExcAssertLess(index, capacity_);
        return vals_[index];
    }

private:
    void operator = (const LogMemStorage & other);
    LogMemStorage(const LogMemStorage & other);
    static Allocator allocator;
};

template<typename Bucket, typename Allocator>
Allocator LogMemStorage<Bucket, Allocator>::
allocator;


/*****************************************************************************/
/* LIGHTWEIGHT HASH BASE                                                     */
/*****************************************************************************/

template<class Key, class Bucket, class Ops, class Storage>
struct Lightweight_Hash_Base {

    Lightweight_Hash_Base()
        : size_(0)
    {
    }

    template<class Iterator>
    Lightweight_Hash_Base(Iterator first, Iterator last, size_t capacity = 0)
        : storage_(capacity), size_(0)
    {
        if (capacity == 0)
            storage_.reserve(std::distance(first, last) * 2);
        
        ssize_t cp = this->capacity();
        if (cp == 0) return;

        for (ssize_t i = -1;  i < cp;  ++i)
            Ops::initEmptyBucket(storage_ + i);

        for (; first != last;  ++first)
            this->find_or_insert(*first);
    }

    Lightweight_Hash_Base(const Lightweight_Hash_Base & other,
                          size_t capacity)
        : storage_(capacity), size_(0)
    {
        ssize_t cp = this->capacity();

        for (ssize_t i = -1;  i < cp;  ++i)
            Ops::initEmptyBucket(storage_ + i);

        ssize_t ocp = other.capacity();
        for (ssize_t i = -1;  i < ocp;  ++i)
            if (Ops::bucketIsFull(other.storage_, i))
                must_insert(other.storage_[i]);
    }

    Lightweight_Hash_Base(const Lightweight_Hash_Base & other)
        : storage_(other.capacity()), size_(other.size_)
    {
        if (capacity() == 0) return;

        ExcAssertEqual(capacity(), other.capacity());

        for (ssize_t i = -1;  i < capacity();  ++i) {
            if (Ops::bucketIsFull(other.storage_, i))
                Ops::initBucket(storage_ + i, other.storage_[i]);
            else Ops::initEmptyBucket(storage_ + i);
        }
    }

    Lightweight_Hash_Base(Lightweight_Hash_Base && other)
        : storage_(std::move(other.storage_)), size_(other.size_)
    {
        other.size_ = 0;
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

    Lightweight_Hash_Base & operator = (Lightweight_Hash_Base && other)
    {
        Lightweight_Hash_Base new_me(other);
        swap(new_me);
        return *this;
    }

    void swap(Lightweight_Hash_Base & other)
    {
        storage_.swap(other.storage_);
        std::swap(size_, other.size_);
    }

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    ssize_t capacity() const { return storage_.capacity(); }

    void clear()
    {
        ssize_t cp = capacity();

        // Empty buckets
        for (ssize_t i = -1;  i < cp;  ++i) {
            if (Ops::bucketIsFull(storage_, i)) {
                try {
                    Ops::emptyBucket(storage_ + i);
                } catch (...) {}
            }
        }

        size_ = 0;
    }

    bool count(const Key & key) const
    {
        ssize_t bucket = this->find_full_bucket(key);
        return bucket != NO_BUCKET;
    }

    void destroy()
    {
        ssize_t cp = capacity();

        // Run destructors
        for (ssize_t i = -1;  i < cp;  ++i) {
            try {
                Ops::destroyBucket(storage_ + i);
            } catch (...) {}
        }
        
        // Destroy the underlying memory
        storage_.destroy();
        size_ = 0;
    }

    void reserve(size_t new_capacity)
    {
        if (new_capacity <= capacity()) return;

        if (new_capacity < capacity() * 2)
            new_capacity = capacity() * 2;

        Lightweight_Hash_Base new_me(*this, new_capacity);
        swap(new_me);
    }

    void dump(std::ostream & stream) const
    {
        using namespace std;
        stream << "Lightweight_Hash: size " << size_ << " capacity "
               << capacity() << endl;
        for (ssize_t i = -1;  i < capacity();  ++i) {
            stream << "  bucket " << i << ": hash "
                   << Ops::hashKey(storage_[i], capacity(), storage_)
                   << " bucket " << storage_[i] << endl;
        }
    }

    bool needs_expansion() const
    {
        return (size_ >= 3 * capacity() / 4);
    }

protected:
    Storage storage_;
    int size_;

    std::pair<ssize_t, bool>
    find_or_insert(const Bucket & toInsert)
    {
        using namespace std;
        //cerr << "find_or_insert " << toInsert << endl;
        Key key = Ops::getKey(toInsert);
        //cerr << "key = " << key << endl;
        //cerr << "Ops::hashKey() = " << Ops::hashKey(key, capacity(), storage_)
        //     << endl;

        ssize_t bucket = find_bucket(key);

        //cerr << "found bucket " << bucket << endl;

        if (bucket != NO_BUCKET && Ops::bucketIsFull(storage_, bucket))
            return std::make_pair(bucket, false);
        return std::make_pair(insert_new(bucket, toInsert), true);
    }

    ssize_t must_insert(const Bucket & toInsert)
    {
        using namespace std;
        //cerr << "must_insert " << toInsert << endl;
        Key key = Ops::getKey(toInsert);
        ssize_t bucket = find_bucket(key);
        if (bucket != NO_BUCKET && Ops::bucketIsFull(storage_, bucket))
            throw ML::Exception("must_insert of value already there");
        return insert_new(bucket, toInsert);
    }

    enum {
        GUARD_BUCKET = -1,
        NO_BUCKET = -2
    };

public:
    //static uint64_t numCalls, numHops;

protected:
    //__attribute__((__noinline__))
    ssize_t find_bucket(const Key & key) const
    {
        using namespace std;

        if (Ops::isGuardValue(key)) {
            //cerr << "is guard value" << endl;
            return GUARD_BUCKET;
        }

        ssize_t cap = capacity();

        if (cap == 0) return NO_BUCKET;
        ssize_t bucket = Ops::hashKey(key, cap, storage_);

        //using namespace std;
        //cerr << "find_bucket: key " << key << " bucket " << bucket
        //     << " capcity " << capacity()
        //     << endl;

        bool wrapped = false;
        ssize_t i;
        //ssize_t hops = 0;
        for (i = bucket;
             Ops::bucketIsFull(storage_, i) && (i != bucket || !wrapped);
             /* no inc */ /*++hops*/) {
            if (Ops::bucketHasKey(storage_[i], key)) {
                //ML::atomic_inc(numCalls);
                //ML::atomic_add(numHops, hops);
                return i;
            }
            ++i;
            // NOTE: we don't wrap around the guard bucket.  It can't be used.
            if (i == cap) { i = 0;  wrapped = true; }
        }

        ExcAssertNotEqual(i, GUARD_BUCKET);

        if (!Ops::bucketIsFull(storage_, i)) {
            //ML::atomic_inc(numCalls);
            //ML::atomic_add(numHops, hops);
            return i;
        }

        // No bucket found; will need to be expanded
        if (size_ != cap) {
            dump(std::cerr);
            throw Exception("find_bucket: inconsistency");
        }
        return NO_BUCKET;
    }

    ssize_t find_full_bucket(const Key & key) const
    {
        ssize_t bucket = find_bucket(key);
        if (bucket == NO_BUCKET || !Ops::bucketIsFull(storage_, bucket)) return NO_BUCKET;
        if (!Ops::bucketHasKey(storage_[bucket], key)) {
#if 0
            using namespace std;
            dump(cerr);
            cerr << "bucket = " << bucket << endl;
            cerr << "hashed = " << hashed << endl;
            cerr << "key = " << key << endl;
            cerr << "storage_[bucket].first = " << storage_[bucket].first << endl;
#endif
            throw Exception("find_full_bucket didn't return correct key");
        }
        
        return bucket;
    }

    //__attribute__((__noinline__))
    ssize_t insert_new(ssize_t bucket, const Bucket & toInsert) 
    {
        //using namespace std;
        //cerr << "insert_new " << bucket << " " << toInsert << endl;

        Key key = Ops::getKey(toInsert);
        if (bucket == GUARD_BUCKET) {
            ExcAssert(Ops::isGuardValue(key));

            if (capacity() != 0) {
                Ops::fillBucket(storage_, bucket, toInsert);
                //using namespace std;
                //cerr << "incrementing size from " << size_ << endl;
                ++size_;
                
                return bucket;
            }
        }
        else {
            ExcAssert(!Ops::isGuardValue(key));
        }
        
        if (needs_expansion()) {
            // expand
            reserve(std::max<size_t>(4, capacity() * 2));
            bucket = find_bucket(key);
            if (bucket == NO_BUCKET || Ops::bucketIsFull(storage_, bucket))
                throw Exception("logic error: bucket appeared after reserve");
        }

        Ops::fillBucket(storage_, bucket, toInsert);
        //using namespace std;
        //cerr << "incrementing size from " << size_ << endl;
        ++size_;

        return bucket;
    }

    ssize_t advance_to_valid(ssize_t index) const
    {
        using namespace std;
        //cerr << "advancing to valid from index " << index << endl;

        if (index < -1 || index >= capacity()) {
            //dump(std::cerr);
            std::cerr << "index = " << index << std::endl;
            throw Exception("advance_to_valid: already at end");
        }

        ssize_t cap = capacity();

        // Scan through until we find a valid bucket
        while (index < cap && !Ops::bucketIsFull(storage_, index))
            ++index;

        //cerr << "advancing to valid: final index " << index << endl;

        return index;
    }

    ssize_t backup_to_valid(ssize_t index) const
    {
        if (index < -1 || index >= capacity())
            throw Exception("backup_to_valid: already outside range");
        
        // Scan through until we find a valid bucket
        while (index >= -1 && !Ops::bucketIsFull(storage_, index))
            --index;
        
        if (index < -1)
            throw Exception("backup_to_valid: none found");

        return index;
    }

    const Bucket & dereference(ssize_t bucket) const
    {
        if (bucket < -1 || bucket > capacity())
            throw Exception("dereferencing invalid iterator");
        if (!Ops::bucketIsFull(storage_, bucket)) {
            using namespace std;
            cerr << "bucket = " << bucket << endl;
            dump(cerr);
            throw Exception("dereferencing invalid iterator bucket");
        }
        return this->storage_[bucket];
    }
};

#if 0
template<class Key, class Bucket, class Ops, class Storage>
uint64_t Lightweight_Hash_Base<Key, Bucket, Ops, Storage>::numCalls = 0;
template<class Key, class Bucket, class Ops, class Storage>
uint64_t Lightweight_Hash_Base<Key, Bucket, Ops, Storage>::numHops = 0;
#endif


/*****************************************************************************/
/* LIGHTWEIGHT HASH MAP                                                      */
/*****************************************************************************/

template<typename Key, typename Value, typename Hash = std::hash<Key>,
         typename Bucket = std::pair<Key, Value> >
struct PairOps {
    static void initEmptyBucket(Bucket * bucket)
    {
        new (bucket) Bucket();
    }

    static void initBucket(Bucket * bucket, const Bucket & value)
    {
        new (bucket) Bucket(value);
    }

    template<typename Storage>
    static void fillBucket(Storage & storage, ssize_t index, const Bucket & value)
    {
        using namespace std;
        //cerr << "index = " << index << endl;
        Bucket * bucket = storage + index;
        //cerr << "bucket = " << bucket << endl;
        bucket->second = value.second;
        ML::memory_barrier();
        bucket->first = value.first;
        if (index == -1)
            storage.setGuardBucketIsOccupied();
        //*bucket = value;
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
    
    template<typename Storage>
    static bool bucketIsFull(const Storage & storage, ssize_t index)
    {
        if (index == -1)
            return storage.guardBucketIsFull();
        return storage[index].first;
    }

    static bool isGuardValue(Key key)
    {
        return key == 0;
    }

    static bool bucketHasKey(Bucket bucket, Key key)
    {
        return bucket.first == key;
    }

    static Key getKey(Bucket bucket)
    {
        return bucket.first;
    }

    template<class Storage>
    static size_t hashKey(Bucket bucket, ssize_t capacity,
                          const Storage & storage)
    {
        return hashKey(getKey(bucket), capacity, storage);
    }

    template<class Storage>
    static size_t hashKey(Key key, ssize_t capacity,
                          const Storage & storage)
    {
        return Hash()(key) % capacity;
    }

    static size_t hashKey(Key key, ssize_t capacity,
                          const LogMemStorage<Bucket> & storage)
    {
        uint64_t mask = (1ULL << ((storage.bits_ - 1))) - 1;
        return Hash()(key) & mask;
    }
 };

template<typename Key,
         typename Value,
         class Bucket = std::pair<Key, Value>,
         class ConstKeyBucket = std::pair<const Key, Value>,
         class Ops = PairOps<Key, Value>,
         class Storage = LogMemStorage<Bucket> >
struct Lightweight_Hash
    : public Lightweight_Hash_Base<Key, Bucket, Ops, Storage> {

    typedef Lightweight_Hash_Iterator<Key, const Value, const Lightweight_Hash,
                                      const Bucket>
    const_iterator;
    typedef Lightweight_Hash_Iterator<Key, Value, Lightweight_Hash,
                                      ConstKeyBucket> iterator;

    typedef Lightweight_Hash_Base<Key, Bucket, Ops, Storage> Base;

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

    Lightweight_Hash(Lightweight_Hash && other)
        : Base(other)
    {
    }

    Lightweight_Hash & operator = (const Lightweight_Hash & other)
    {
        Lightweight_Hash new_me(other);
        swap(new_me);
        return *this;
    }

    Lightweight_Hash & operator = (Lightweight_Hash && other)
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
    using Base::NO_BUCKET;

    iterator begin()
    {
        if (empty()) return end();
        return iterator(this, -1);
    }

    iterator end()
    {
        return iterator(this, this->capacity());
    }

    const_iterator begin() const
    {
        if (empty()) return end();
        return const_iterator(this, -1);
    }

    const_iterator end() const
    {
        return const_iterator(this, this->capacity());
    }

    iterator find(const Key & key)
    {
        ssize_t bucket = this->find_full_bucket(key);
        if (bucket == NO_BUCKET) return end();
        return iterator(this, bucket);
    }

    const_iterator find(const Key & key) const
    {
        ssize_t bucket = this->find_full_bucket(key);
        if (bucket == NO_BUCKET) return end();
        return const_iterator(this, bucket);
    }

    Value & operator [] (const Key & key)
    {
        ssize_t bucket = this->find_or_insert(Bucket(key, Value())).first;
        assert(this->storage_[bucket].first == key);
        return this->storage_[bucket].second;
    }

    std::pair<iterator, bool>
    insert(const Bucket & val)
    {
        std::pair<ssize_t, bool> r = this->find_or_insert(val);
        try {
            return make_pair(iterator(this, r.first), r.second);
        } catch (...) {
            using namespace std;

            cerr << "r.first = " << r.first << endl;
            cerr << "r.second = " << r.second << endl;
            cerr << "size = " << size() << endl;
            cerr << "capacity = " << capacity() << endl;
            cerr << "val = (" << val.first << "," << val.second << ")"
                 << endl;
            ssize_t cap = capacity();
            auto key = Ops::getKey(val);
            ssize_t bucket = Ops::hashKey(key, cap, this->storage_);
            cerr << "bucket = " << bucket << endl;

            throw;
        }
    }

    using Base::reserve;

private:
    template<typename K, typename V, class H, class CB>
    friend class Lightweight_Hash_Iterator;

    using Base::dereference;
    ConstKeyBucket & dereference(ssize_t bucket)
    {
        if (bucket < 0 || bucket > this->capacity())
            throw Exception("dereferencing invalid iterator");
        if (!this->storage_[bucket].first) {
            using namespace std;
            cerr << "bucket = " << bucket << endl;
            dump(cerr);
            throw Exception("dereferencing invalid iterator bucket");
        }
        return reinterpret_cast<ConstKeyBucket &>(this->storage_[bucket]);
    }

public:
    void dump(std::ostream & stream) const
    {
        using namespace std;
        stream << "Lightweight_Hash: size " << this->size_ << " capacity "
               << this->capacity() << endl;
        for (ssize_t i = -1;  i < this->capacity();  ++i) {
            stream << "  bucket " << i << ": hash "
                   << Ops::hashKey(this->storage_[i], this->capacity(),
                                   this->storage_)
                   << " key " << this->storage_[i].first;
            if (this->storage_[i].first)
                stream << " value " << this->storage_[i].second;
            stream << endl;
        }
    }
};


/*****************************************************************************/
/* LIGHTWEIGHT HASH SET                                                      */
/*****************************************************************************/

template<typename Key, typename Hash>
struct ScalarOps {
    typedef Key Bucket;

    static const Key guard;

    static void initEmptyBucket(Bucket * bucket)
    {
        new (bucket) Bucket(guard);
    }

    static void initBucket(Bucket * bucket, const Bucket & value)
    {
        new (bucket) Bucket(value);
    }

    template<typename Storage>
    static void fillBucket(Storage & storage, ssize_t index, const Bucket & value)
    {
        Bucket * bucket = storage + index;
        *bucket = value;
        if (index == -1)
            storage.setGuardBucketIsOccupied();
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
    
    template<typename Storage>
    static bool bucketIsFull(const Storage & storage, ssize_t index)
    {
        if (index == -1)
            return storage.guardBucketIsFull();
        return storage[index] != guard;
    }

    static bool isGuardValue(Key key)
    {
        return key == guard;
    }

    static bool bucketHasKey(Bucket bucket, Key key)
    {
        return bucket == key;
    }

    static Key getKey(const Bucket & bucket)
    {
        return bucket;
    }

    template<class Storage>
    static size_t hashKey(Key key, ssize_t capacity, const Storage & storage)
    {
        return Hash()(key) % capacity;
    }

    static size_t hashKey(Key key, ssize_t capacity,
                          const LogMemStorage<Bucket> & storage)
    {
        uint64_t mask = (1ULL << ((storage.bits_ - 1))) - 1;
        return Hash()(key) & mask;
    }
};

template<typename Key, typename Hash>
const Key
ScalarOps<Key, Hash>::guard(-2 /*NO_BUCKET*/);

template<typename Key, class Hash = std::hash<Key>,
         class Bucket = Key,
         class Ops = ScalarOps<Key, Hash>,
         class Storage = LogMemStorage<Bucket> >
struct Lightweight_Hash_Set
    : public Lightweight_Hash_Base<Key, Bucket, Ops, Storage> {

    typedef Lightweight_Hash_Iterator<Key, const Key, const Lightweight_Hash_Set,
                                      const Bucket>
    const_iterator;
    typedef const_iterator iterator;

    typedef Lightweight_Hash_Base<Key, Bucket, Ops, Storage> Base;

    Lightweight_Hash_Set()
    {
    }

    Lightweight_Hash_Set(const std::initializer_list<Key> & init)
        : Base(init.begin(), init.end(), init.size())
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

    Lightweight_Hash_Set(Lightweight_Hash_Set && other)
        : Base(other)
    {
    }

    Lightweight_Hash_Set & operator = (const Lightweight_Hash_Set & other)
    {
        Lightweight_Hash_Set new_me(other);
        swap(new_me);
        return *this;
    }

    Lightweight_Hash_Set & operator = (Lightweight_Hash_Set && other)
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
    using Base::NO_BUCKET;

    const_iterator begin() const
    {
        if (empty()) return end();
        return const_iterator(this, -1);
    }

    const_iterator end() const
    {
        return const_iterator(this, this->capacity());
    }

    const_iterator find(const Key & key) const
    {
        ssize_t bucket = this->find_full_bucket(key);
        if (bucket == NO_BUCKET) return end();
        return const_iterator(this, bucket);
    }

    std::pair<const_iterator, bool>
    insert(const Key & val)
    {
        std::pair<ssize_t, bool> r = this->find_or_insert(val);
        return make_pair(const_iterator(this, r.first), r.second);
    }

    template<typename Iterator>
    size_t insert(Iterator first, Iterator last)
    {
        size_t result = 0;
        for (; first != last;  ++first)
            result += insert(*first).second;
        return result;
    }

    using Base::count;
    using Base::reserve;

private:
    template<typename K, typename V, class H, class CB>
    friend class Lightweight_Hash_Iterator;
};


} // file scope


#endif /* __jml__utils__lightweight_hash_h__ */

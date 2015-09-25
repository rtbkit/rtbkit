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
        if (index == hash->capacity())
            throw Exception("increment past the end");
        ++index;
        if (index != hash->capacity()) index = hash->advance_to_valid(index);
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
/* MEM STORAGE                                                               */
/*****************************************************************************/

template<typename Bucket, typename Allocator = std::allocator<Bucket> >
struct MemStorage {
    MemStorage()
        : capacity_(0), vals_(0)
    {
    }

    MemStorage(size_t capacity)
        : capacity_(0), vals_(0)
    {
        reserve(capacity);
    }

    MemStorage(MemStorage && other)
        : capacity_(other.capacity_), vals_(other.vals_)
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

    void swap(MemStorage & other)
    {
        std::swap(capacity_, other.capacity_);
        std::swap(vals_, other.vals_);
    }

    size_t capacity() const JML_PURE_FN { return capacity_; }
    
    void reserve(size_t newCapacity)
    {
        if (vals_)
            throw ML::Exception("can't double initialize storage");
            
        vals_ = allocator.allocate(newCapacity);
        capacity_ = newCapacity;
    }

    void destroy()
    {
        if (!vals_) return;
        try {
            allocator.deallocate(vals_, capacity_);
        } catch (...) {}
        vals_ = 0;
        capacity_ = 0;
    }

    Bucket * operator + (size_t index)
    {
        //ExcAssert(vals_);
        //ExcAssertLess(index, capacity_);
        return vals_ + index;
    }
    
    Bucket & operator [] (size_t index)
    {
        //ExcAssert(vals_);
        //ExcAssertLess(index, capacity_);
        return vals_[index];
    }

    const Bucket & operator [] (size_t index) const
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
        : vals_(0), bits_(0)
    {
    }

    LogMemStorage(size_t capacity)
        : vals_(0), bits_(0)
    {
        reserve(capacity);
    }

    LogMemStorage(LogMemStorage && other)
        : bits_(other.bits_), vals_(other.vals_)
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

    void swap(LogMemStorage & other)
    {
        std::swap(bits_, other.bits_);
        std::swap(vals_, other.vals_);
    }

    size_t capacity() const JML_PURE_FN
    {
        return size_t(bits_ != 0) * (1ULL << (bits_ - 1));
    }
    
    void reserve(size_t newCapacity)
    {
        if (vals_)
            throw ML::Exception("can't double initialize storage");
        
        if (newCapacity == 0) return;

        bits_ = ML::highest_bit((newCapacity - 1), -1) + 2;
        vals_ = allocator.allocate(capacity());

        ExcAssertGreaterEqual(capacity(), newCapacity);
    }

    void destroy()
    {
        if (!vals_) return;
        try {
            allocator.deallocate(vals_, capacity());
        } catch (...) {}
        vals_ = 0;
        bits_ = 0;
    }

    Bucket * operator + (size_t index)
    {
        //ExcAssert(vals_);
        //ExcAssertLess(index, capacity_);
        return vals_ + index;
    }
    
    Bucket & operator [] (size_t index)
    {
        //ExcAssert(vals_);
        //ExcAssertLess(index, capacity_);
        return vals_[index];
    }

    const Bucket & operator [] (size_t index) const
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

        size_t cp = this->capacity();
        if (cp == 0) return;

        for (unsigned i = 0;  i < cp;  ++i)
            Ops::initEmptyBucket(storage_ + i);

        for (; first != last;  ++first)
            this->find_or_insert(*first);
    }

    Lightweight_Hash_Base(const Lightweight_Hash_Base & other,
                          size_t capacity)
        : storage_(capacity), size_(0)
    {
        size_t cp = this->capacity();

        for (unsigned i = 0;  i < cp;  ++i)
            Ops::initEmptyBucket(storage_ + i);

        size_t ocp = other.capacity();
        for (unsigned i = 0;  i < ocp;  ++i)
            if (Ops::bucketIsFull(other.storage_[i]))
                must_insert(other.storage_[i]);
    }

    Lightweight_Hash_Base(const Lightweight_Hash_Base & other)
        : storage_(other.capacity()), size_(other.size_)
    {
        if (capacity() == 0) return;

        ExcAssertEqual(capacity(), other.capacity());

        for (unsigned i = 0;  i < capacity();  ++i) {
            if (Ops::bucketIsFull(other.storage_[i]))
                Ops::initBucket(storage_ + i, other.storage_[i]);
            else Ops::initEmptyBucket(storage_ + i);
        }
    }

    Lightweight_Hash_Base(Lightweight_Hash_Base && other)
        : storage_(other.storage_), size_(other.size_)
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
    size_t capacity() const { return storage_.capacity(); }

    void clear()
    {
        size_t cp = capacity();

        // Empty buckets
        for (unsigned i = 0;  i < cp;  ++i) {
            if (Ops::bucketIsFull(storage_[i])) {
                try {
                    Ops::emptyBucket(storage_ + i);
                } catch (...) {}
            }
        }

        size_ = 0;
    }

    bool count(const Key & key) const
    {
        int bucket = this->find_full_bucket(key);
        return bucket != -1;
    }

    void destroy()
    {
        size_t cp = capacity();

        // Run destructors
        for (unsigned i = 0;  i < cp;  ++i) {
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
        for (unsigned i = 0;  i < capacity();  ++i) {
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

    std::pair<int, bool>
    find_or_insert(const Bucket & toInsert)
    {
        Key key = Ops::getKey(toInsert);
        int bucket = find_bucket(key);
        if (bucket != -1 && Ops::bucketIsFull(storage_[bucket]))
            return std::make_pair(bucket, false);
        return std::make_pair(insert_new(bucket, toInsert), true);
    }

    int must_insert(const Bucket & toInsert)
    {
        Key key = Ops::getKey(toInsert);
        int bucket = find_bucket(key);
        if (bucket != -1 && Ops::bucketIsFull(storage_[bucket]))
            throw ML::Exception("must_insert of value already there");
        return insert_new(bucket, toInsert);
    }

public:
    //static uint64_t numCalls, numHops;

protected:
    //__attribute__((__noinline__))
    int find_bucket(const Key & key) const
    {
        if (Ops::isGuardValue(key))
            throw Exception("searching for or inserting guard value");

        size_t cap = capacity();

        if (cap == 0) return -1;
        int bucket = Ops::hashKey(key, cap, storage_);

        //using namespace std;
        //cerr << "find_bucket: key " << key << " bucket " << bucket
        //     << " capcity " << capacity()
        //     << endl;

        bool wrapped = false;
        int i;
        //int hops = 0;
        for (i = bucket;
             Ops::bucketIsFull(storage_[i]) && (i != bucket || !wrapped);
             /* no inc */ /*++hops*/) {
            if (Ops::bucketHasKey(storage_[i], key)) {
                //ML::atomic_inc(numCalls);
                //ML::atomic_add(numHops, hops);
                return i;
            }
            ++i;
            if (i == cap) { i = 0;  wrapped = true; }
        }

        if (!Ops::bucketIsFull(storage_[i])) {
            //ML::atomic_inc(numCalls);
            //ML::atomic_add(numHops, hops);
            return i;
        }

        // No bucket found; will need to be expanded
        if (size_ != cap) {
            dump(std::cerr);
            throw Exception("find_bucket: inconsistency");
        }
        return -1;
    }

    int find_full_bucket(const Key & key) const
    {
        int bucket = find_bucket(key);
        if (bucket == -1 || !Ops::bucketIsFull(storage_[bucket])) return -1;
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
    int insert_new(int bucket, const Bucket & toInsert) 
    {
        Key key = Ops::getKey(toInsert);
        if (Ops::isGuardValue(key))
            throw Exception("searching for or inserting guard value");
        
        if (needs_expansion()) {
            // expand
            reserve(std::max<size_t>(4, capacity() * 2));
            bucket = find_bucket(key);
            if (bucket == -1 || Ops::bucketIsFull(storage_[bucket]))
                throw Exception("logic error: bucket appeared after reserve");
        }

        Ops::fillBucket(storage_ + bucket, toInsert);
        ++size_;

        return bucket;
    }

    int advance_to_valid(int index) const
    {
        if (index < 0 || index >= capacity()) {
            //dump(std::cerr);
            std::cerr << "index = " << index << std::endl;
            throw Exception("advance_to_valid: already at end");
        }

        size_t cap = capacity();

        // Scan through until we find a valid bucket
        while (index < cap && !Ops::bucketIsFull(storage_[index]))
            ++index;

        return index;
    }

    int backup_to_valid(int index) const
    {
        if (index < 0 || index >= capacity())
            throw Exception("backup_to_valid: already outside range");
        
        // Scan through until we find a valid bucket
        while (index >= 0 && !Ops::bucketIsFull(storage_[index]))
            --index;
        
        if (index < 0)
            throw Exception("backup_to_valid: none found");

        return index;
    }

    const Bucket & dereference(int bucket) const
    {
        if (bucket < 0 || bucket > capacity())
            throw Exception("dereferencing invalid iterator");
        if (!Ops::bucketIsFull(storage_[bucket])) {
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

    static void fillBucket(Bucket * bucket, const Bucket & value)
    {
        bucket->second = value.second;
        ML::memory_barrier();
        bucket->first = value.first;
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
    
    static bool bucketIsFull(Bucket bucket) JML_PURE_FN
    {
        return bucket.first;
    }

    static bool isGuardValue(Key key) JML_CONST_FN
    {
        return key == 0;
    }

    static bool bucketHasKey(Bucket bucket, Key key) JML_PURE_FN
    {
        return bucket.first == key;
    }

    static Key getKey(Bucket bucket) JML_CONST_FN
    {
        return bucket.first;
    }

    template<class Storage>
    static size_t hashKey(Bucket bucket, int capacity,
                          const Storage & storage)
    {
        return hashKey(getKey(bucket), capacity, storage);
    }

    template<class Storage>
    static size_t hashKey(Key key, int capacity,
                          const Storage & storage)
    {
        return Hash()(key) % capacity;
    }

    static size_t hashKey(Key key, int capacity,
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

    iterator begin()
    {
        if (empty()) return end();
        return iterator(this, 0);
    }

    iterator end()
    {
        return iterator(this, this->capacity());
    }

    const_iterator begin() const
    {
        if (empty()) return end();
        return const_iterator(this, 0);
    }

    const_iterator end() const
    {
        return const_iterator(this, this->capacity());
    }

    iterator find(const Key & key)
    {
        int bucket = this->find_full_bucket(key);
        if (bucket == -1) return end();
        return iterator(this, bucket);
    }

    const_iterator find(const Key & key) const
    {
        int bucket = this->find_full_bucket(key);
        if (bucket == -1) return end();
        return const_iterator(this, bucket);
    }

    Value & operator [] (const Key & key)
    {
        int bucket = this->find_or_insert(Bucket(key, Value())).first;
        assert(this->storage_[bucket].first == key);
        return this->storage_[bucket].second;
    }

    std::pair<iterator, bool>
    insert(const Bucket & val)
    {
        std::pair<int, bool> r = this->find_or_insert(val);
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
            size_t cap = capacity();
            auto key = Ops::getKey(val);
            int bucket = Ops::hashKey(key, cap, this->storage_);
            cerr << "bucket = " << bucket << endl;

            throw;
        }
    }

    using Base::reserve;

private:
    template<typename K, typename V, class H, class CB>
    friend class Lightweight_Hash_Iterator;

    using Base::dereference;
    ConstKeyBucket & dereference(int bucket)
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

    void dump(std::ostream & stream) const
    {
        using namespace std;
        stream << "Lightweight_Hash: size " << this->size_ << " capacity "
               << this->capacity() << endl;
        for (unsigned i = 0;  i < this->capacity();  ++i) {
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
    
    static bool bucketIsFull(Bucket bucket)
    {
        return bucket != guard;
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
    static size_t hashKey(Key key, int capacity, const Storage & storage)
    {
        return Hash()(key) % capacity;
    }

    static size_t hashKey(Key key, int capacity,
                          const LogMemStorage<Bucket> & storage)
    {
        uint64_t mask = (1ULL << ((storage.bits_ - 1))) - 1;
        return Hash()(key) & mask;
    }
};

template<typename Key, typename Hash>
const Key
ScalarOps<Key, Hash>::guard(-1);

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

    const_iterator begin() const
    {
        if (empty()) return end();
        return const_iterator(this, 0);
    }

    const_iterator end() const
    {
        return const_iterator(this, this->capacity());
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

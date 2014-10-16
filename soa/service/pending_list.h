/* pending_list.h                                                  -*- C++ -*-
   Jeremy Barnes, 2 February 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   List of things that are pending; can be made persistent.
*/

#ifndef __router__pending_list_h__
#define __router__pending_list_h__

#include "timeout_map.h"
#include "leveldb/db.h"
#include "jml/utils/guard.h"

namespace Datacratic {

struct PendingPersistence {
    virtual ~PendingPersistence()
    {
    }

    virtual void put(const std::string & key, const std::string & value) = 0;

    virtual std::string pop(const std::string & key) = 0;

    virtual std::string
    get(const std::string & key) const = 0;

    virtual void erase(const std::string & key) = 0;

    typedef boost::function<void (std::string, std::string) > OnEntry;
    typedef boost::function<void (std::string, std::string) > OnError;

    virtual void scan(const OnEntry & fn,
                      const OnError & onError = OnError()) const = 0;
};

struct LeveldbPendingPersistence : public PendingPersistence {
    std::shared_ptr<leveldb::DB> db;

    void open(const std::string & filename)
    {
        leveldb::DB* db;
        leveldb::Options options;
        options.create_if_missing = true;
        leveldb::Status status
            = leveldb::DB::Open(options, filename, &db);
        this->db.reset(db);
        if (!status.ok()) {
            throw ML::Exception("Opening leveldb: " + status.ToString());
        }
    }

    void compact()
    {
        using namespace std;
        Date start = Date::now();
        db->CompactRange(0, 0);
        Date end = Date::now();
        cerr << "compact took " << end.secondsSince(start) << "s" << endl;
    }

    uint64_t getDbSize()
    {
        leveldb::Range range;

        const leveldb::Snapshot * snapshot
            = db->GetSnapshot();
        ML::Call_Guard guard([&] () { this->db->ReleaseSnapshot(snapshot); });
        
        leveldb::ReadOptions options;
        options.verify_checksums = false;
        options.snapshot = snapshot;

        // Now iterate over everything in the database
        std::auto_ptr<leveldb::Iterator> it
            (db->NewIterator(options));
        it->SeekToFirst();
        if (!it->status().ok()) {
            throw ML::Exception("leveldb seek to first: "
                                + it->status().ToString());
        }
        if (!it->Valid())
            return 0;
        std::string start = it->key().ToString(); // important (dangling)
        range.start = start;

        it->SeekToLast();
        if (!it->status().ok()) {
            throw ML::Exception("leveldb seek to last: "
                                + it->status().ToString());
        }
        if (!it->Valid())
            return 0;
        std::string end = it->key().ToString();
        range.limit = end;

        uint64_t size = 0;
        db->GetApproximateSizes(&range, 1, &size);
        return size;
    }

    virtual void put(const std::string & key, const std::string & value)
    {
        leveldb::WriteOptions options;
        leveldb::Status status = db->Put(options, key, value);
        if (!status.ok()) {
            throw ML::Exception("Writing to leveldb: " + status.ToString());
        }
    }

    virtual std::string pop(const std::string & key)
    {
        std::string result = get(key);
        erase(key);
        return result;
    }

    virtual std::string
    get(const std::string & key) const
    {
        leveldb::ReadOptions options;
        std::string value;
        leveldb::Status status = db->Get(options, key, &value);
        if (!status.ok()) {
            throw ML::Exception("Writing to leveldb: " + status.ToString());
        }
        return value;
    }

    virtual void erase(const std::string & key)
    {
        leveldb::WriteOptions options;
        leveldb::Status status = db->Delete(options, key);
        if (!status.ok()) {
            throw ML::Exception("Writing to leveldb: " + status.ToString());
        }
    }

    virtual void scan(const OnEntry & fn,
                      const OnError & onError) const
    {
        using namespace std;
        //cerr << "compacting" << endl;
        //db->CompactRange(0, 0);
        //cerr << "done compacting" << endl;

        leveldb::ReadOptions options;
        options.verify_checksums = true;

        // Now iterate over everything in the database
        std::auto_ptr<leveldb::Iterator> it
            (db->NewIterator(options));

        it->SeekToFirst();

        unsigned numScanned = 0;
        for (it->SeekToFirst();  it->Valid();  it->Next(), ++numScanned) {
            std::string key = it->key().ToString();
            std::string value = it->value().ToString();

            //cerr << "key = " << key << endl;
            //cerr << "value = " << value << endl;

            try {
                fn(key, value);
            } catch (const std::exception & exc) {
                cerr << "bad entry scanning leveldb store: "
                     << exc.what() << endl;
                if (onError)
                    onError(key, value);
                else throw ML::Exception("LevelDbPersistence::scan(): bad entry");
            }
        }

        using namespace std;
        cerr << "scanned " << numScanned << " entries" << endl;
    }
};

template<typename Key, typename Value>
struct PendingPersistenceT {
    typedef boost::function<std::string(const Key &)> StringifyKey;
    typedef boost::function<Key (const std::string &)> UnstringifyKey;
    typedef boost::function<std::string(const Value &)> StringifyValue;
    typedef boost::function<Value (const std::string &)> UnstringifyValue;

    StringifyKey stringifyKey;
    StringifyValue stringifyValue;
    UnstringifyKey unstringifyKey;
    UnstringifyValue unstringifyValue;

    std::shared_ptr<PendingPersistence> store;

    typedef boost::function<void (Key &, Value &) > OnEntry;
    typedef PendingPersistence::OnError OnError;

    PendingPersistenceT()
    {
    }

    PendingPersistenceT(std::shared_ptr<PendingPersistence> store)
        : store(store)
    {
    }

    void scan(const OnEntry & onEntry, const OnError & onError = OnError())
    {
        auto onEntry2 = [&] (const std::string & skey,
                             const std::string & svalue)
            {
                Key key = this->unstringifyKey(skey);
                Value value = this->unstringifyValue(svalue);
                onEntry(key, value);
            };

        if (!store) return;
        store->scan(onEntry2, onError);
    }

    void erase(const Key & key)
    {
        if (!store) return;
        store->erase(stringifyKey(key));
    }

    void put(const Key & key, const Value & value)
    {
        if (!store) return;
        store->put(stringifyKey(key), stringifyValue(value));
    }
};

struct IsPrefixPair {
    template<typename T1, typename T2>
    bool operator () (const std::pair<T1, T2> & p1,
                      const std::pair<T1, T2> & p2) const
    {
        return p1.first == p2.first;
    }
};


template<typename Key, typename Value>
struct PendingList {

    typedef PendingPersistenceT<Key, Value> Persistence;
    std::shared_ptr<Persistence> persistence;
    typedef boost::function<bool (Key & key, Value & value, Date & timeout)>
        AcceptEntry;

    void initFromStore(std::shared_ptr<Persistence> persistence,
                       AcceptEntry acceptEntry,
                       Date timeout)
    {
        timeouts.clear();

        this->persistence = persistence;

        auto onEntry = [&] (Key & key, Value & value)
            {
                try {
                    Date t = timeout;
                    if (acceptEntry && ! acceptEntry(key, value, t))
                        return;
                    this->timeouts.insert(key, value, t);
                } catch (const std::exception & exc) {
                    using namespace std;
                    cerr << "error reconstituting pending entry" << endl;
                }
            };
        
        std::vector<std::string> toDelete;

        auto onError = [&] (const std::string & key,
                            const std::string & value)
            {
                toDelete.push_back(key);
            };

        persistence->scan(onEntry, onError);

        using namespace std;
        cerr << "deleting " << toDelete.size() << " invalid entries"
             << endl;

        for (unsigned i = 0;  i < toDelete.size();  ++i) {
            persistence->store->erase(toDelete[i]);
        }
    }

    void initFromStore(std::shared_ptr<Persistence> persistence,
                       Date timeout)
    {
        return initFromStore(persistence, AcceptEntry(), timeout);
    }

    size_t size() const
    {
        return timeouts.size();
    }

    template<typename Callback>
    void expire(const Callback & callback, Date now = Date::now())
    {
        auto myCallback = [&] (const Key & key, Value & value) -> Date
            {
                Date newExpiry = callback(key, value);
                if (newExpiry == Date()) {
                    if (this->persistence)
                        this->persistence->erase(key);
                }
                else if (this->persistence)
                    this->persistence->put(key, value);
                return newExpiry;
            };
        
        timeouts.expire(myCallback, now);
    }

    void expire(Date now = Date::now())
    {
        auto myCallback = [&] (const Key & key, Value & value) -> Date
            {
                return Date();
            };
        
        expire(myCallback, now);
    }

    bool count(const Key & key) const
    {
        return timeouts.count(key);
    }

    Value get(const Key & key) const
    {
        return timeouts.get(key);
    }

    Value pop(const Key & key)
    {
        Value result = timeouts.get(key);
        erase(key);
        return result;
    }

    /** key is a partial key.  Returns the first key result that is in the
        map for which isPrefix(result, key) is true if it exists, or
        Key() if none exists.

        Useful, for example, for isPrefixPair();
    */
    template<typename IsPrefix>
    Key completePrefix(const Key & key, IsPrefix isPrefix)
    {
        //using namespace std;
        //cerr << "looking for " << key << endl;
        auto it = timeouts.nodes.lower_bound(key);
        if (it == timeouts.nodes.end()) {
            //cerr << "  *** lower bound at end" << endl;
            return Key();
        }
        //cerr << "  lower bound returned " << it->first << endl;
        if (isPrefix(it->first, key)) {
            //cerr << "  *** isPrefix(" << it->first << "," << key << ") returned true" << endl;
            return it->first;
        }
        auto it2 = boost::next(it);
        if (it2 == timeouts.nodes.end()) {
            //cerr << "  *** next at end" << endl;
            return Key();
        }

        //cerr << "  next after lower bound returned " << it2->first << endl;
        if (isPrefix(it2->first, key)) {
            //cerr << "  *** isPrefix(" << it2->first << "," << key << ") returned true" << endl;
            return it2->first;
        }
        
        //cerr << "  *** no match" << endl;
        return Key();
    }

#if 0
    // If the key exists, return it
    // Otherwise, return the next key after this one were it to be inserted
    Key nextKey(const Key & key) const
    {
        auto it = timeouts.nodes.lower_bound(key);
        if (it == timeouts.nodes.end())
            return Key();
        if (it->first == key) return key;
        auto it2 = boost::next(it);
        if (it2 == timeouts.nodes.end())
            return Key();
        
        return it->first;
    }
#endif

    bool erase(const Key & key)
    {
        bool result = timeouts.erase(key);
        if (result && persistence) 
            persistence->erase(key);
        return result;
    }

    void insert(const Key & key, const Value & value, Date timeout)
    {
        timeouts.insert(key, value, timeout);
        if (persistence) persistence->put(key, value);
    }

    void update(const Key & key, const Value & value)
    {
        timeouts.update(key, value);
        if (persistence) persistence->put(key, value);
    }

    void update(const Key & key, const Value && value)
    {
        timeouts.update(key, value);
        if (persistence) persistence->put(key, value);
    }

    typedef TimeoutMap<Key, Value> Timeouts;
    Timeouts timeouts;

    typedef typename Timeouts::const_iterator const_iterator;
    const_iterator begin() const { return timeouts.begin(); }
    const_iterator end() const { return timeouts.end(); }
    const_iterator lower_bound(const Key & key) const
    {
        return timeouts.lower_bound(key);
    }
    const_iterator find(const Key & key) const
    {
        return timeouts.find(key);
    }

};

} // namespace Datacratic


#endif /* __router__pending_list_h__ */


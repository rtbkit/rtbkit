/* rcu_protected_test.cc
   Jeremy Barnes, 12 April 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

*/

#include "soa/gc/rcu_protected.h"
#include "jml/utils/vector_utils.h"
#include <thread>

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>


using namespace std;
using namespace Datacratic;


struct Collection1 {
    Collection1()
        : entries(entriesLock)
    {
        entries.replace(new Entries());
    }

    bool addEntry(const std::string & key,
                  std::shared_ptr<std::string> value,
                  bool mustAdd)
    {
        GcLock::SharedGuard guard(entriesLock);

        for (;;) {
            auto oldEntries = entries();

            ExcAssert(entriesLock.isLockedShared());

            auto it = oldEntries->find(key);
            if (it != oldEntries->end())
                return false;

            std::auto_ptr<Entries> newEntries(new Entries(*oldEntries));
            (*newEntries)[key] = *value;

            if (entries.cmp_xchg(oldEntries, newEntries, true /* defer */)) {
                return true;
            }
            // RCU raced
        }
    }

    void recycleEntries()
    {
        GcLock::SharedGuard guard(entriesLock);

        for (;;) {
            auto oldEntries = entries();

            ExcAssert(entriesLock.isLockedShared());

            std::auto_ptr<Entries> newEntries(new Entries(*oldEntries));

            ExcAssertEqual(newEntries->size(), oldEntries->size());

            if (entries.cmp_xchg(oldEntries, newEntries, true /* defer */)) {
                return;
            }
            // RCU raced
        }
    }

    bool deleteEntry(const std::string & key)
    {
        GcLock::SharedGuard guard(entriesLock);

        for (;;) {
            auto oldEntries = entries();

            ExcAssert(entriesLock.isLockedShared());

            auto it = oldEntries->find(key);
            if (it == oldEntries->end())
                return false;

            std::auto_ptr<Entries> newEntries(new Entries(*oldEntries));

            ExcAssert(newEntries->count(key));

            //auto entry = (*newEntries)[key];
            newEntries->erase(key);

            if (entries.cmp_xchg(oldEntries, newEntries,
                                 true /* defer */)) {
                return true;
            }
            // RCU raced
        }
    }

    bool forEachEntry(const std::function<bool (std::string, std::string)> & fn)
    {
        GcLock::SharedGuard guard(entriesLock);

        auto es = entries();

        for (auto & e: *es) {

            ExcAssert(entriesLock.isLockedShared());

            if (!fn(e.first, e.second))
                return false;
        }
        return true;
    }

    typedef std::map<std::string, std::string> Entries;
    GcLock entriesLock;
    RcuProtected<Entries> entries;
};

BOOST_AUTO_TEST_CASE( test_one_writer_one_reader )
{
    Collection1 collection;
    
    volatile bool shutdown = false;

    for (unsigned i = 0;  i < 20;  ++i)
        collection.addEntry("item" + to_string(i),
                            std::make_shared<std::string>("hello"),
                            false /* mustAdd */);

    auto writerThread = [&] ()
        {
            while (!shutdown) {
                // Add a random entry
                collection.addEntry("item" + to_string(random() % 20),
                                    std::make_shared<std::string>("hello"),
                                    false /* mustAdd */);

                auto onEntry = [&] (string key, string value)
                {
                    ExcAssertEqual(value, "hello");
                    return true;
                };

                // Check that we can iterate them properly
                collection.forEachEntry(onEntry);

                // Try to purturb things a bit
                collection.recycleEntries();

                // Now delete a random entry
                collection.deleteEntry("item" + to_string(random() % 20));
            };

            cerr << "finished writer thread" << endl;
        };

    auto readerThread = [&] ()
        {
            while (!shutdown) {
                auto onEntry = [&] (string key, string value)
                {
                    ExcAssertEqual(value, "hello");
                    return true;
                };

                // Check that we can iterate them properly
                collection.forEachEntry(onEntry);
            };

            cerr << "finished reader thread" << endl;
        };

    std::thread reader(readerThread);
    std::thread writer(writerThread);

    ::sleep(5);

    shutdown = true;
    
    reader.join();
    writer.join();
}

#if 1
BOOST_AUTO_TEST_CASE( test_multithreaded_access )
{
    Collection1 collection;
    
    volatile bool shutdown = false;

    for (unsigned i = 0;  i < 20;  ++i)
        collection.addEntry("item" + to_string(i),
                            std::make_shared<std::string>("hello"),
                            false /* mustAdd */);

    auto workerThread = [&] ()
        {
            while (!shutdown) {
                // Add a random entry
                collection.addEntry("item" + to_string(random() % 20),
                                    std::make_shared<std::string>("hello"),
                                    false /* mustAdd */);

                auto onEntry = [&] (string key, string value)
                {
                    ExcAssertEqual(value, "hello");
                    return true;
                };

                // Check that we can iterate them properly
                collection.forEachEntry(onEntry);

                // Try to purturb things a bit
                collection.recycleEntries();

                // Now delete a random entry
                collection.deleteEntry("item" + to_string(random() % 20));
            };

            cerr << "finished work thread" << endl;
        };

    int numThreads = 10;

    std::vector<std::thread> threads;
    for (unsigned i = 0;  i < numThreads;  ++i) {
        threads.emplace_back(workerThread);
    }

    ::sleep(5);

    shutdown = true;
    
    for (auto & t: threads)
        t.join();
}
#endif

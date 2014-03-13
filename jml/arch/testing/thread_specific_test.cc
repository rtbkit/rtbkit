/** thread_specitic_test.cc                                 -*- C++ -*-
    Rémi Attab, 30 Jul 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Tests for the instanced TLS class.

*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "jml/arch/thread_specific.h"

#include <boost/test/unit_test.hpp>
#include <atomic>
#include <thread>
#include <cassert>
#include <array>

using namespace std;
using namespace ML;

struct Data
{
    Data() : init(true)
    {
        init = true;
        constructed++;
    }

    ~Data()
    {
        check();
        init = false;
        destructed++;
    }

    void check() { assert(init); }

    bool init;

    static size_t constructed;
    static size_t destructed;
    static void validate()
    {
        assert(constructed == destructed);
        constructed = destructed = 0;
    }
};

size_t Data::constructed = 0;
size_t Data::destructed = 0;

typedef ML::ThreadSpecificInstanceInfo<Data, Data> Tls;


BOOST_AUTO_TEST_CASE(sanityTest)
{
    {
        ML::ThreadSpecificInstanceInfo<Data, Data> data;
        data.get();
    }

    BOOST_CHECK_EQUAL(Data::constructed, Data::destructed);
}

BOOST_AUTO_TEST_CASE( test_single_thread )
{
    {
        Tls t1;
        t1.get()->check();
    }
    Data::validate();

    {
        Tls t1;
        {
            Tls t2;
            t2.get()->check();
        }
        {
            Tls t2;
            Tls t3;
            t2.get()->check();
            t3.get()->check();
        }
    }
    Data::validate();

    {
        unique_ptr<Tls> t1(new Tls());
        unique_ptr<Tls> t2(new Tls());
        t1.reset();
        t2.reset();
    }
    Data::validate();
}


struct TlsThread
{
    TlsThread(Tls& tls) : tls(tls), done(false)
    {
        th = thread([=] { this->run(); });
    }

    ~TlsThread()
    {
        done = true;
        th.join();
    }

    void run()
    {
        tls.get()->check();
        while(!done);
    }

    Tls& tls;
    atomic<bool> done;
    thread th;
};


BOOST_AUTO_TEST_CASE( test_multi_threads_simple )
{
    Tls tls;

    {
        TlsThread t1(tls);
    }
    Data::validate();

    {
        TlsThread t1(tls);
        TlsThread t2(tls);
    }
    Data::validate();

    {
        TlsThread t1(tls);
        {
            TlsThread t2(tls);
        }
        {
            TlsThread t2(tls);
            TlsThread t3(tls);
        }
    }
    Data::validate();

    {
        unique_ptr<TlsThread> t1(new TlsThread(tls));
        unique_ptr<TlsThread> t2(new TlsThread(tls));
        t1.reset();
        t2.reset();
    }
}

BOOST_AUTO_TEST_CASE( test_multi_instance )
{
    enum {
        Instances = 64,
        Threads = 3
    };

    array<Tls*, Instances> instances;

    for (auto& instance : instances) instance = new Tls();

    auto runThread = [&] {
        for (auto& instance : instances)
            instance->get()->check();
    };
    for (size_t i = 0; i < Threads; ++i)
        thread(runThread).join();

    for (Tls* instance : instances) delete instance;

    Data::validate();
}

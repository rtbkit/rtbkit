/* nprobe_test.cc
   Mathieu Stefani, 23 September 2013

   Test for the tracing system
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <tuple>
#include <iostream>
#include <random>

#include "soa/service/nprobe.h"
#include "jml/arch/timers.h"

namespace {
    struct rand_init {
        rand_init() {
            srand(time(NULL));
        }
    } init;
}

struct Widget {
    Widget()
    {
        boost::uuids::random_generator gen;
        const auto uuid = gen();
        id = boost::to_string(uuid);
    }

    std::string id;
};

 
struct JoinGuard {
    enum Action { Join, Detach };

    JoinGuard(std::thread &&thread, Action action = Join)
        : thread_ (std::move(thread))
        , action_ (action)
    {
    }

    ~JoinGuard()
    {
        /* Data-race free */
        if (thread_.joinable())
        {
            if (action_ == Join)
                thread_.join();
            else if (action_ == Detach)
                thread_.detach();
        }
    }


private:
    std::thread thread_;
    Action action_;
};

std::tuple<const char *, std::string, int> do_probe(const Widget &w)
{
    return std::make_tuple("Widget", w.id, 1);
}

static void doSomethingWithWidget(const Widget &w)
{
    Datacratic::Trace<Widget>(w, "doSomethingWithWidget");

    for (;;) {
        if (((rand() << 8) % 675) == 450)
            break;
    }

}

BOOST_AUTO_TEST_CASE( basic_test )
{
    Widget widget;

    enum { 
        Traces = 50,
        Threads = 1
    };

    auto doTrace = [&]() {
        Datacratic::Trace<Widget>(widget, "doTrace");
        while (rand() % 761 != 327) ;


        for (size_t i = 0; i < Threads; ++i) {
            std::thread thr(doSomethingWithWidget, std::cref(widget));
            JoinGuard guard(std::move(thr));
        }

    };

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.3, 0.8);
    for (size_t i = 0; i < Traces; ++i) {
        doTrace();
        ML::sleep(dis(gen));
    }

}

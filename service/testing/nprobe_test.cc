/* nprobe_test.cc
   Mathieu Stefani, 23 September 2013

   Tests for the tracing system
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
#include <chrono>
#include <atomic>

#include "soa/service/nprobe.h"
#include "jml/arch/timers.h"


using namespace Datacratic;

namespace {
    struct rand_init {
        rand_init() {
            srand(time(NULL));
        }
    } init;
}

struct Traceable {
    virtual std::string id() const = 0;
    virtual const char *name() const = 0;
};

template<typename T>
std::tuple<const char *, std::string, int>
do_probe(const T &object) {
    static_assert(!std::is_base_of<T, Traceable>::value,
                  "do_probe<T>: T must inherit Traceable");
    return make_tuple(object.name(), object.id(), 1);
}
 
/**
 * RAII classes to manage thread joinability. Might want to make it 
 * public
 */
struct JoinGuard {
    enum Action { Join, Detach };

    JoinGuard(std::thread &&thread, Action action = Join)
        : thread_ (std::move(thread))
        , action_ (action)
    {
    }

    ~JoinGuard()
    {
        /* Data-race free since we have exclusive access to the thread
         * (we moved it) 
         */
        if (thread_.joinable())
        {
            if (action_ == Join)
                thread_.join();
            else if (action_ == Detach)
                thread_.detach();
        }
    }

    JoinGuard(JoinGuard &&) = default;

private:
    std::thread thread_;
    Action action_;
};

struct JoinGroup {

    JoinGroup(JoinGuard::Action action = JoinGuard::Join, size_t size = 0) :
        action_ (action)
    {
        if (size != 0)
            guards.reserve(size);
    }

    void addThread(std::thread &&thread) {
        guards.emplace_back(std::move(thread), action_);
    }

private:
    JoinGuard::Action action_;
    std::vector<JoinGuard> guards;
};


struct Object : public Traceable {
    Object() : id_ { instance++ }
    { }

    int id_;

    const char *name() const { return "Object"; }
    std::string id() const { return std::to_string(id_); }

private:
    static int instance;
};

int Object::instance = 0;


BOOST_AUTO_TEST_CASE( test_simple_trace )
{
    std::cout << "=========================================" << std::endl
              << " test_simple_trace" << std::endl
              << "-----------------------------------------" << std::endl;

    enum { SleepTime = 400 };

    Object obj;

    auto sinkFn = [&](const ProbeCtx &ctx, const std::vector<Span>& vs) {
        const char *tag;
        std::string id;
        uint32_t sampling;

        std::tie(tag, id, sampling) = ctx;
        BOOST_CHECK_EQUAL(tag, obj.name());
        BOOST_CHECK_EQUAL(id, obj.id());
        BOOST_CHECK_EQUAL(sampling, 1);

        BOOST_CHECK_EQUAL(vs.size(), 1);

        const auto &span = vs[0];
        BOOST_CHECK_EQUAL(span.tag_, "test_simple_trace");
        BOOST_CHECK_EQUAL(span.id_, 1);
        BOOST_CHECK_EQUAL(span.pid_, -1);

        const auto ms = 
          std::chrono::duration_cast<std::chrono::milliseconds>(span.end_ - span.start_).count();
        BOOST_CHECK_GE(ms, SleepTime);
    };

    Datacratic::Trace<Object> trace_(obj, "test_simple_trace");
    trace_.set_sinkCb(sinkFn);

    ML::sleep(SleepTime / 1000.0);
}

BOOST_AUTO_TEST_CASE( test_multiple_traces )
{
    std::cout << "=========================================" << std::endl
              << " test_multiple_traces" << std::endl
              << "-----------------------------------------" << std::endl;

    enum {
        SleepTime = 50,
        Iterations = 10
    };

    Object obj;

    int total { 0 };

    auto sinkFn = [&](const ProbeCtx &ctx, const std::vector<Span> &vs) {
        BOOST_CHECK_EQUAL(vs.size(), 3);

        struct Expected {
            const char *tag;
            int id;
            int pid;

            double elapsed;
        } expected[] = {
           { "test_multiple_traces::lambda<doTrace>::nested", 3, 2, ((SleepTime / 1000.0) * 2) },
           { "test_multiple_traces::lambda<doTrace>", 2, 1, (SleepTime / 1000.0) },
           { "test_multiple_traces", 1, -1, 0 },
        };

        for (size_t i { 0 }; i < vs.size(); ++i) {
            BOOST_CHECK_EQUAL(vs[i].tag_, expected[i].tag);
            BOOST_CHECK_EQUAL(vs[i].id_, expected[i].id);
            BOOST_CHECK_EQUAL(vs[i].pid_, expected[i].pid);

            const auto ms =
                std::chrono::duration_cast<std::chrono::milliseconds>(
                        vs[i].end_ - vs[i].start_).count();
            BOOST_CHECK_GE(ms, expected[i].elapsed);
        }
        ++total;
    };

    for (size_t i { 0 }; i < Iterations; ++i) {

        TRACE_FN(obj, "test_multiple_traces", sinkFn);
        

        auto doTrace = [&]() {
            TRACE_FN(obj, "test_multiple_traces::lambda<doTrace>", sinkFn);

            ML::sleep(SleepTime / 1000.0);

            {
                TRACE_FN(obj, "test_multiple_traces::lambda<doTrace>::nested", sinkFn);

                ML::sleep((SleepTime / 1000.0) * 2);
            }
        };


        doTrace();
    }

    BOOST_CHECK_EQUAL(total, Iterations);
}

struct Widget : public Traceable {
    Widget()
    {
        boost::uuids::random_generator gen;
        const auto uuid = gen();
        id_ = boost::to_string(uuid);
    }

    const char *name() const { return "Widget"; }
    std::string id() const { return id_; }

    std::string id_;
};

BOOST_AUTO_TEST_CASE( test_multithreaded_traces )
{
    std::cout << "=========================================" << std::endl
              << " test_multithreaded_traces" << std::endl
              << "-----------------------------------------" << std::endl;

    Widget widget;

    enum { 
        Iterations = 5,
        Threads = 1
    };

    std::atomic<int> total { 0 };
    double sleepTime { 0.0 };

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(0.3, 0.8);
    bool check { true };

    auto sinkFn = [&](const ProbeCtx &ctx, const std::vector<Span> &vs) {
        if (!check)
            return;

        total.fetch_add(1);
        BOOST_CHECK_EQUAL(vs.size(), 2);

        const auto ms_1 =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                    vs[0].end_ - vs[0].start_).count();
        BOOST_CHECK_GE(ms_1, sleepTime);

        const auto ms_2 =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                    vs[1].end_ - vs[1].start_).count();
        BOOST_CHECK_GE(ms_2, sleepTime + 0.2);
    };

    auto doSomethingWithWidget = [&](const Widget &w)
    {
        TRACE_FN(w, "test_multithreaded_traces::doSomethingWithWidget", sinkFn);

        ML::sleep(0.2);

        auto doWidget = [&]() {
            TRACE_FN(w, "test_multithreaded_traces::doSomethingWithWidget::doWidget", sinkFn);

            sleepTime = dis(gen);
            ML::sleep(sleepTime);

        };

        doWidget();

    };

    for (size_t i { 0 }; i < Iterations; ++i) {
        TRACE(widget, "test_multithreaded_traces");

        while (rand() % 761 != 327) ;

        check = true;
        {
            JoinGroup group(JoinGuard::Join, Threads);
            for (size_t j { 0 }; j < Threads; ++j) {
                group.addThread(std::thread(doSomethingWithWidget, std::cref(widget)));
            }
        }
        check = false;

    }

    BOOST_CHECK_EQUAL(total, (Iterations * Threads));

}

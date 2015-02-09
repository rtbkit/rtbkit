/* scope_test.cc
   Mathieu Stefani, 28 November 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.
   
   Tests for the scope utils
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "soa/utils/scope.h"
#include "jml/arch/exception.h"

#include <boost/test/unit_test.hpp>

using namespace std;
using namespace Datacratic;

struct FailureException : public ML::Exception {
    FailureException(std::string message) : ML::Exception(std::move(message))
    { }
};

struct ScopeState {
    ScopeState()
    { reset(); }

    void reset() {
        exited = succeeded = failed = false;
    }

    void exit() { exited = true;}
    void fail() { failed = true; }
    void succeed() { succeeded = true; }

    bool exited;
    bool succeeded;
    bool failed;
};

BOOST_AUTO_TEST_CASE( macros_test )
{
    JML_TRACE_EXCEPTIONS(false);

    ScopeState state;
    {
        Scope_Exit(state.exit());
    }

    BOOST_CHECK_EQUAL(state.exited, true);
    state.reset();

    {
        Scope_Success(state.succeed());

        Scope_Failure(state.fail());
    }

    BOOST_CHECK_EQUAL(state.exited, false);
    BOOST_CHECK_EQUAL(state.succeeded, true);
    BOOST_CHECK_EQUAL(state.failed, false);

    state.reset();

    try {

        {
            Scope_Failure(state.fail());
            Scope_Success(state.succeed());
            throw FailureException("Panic");
        }

        BOOST_CHECK_EQUAL(state.failed, true);
        BOOST_CHECK_EQUAL(state.succeeded, false);
    } catch (const FailureException& e) { }

    state.reset();

    try {

        {
            Scope_Failure(state.fail());
            Scope_Success(state.succeed());
            Scope_Exit(state.exit());

            throw FailureException("Panic");
        }

        BOOST_CHECK_EQUAL(state.failed, true);
        BOOST_CHECK_EQUAL(state.succeeded, false);
        BOOST_CHECK_EQUAL(state.exited, true);
    } catch (const FailureException& e) { }
}

BOOST_AUTO_TEST_CASE ( functions_test )
{
    JML_TRACE_EXCEPTIONS(false);

    ScopeState state;
    {
        auto exit = ScopeExit([&]() noexcept { state.exit(); });
    }
    BOOST_CHECK_EQUAL(state.exited, true);

    state.reset();
    {
        auto exit = ScopeExit([&]() noexcept { state.exit(); });

        exit.clear();
    }
    BOOST_CHECK_EQUAL(state.exited, false);

    state.reset();
    std::string failMessage;
    {
        auto failure = ScopeFailure([&]() noexcept { state.fail(); });
        BOOST_CHECK_EQUAL(failure.ok(), true);

        fail(failure, [&] { failMessage = "panic"; }); 

        BOOST_CHECK_EQUAL(failure.ok(), false);
    }

    BOOST_CHECK_EQUAL(state.failed, true);
    BOOST_CHECK_EQUAL(failMessage, "panic");

}

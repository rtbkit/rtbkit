#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include <iostream>
#include <sstream>

#include "jml/arch/exception.h"

#include "soa/service/sink.h"


using namespace std;
using namespace Datacratic;

BOOST_AUTO_TEST_CASE( test_outputsink_istream )
{
    string received;
    bool closed(false);

    auto onData = [&] (string && data) {
        received += move(data);
        return true;
    };
    auto onClose = [&] () {
        closed = true;
    };

    CallbackOutputSink outputSink(onData, onClose);

    string data("I am sending data.");
    string expected("Iamsendingdata.");
    std::istringstream datastream(data);

    string testdata;
    while (datastream) {
        datastream >> outputSink;
    }

    BOOST_CHECK_EQUAL(received, expected);
}

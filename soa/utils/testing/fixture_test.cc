/** fixture_test.cc                                 -*- C++ -*-
    Rémi Attab, 02 May 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Small example/test on how to use the datacratic fixture.

*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "soa/utils/fixtures.h"

#include <boost/test/unit_test.hpp>
#include <fstream>
#include <iostream>

using namespace std;
using namespace Datacratic;


// I'm lazy...
string readLine(ifstream& ifs)
{
    string line;
    getline(ifs, line);
    return line;
}


/** Creates a magic fixture compatible with boost test for our test. Note that
    the name of the fixture should be globally unique (no two test files in the
    checked out repo should share the same name. This constraint can easily be
    met by using the filename as the name of the fixture.

    The purpose of this fixture is to create a file sandbox for each tests. This
    sandbox ensures that any created files will not be visible to any other
    running tests and that the test never has to do any cleanup. This is even
    true if the process crashes due to signals.
 */
DATACRATIC_FIXTURE(FixtureTest);


/** Simple test that reads and writes a file in it's sandbox */
BOOST_FIXTURE_TEST_CASE(basicTest, FixtureTest)
{
    // The fixture also provides a handy uniqueName() utility which generates a
    // unique name for the test that can be useful to create things like shm
    // files.
    const string filename = uniqueName();
    ofstream(filename) << "Test";

    ifstream ifs(filename);
    BOOST_CHECK_EQUAL(readLine(ifs), "Test");
}


// This is an input file for our test.
namespace { const string inputFile = "soa/utils/testing/test_input.txt"; }


/** Test that reads and input file that sits somewhere in the repo. */
BOOST_FIXTURE_TEST_CASE(inputFileTest, FixtureTest)
{
    // To access a file outside our sandbox just invoke resolvePath with your
    // input file.
    ifstream ifs(resolvePath(inputFile));
    BOOST_CHECK_EQUAL(readLine(ifs), "Hello World!");
}


/** The fixture is only a class so it can be used in a variety of interesting
    ways. As an example, you can even write your own fixture that subclasses
    this fixture.

    In this case we just want to run the test many times where each execution is
    isolated in its own sandbox.
*/
void doTest(int i)
{
    FixtureTest fixture;

    ofstream("thingy", ios::app) << to_string(i);

    ifstream ifs("thingy");
    BOOST_CHECK_EQUAL(readLine(ifs), to_string(i));
}

BOOST_AUTO_TEST_CASE(repeatedTest)
{
    for (int i = 0; i < 3; ++i) doTest(i);
}

/** fixtures.h                                 -*- C++ -*-
    Rémi Attab, 30 Apr 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Testing fixtures.
*/

#pragma once

#include <string>

namespace Datacratic {

/******************************************************************************/
/* TEST FOLDER FIXTURE                                                        */
/******************************************************************************/

/** Changes the current working dir to a test folder where files can be created
    and deleted without impacting the source tree.
 */
struct TestFolderFixture
{
    TestFolderFixture(const std::string& name);
    virtual ~TestFolderFixture();

    /** Returns a that should be added to a path to acccess input test files. */
    std::string resolvePath(const std::string& path) const;

    /** Returns a name that is unique per test and per user. Suitable for a mmap
        file name.
    */
    std::string uniqueName() const;

private:
    std::string name;
    std::string path;
    std::string oldPath;
    static int testCount;
};


/** Creates a fixture with the given name to be used with fixtures. The name
    given to the fixture should be unique and should usually just reflect the
    name of the file in which the test will reside.

    Quick example: let's say we have a test in test_awesome_component.cc

    DATACRATIC_FIXTURE(TestAwesomeComponent);
    BOOST_FIXTURE_TEST_CASE(myTest, TestAwesomeComponent) {...}

 */
#define DATACRATIC_FIXTURE(_name_)                      \
    struct _name_ : public TestFolderFixture {          \
        _name_() : TestFolderFixture(#_name_) {}        \
    }


} // namespace Datacratic


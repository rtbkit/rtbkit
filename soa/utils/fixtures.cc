/** fixtures.cc                                 -*- C++ -*-
    Rémi Attab, 30 Apr 2013
    Copyright (c) 2013 Datacratic.  All rights reserved.

    Testing fixtures.
*/


#include "fixtures.h"

#include <boost/filesystem.hpp>
#include <iostream>

using namespace std;
namespace fs = boost::filesystem;

namespace Datacratic {

/******************************************************************************/
/* TEST FOLDER FIXTURE                                                        */
/******************************************************************************/

namespace {

const string tmpDir = "./build/x86_64/tmp/";
const string prefixDir = "./../../../../";

};

int TestFolderFixture::testCount = 0;

TestFolderFixture::
TestFolderFixture(const string& name) :
    name(name)
{
    path = tmpDir + name + "_" + to_string(testCount++);

    if (fs::is_directory(path))
        fs::remove_all(path);
    fs::create_directories(path);

    oldPath = fs::current_path().string();
    fs::current_path(path);
}

TestFolderFixture::
~TestFolderFixture()
{
    fs::current_path(oldPath);
    fs::remove_all(path);
}

string
TestFolderFixture::
resolvePath(const string& path) const
{
    return prefixDir + path;
}

string
TestFolderFixture::
uniqueName() const
{
    return name + "_" + to_string(testCount) + "_" + to_string(getuid());
}


} // namespace Datacratic

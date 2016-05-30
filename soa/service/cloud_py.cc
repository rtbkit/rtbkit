/* cloud_py.cc
   Wolfgang Sourdeau, Feb 2nd, 2016
   Copyright (c) 2016  Datacratic.  All rights reserved.
   
   Python wrappers for libcloud
*/


#include <Python.h>
#include <string>
#include "platform/python/pointer_fix.h" //must come before boost python
#include <boost/python.hpp>
#include "s3.h"

using namespace std;
using namespace boost;
using namespace Datacratic;


namespace {

void defaultRegisterS3Bucket(const string & bucketName,
                             const string & keyId, const string &key)
{
    registerS3Bucket(bucketName, keyId, key);
}

void defaultRegisterS3Buckets(const string & keyId, const string &key)
{
    registerS3Buckets(keyId, key);
}

}


BOOST_PYTHON_MODULE(py_cloud) {
    python::object package = python::scope();
    package.attr("__path__") = "py_cloud";

    python::def("register_s3_bucket",
                defaultRegisterS3Bucket,
                python::args("bucketName", "keyId", "key"),
                "Register a specific bucket with the given key");

    python::def("register_s3_buckets",
                defaultRegisterS3Buckets,
                python::args("keyId", "key"),
                "Registers all available buckets with the given key");
}

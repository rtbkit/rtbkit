/**
 * mongo_temp_server_wrapping.cc
 * Mich, 2014-12-15
 * Copyright (c) 2014 Datacratic Inc.  All rights reserved.
 **/

#include <boost/python.hpp>
#include "../mongo_temporary_server.h"

using namespace std;
using namespace boost::python;
using namespace Datacratic;
using namespace Mongo;

struct MongoTemporaryServerPtr {

    shared_ptr<MongoTemporaryServer> mongoTmpServer;

    MongoTemporaryServerPtr(const std::string & uniquePath = "",
                            const int portNum = 28356) {
        try {
            MongoTemporaryServer srv(uniquePath, portNum);
        }
        catch (...) {
            cerr << "caught exception" << endl;
        }
    }

    void testConnection() {
        mongoTmpServer->testConnection();
    }
};

BOOST_PYTHON_MODULE(python_mongo_temp_server_wrapping) {
    class_<MongoTemporaryServerPtr>("MongoTemporaryServerPtr",
                                    init<std::string, int>())
        .def("test_connection", &MongoTemporaryServerPtr::testConnection);
}


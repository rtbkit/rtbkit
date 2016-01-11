/* filter_streams_py.cc
   Wolfgang Sourdeau
   Jeremy Barnes, Dec 8th 2015
   Copyright (c) 2015  Datacratic.  All rights reserved.
   
   Python wrappers for ML::filter_[oi]stream
*/


#include <Python.h>
#include <string>
#include "platform/python/pointer_fix.h" //must come before boost python
#include <boost/python.hpp>
#include "filter_streams.h"

using namespace std;
using namespace boost;


namespace {

void filter_ostream__write(ML::filter_ostream & stream,
                           const string & data)
{
    stream.write(data.c_str(), data.size());
}

std::string filter_istream__read0(ML::filter_istream & stream)
{
    return stream.readAll();
}

std::string filter_istream__read1(ML::filter_istream & stream, size_t nBytes)
{
    string result;

    char buffer[65536];
    size_t remaining = nBytes;
    while (stream && remaining > 0) {
        size_t toRead = sizeof(buffer);
        if (remaining < toRead) {
            toRead = remaining;
        }
        stream.read(buffer, toRead);
        size_t readSize = stream.gcount();
        result.append(buffer, readSize);
        remaining -= readSize;
    }

    return result;
}

std::string filter_istream__readline1(ML::filter_istream & stream,
                                      ssize_t nBytes)
{
    string line;

    if (stream) {
        getline(stream, line);
        line += "\n";
        if (nBytes > -1) {
            line.resize(nBytes);
        }
    }

    return line;
}

std::string filter_istream__readline0(ML::filter_istream & stream)
{
    return filter_istream__readline1(stream, -1);
}

} // file scope


BOOST_PYTHON_MODULE(filter_streams) {
    // BOOST_PYTHON_MODULE_INIT();

    python::class_<ML::filter_ostream,
                   std::shared_ptr<ML::filter_ostream>,
                   boost::noncopyable>
        ("filter_ostream", python::init<const std::string &>())
         .def("write", &filter_ostream__write)
         .def("close", &ML::filter_ostream::close);

    python::class_<ML::filter_istream,
                   std::shared_ptr<ML::filter_istream>,
                   boost::noncopyable>
        ("filter_istream", python::init<const std::string &>())
         .def("read", &filter_istream__read0)
         .def("read", &filter_istream__read1)
         .def("readline", &filter_istream__readline0)
         .def("readline", &filter_istream__readline1)
         .def("close", &ML::filter_istream::close);
}

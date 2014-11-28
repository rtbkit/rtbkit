/**
 * logger_metrics_test.cc
 * Mich, 2014-11-17
 * Copyright (c) 2014 Datacratic Inc. All rights reserved.
 *
 * Manual test for the logger metrics. Provide the proper json config and
 * run.
 **/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <boost/smart_ptr.hpp>
#include <boost/thread/barrier.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/regex.hpp>

#include "soa/logger/logger_metrics_interface.h"

using namespace Datacratic;
using namespace std;

BOOST_AUTO_TEST_CASE( test_logger_metrics ) {
    shared_ptr<ILoggerMetrics> logger =
        ILoggerMetrics::setup("metricsLogger", "lalmetrics", "test");

    logger->logMeta({"a", "b"}, "taratapom");
}

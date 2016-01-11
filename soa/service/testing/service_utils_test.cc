/** service_utils_test.cc                                 -*- C++ -*-
    Sirma Cagil Altay, 20 Oct 2015
    Copyright (c) 2015 Datacratic.  All rights reserved.
    
    Test dynamic library loading ability of ServiceProxyArguments
*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <string>
#include <stdlib.h>
#include <vector>

#include <boost/test/unit_test.hpp>
#include <boost/thread/thread.hpp>

#include "jml/utils/filter_streams.h"
#include "jml/utils/environment.h"
#include "soa/service/testing/dynamic_loading_test_table.h"

#include "soa/service/service_utils.h"

using namespace std;
using namespace ML;
using namespace Datacratic;

static const string buildOptions("build/x86_64/bin/libcustom_preload_1.so,"
                                 "soa/service/testing/libs-to-dynamically-load");
static const string envOptions("build/x86_64/bin/libcustom_preload_4");
static const string RtbkitPreload("RTBKIT_PRELOAD");

BOOST_AUTO_TEST_CASE( test_service_utils_preload )
{
    BOOST_REQUIRE_EQUAL(TEST::DynamicLoading::custom_lib_1,0);
    BOOST_REQUIRE_EQUAL(TEST::DynamicLoading::custom_lib_2,0);
    BOOST_REQUIRE_EQUAL(TEST::DynamicLoading::custom_lib_3,0);
    BOOST_REQUIRE_EQUAL(TEST::DynamicLoading::custom_lib_4,0);

    ServiceProxyArguments myService;
    myService.preload = buildOptions;
    setenv(RtbkitPreload.c_str(),envOptions.c_str(),1);
    myService.makeServiceProxies();

    BOOST_REQUIRE_EQUAL(TEST::DynamicLoading::custom_lib_1,1);
    BOOST_REQUIRE_EQUAL(TEST::DynamicLoading::custom_lib_2,1);
    BOOST_REQUIRE_EQUAL(TEST::DynamicLoading::custom_lib_3,1);
    BOOST_REQUIRE_EQUAL(TEST::DynamicLoading::custom_lib_4,1);
}

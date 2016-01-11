/** plugin_table_test.cc                                 -*- C++ -*-
    Sirma Cagil Altay, 22 Oct 2015
    Copyright (c) 2015 Datacratic.  All rights reserved.

    Tests for plugin table utilities

*/

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "rtbkit/common/plugin_table.h"
#include "rtbkit/common/testing/custom_base_plugin.h"

#include <boost/test/unit_test.hpp>
#include <thread>

using namespace std;
using namespace RTBKIT;
using namespace TEST;

BOOST_AUTO_TEST_CASE(naming_convention_test_1)
{
    auto f = PluginTable<TestPlugin::Factory>::instance().getPlugin("custom_1","plugin");

    int num = f()->getNum();

    BOOST_REQUIRE_EQUAL(num,1);
}

BOOST_AUTO_TEST_CASE(naming_convention_test_2)
{
    auto f = PluginTable<TestPlugin::Factory>::instance().getPlugin("custom_1_plugin.custom_1","plugin");

    int num = f()->getNum();

    BOOST_REQUIRE_EQUAL(num,1);
}

BOOST_AUTO_TEST_CASE(multi_thread_getPlugin_stress_test)
{
    const size_t totalThreads = 10;
    const size_t arrayGetPerThread = 1000;

    vector<thread> threads;
    vector<TestPlugin::Factory> result(totalThreads);

    auto threadFn = [&](unsigned int threadNum){
        for(int i=0; i<arrayGetPerThread; i++)
            result[threadNum]=
                PluginTable<TestPlugin::Factory>::instance().getPlugin("custom_1_plugin.custom_1","plugin");
    };

    for(size_t i=0; i<totalThreads; i++)
        threads.emplace_back(threadFn,i);

    for(auto & th: threads)
        th.join();

    for(size_t i=0; i<totalThreads; i++){
        int num = result[i]()->getNum();
        BOOST_REQUIRE_EQUAL(num,1);
    }
}

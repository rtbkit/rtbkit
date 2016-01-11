/** custom_preload_3.cc                                 -*- C++ -*-
    Sirma Cagil Altay, 20 Oct 2015
    Copyright (c) 2015 Datacratic.  All rights reserved.
    
    File to be dynamically loaded during service_util test
*/

#include "soa/service/testing/dynamic_loading_test_table.h"

namespace {

struct AtInit {
    AtInit()
    {
        TEST::DynamicLoading::custom_lib_3 = 1;
    }
} AtInit;

} // namespace anonymous

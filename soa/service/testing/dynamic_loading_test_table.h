/** dynamic_loading_test_table.h
    Sirma Cagil Altay, 20 Oct 2015
    Copyright (c) 2015 Datacratic.  All rights reserved.
    
    Class created for testing the dynamic library
    loading during service_util test
*/

#pragma once

namespace TEST{

class DynamicLoading {
public:
    static int custom_lib_1;
    static int custom_lib_2;
    static int custom_lib_3;
    static int custom_lib_4;
};

int DynamicLoading::custom_lib_1 = 0;
int DynamicLoading::custom_lib_2 = 0;
int DynamicLoading::custom_lib_3 = 0;
int DynamicLoading::custom_lib_4 = 0;

} // namespace TEST

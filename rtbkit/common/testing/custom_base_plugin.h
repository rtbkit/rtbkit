/** custom_base_plugin.h                                 -*- C++ -*-
    Sirma Cagil Altay, 22 Oct 2015
    Copyright (c) 2015 Datacratic.  All rights reserved.

    Base class for other plugins created just for test purposes
*/

#pragma once

#include <string>
#include <functional>
#include <memory>

#include "rtbkit/common/plugin_table.h"

namespace TEST {

class TestPlugin {

public:

    virtual int getNum() { return 0; }
   
    typedef std::function<TestPlugin * ()> Factory;

};

} // namespace TEST

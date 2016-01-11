/** custom_1_plugin.cc                                 -*- C++ -*-
    Sirma Cagil Altay, 22 Oct 2015
    Copyright (c) 2015 Datacratic.  All rights reserved.

    Plugin created just for test purposes
*/

#include "rtbkit/common/testing/custom_base_plugin.h"

namespace TEST {

class Plugin1 : public TestPlugin {
    
public:

    virtual int getNum() { return 1; }

    static const std::string name;
};

const std::string Plugin1::name = "custom_1";

} // namespace TEST

/******************************************************************************/
/* INIT PLUGIN1                                                               */
/******************************************************************************/

namespace {

struct AtInit {
    AtInit()
    {
        using namespace TEST;
        TestPlugin::Factory f = []() -> TestPlugin* {return new Plugin1();};
        RTBKIT::PluginTable<TestPlugin::Factory>::instance()
                            .registerPlugin(Plugin1::name,f); 
    }

} AtInit;

} // namespace anonymous

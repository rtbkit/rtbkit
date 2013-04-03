/* value_description.cc                                            -*- C++ -*-
   Jeremy Barnes, 29 March 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Code for description and introspection of values and structures.  Used
   to allow for automated formatters and parsers to be built.
*/


#include "value_description.h"
#include "jml/arch/demangle.h"


using namespace std;
using namespace ML;


namespace Datacratic {

void registerValueDescription(const std::type_info & type,
                              std::function<ValueDescription * ()> fn,
                              bool isDefault)
{
    auto desc = fn();

    cerr << "got " << ML::demangle(type.name())
         << " with description "
         << ML::type_name(*desc) << endl;

    delete desc;
}


} // namespace Datacratic

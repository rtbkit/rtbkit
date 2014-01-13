/* expand_variable.cc                           -*- C++ -*-
   Thomas Sanchez, 22 October 2013
   Copyright (c) 2013 mbr targeting GmbH.  All rights reserved.
*/

#include <boost/algorithm/string.hpp>
#include "rtbkit/common/expand_variable.h"

namespace RTBKIT {

ExpandVariable::ExpandVariable(const std::string& v, int begin, int end)
: variable_(v), startIndex_(begin), endIndex_(end)
{
    parseVariable();
}

void ExpandVariable::parseVariable()
{
    std::string variable = variable_;

    { // Do some filtering if required
        boost::algorithm::split(filters_,
                                variable,
                                boost::algorithm::is_any_of("#"),
                                boost::algorithm::token_compress_on);

        variable = std::move(*filters_.begin());
        filters_.erase(filters_.begin());
    }

    { // finally we extract the full path of the value we want to fetch
        boost::algorithm::split(path_,
                                variable,
                                boost::algorithm::is_any_of("."),
                                boost::algorithm::token_compress_on);
    }

    variable_ = std::move(variable);
}

} // namespace RTBKIT

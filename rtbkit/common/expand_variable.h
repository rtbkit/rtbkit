/* expand_variable.h                           -*- C++ -*-
   Thomas Sanchez, 22 October 2013
   Copyright (c) 2013 mbr targeting GmbH.  All rights reserved.
*/
#pragma once

#include <string>
#include <vector>

namespace RTBKIT {

class ExpandVariable
{
public:
    ExpandVariable(const std::string& variable, int begin, int end);

    const std::string& getVariable() const
    {
        return variable_;
    }

    bool operator<(const ExpandVariable& rhs) const
    {
        return variable_ < rhs.variable_;
    }

    const std::vector<std::string>& getPath() const
    {
        return path_;
    }

    const std::vector<std::string>& getFilters() const
    {
        return filters_;
    }

    std::pair<int, int> getReplaceLocation() const
    {
        return std::make_pair(startIndex_, endIndex_);
    }

private:

    void parseVariable();

    std::string variable_;
    std::vector<std::string> filters_;
    std::vector<std::string> path_;
    int startIndex_;
    int endIndex_;
};

} // namespace RTBKIT

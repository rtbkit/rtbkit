/* config_options.cc
   Jeremy Barnes, 15 March 2006
   Copyright (c) 2006 Jeremy Barnes  All rights reserved.
   $Source$

   Class to document the configuration options of a given class.
*/

#include "config_options.h"
#include "jml/utils/string_functions.h"

using namespace std;

namespace ML {

std::string Config_Option::print() const
{
    return format("%-35s %8s   %s",
                  (name + " [" + range + "]").c_str(), value.c_str(),
                  doc.c_str());
}

Config_Option
option(const std::string & name, const std::string & type,
       const std::string & value, const std::string & range,
       const std::string & doc)
{
    Config_Option result;
    result.name = name;
    result.type = type;
    result.value = value;
    result.range = range;
    result.doc = doc;
    return result;
}

void
Config_Options::
dump(std::ostream & stream, int indent, int cols) const
{
    string ind(indent, ' ');
    for (unsigned i = 0;  i < size();  ++i)
        stream << ind << at(i).print() << endl;
}

} // namespace ML

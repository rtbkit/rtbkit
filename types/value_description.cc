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

std::ostream & operator << (std::ostream & stream, ValueKind kind)
{
    switch (kind) {
    case ValueKind::ATOM: return stream << "ATOM";
    case ValueKind::INTEGER: return stream << "INTEGER";
    case ValueKind::FLOAT: return stream << "FLOAT";
    case ValueKind::BOOLEAN: return stream << "BOOLEAN";
    case ValueKind::STRING: return stream << "STRING";
    case ValueKind::ENUM: return stream << "ENUM";
    case ValueKind::OPTIONAL: return stream << "OPTIONAL";
    case ValueKind::ARRAY: return stream << "ARRAY";
    case ValueKind::STRUCTURE: return stream << "STRUCTURE";
    case ValueKind::TUPLE: return stream << "TUPLE";
    case ValueKind::VARIANT: return stream << "VARIANT";
    case ValueKind::MAP: return stream << "MAP";
    case ValueKind::ANY: return stream << "ANY";
    default:
        return stream << "ValueKind(" << to_string((int)kind) << ")";
    }
}

namespace {
    std::unordered_map<std::string, ValueDescription *> registry;
}

ValueDescription * ValueDescription::get(std::string const & name) {
    auto i = registry.find(name);
    return registry.end() != i ? i->second : 0;
}

void registerValueDescription(const std::type_info & type,
                              std::function<ValueDescription * ()> fn,
                              bool isDefault)
{
    auto desc = fn();

    /*
    cerr << "got " << ML::demangle(type.name())
         << " with description "
         << ML::type_name(*desc) << endl;
    */

    registry[desc->typeName] = desc;
}

void
ValueDescription::
convertAndCopy(const void * from,
               const ValueDescription & fromDesc,
               void * to) const
{
    StructuredJsonPrintingContext context;
    fromDesc.printJson(from, context);

    StructuredJsonParsingContext context2(context.output);
    parseJson(to, context2);
}

} // namespace Datacratic

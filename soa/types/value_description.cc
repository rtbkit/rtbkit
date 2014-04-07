/* value_description.cc                                            -*- C++ -*-
   Jeremy Barnes, 29 March 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Code for description and introspection of values and structures.  Used
   to allow for automated formatters and parsers to be built.
*/


#include <mutex>
#if 0
#include "jml/arch/demangle.h"
#endif
#include "jml/utils/exc_assert.h"
#include "value_description.h"

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
        return stream << "ValueKind(" << std::to_string((int)kind) << ")";
    }
}

namespace {
std::recursive_mutex registryMutex;
std::unordered_map<std::string, std::shared_ptr<const ValueDescription> > registry;
}

std::shared_ptr<const ValueDescription>
ValueDescription::
get(std::string const & name)
{
    std::unique_lock<std::recursive_mutex> guard(registryMutex);
    auto i = registry.find(name);
    return registry.end() != i ? i->second : nullptr;
}

std::shared_ptr<const ValueDescription>
ValueDescription::
get(const std::type_info & type)
{
    return get(type.name());
}

void registerValueDescription(const std::type_info & type,
                              std::function<ValueDescription * ()> fn,
                              bool isDefault)
{
    registerValueDescription(type, fn, [] (ValueDescription &) {}, isDefault);
}

void
registerValueDescription(const std::type_info & type,
                         std::function<ValueDescription * ()> createFn,
                         std::function<void (ValueDescription &)> initFn,
                         bool isDefault)
{
    std::unique_lock<std::recursive_mutex> guard(registryMutex);

    std::shared_ptr<ValueDescription> desc(createFn());
    ExcAssert(desc);
    registry[desc->typeName] = desc;
    registry[type.name()] = desc;

    initFn(*desc);

#if 0
    cerr << "type " << ML::demangle(type.name())
         << " has description "
         << ML::type_name(*desc) << " default " << isDefault << endl;

    if (registry.count(type.name()))
        throw ML::Exception("attempt to double register "
                            + ML::demangle(type.name()));
#endif
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

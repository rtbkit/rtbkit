/* value_description.h                                             -*- C++ -*-
   Jeremy Barnes, 29 March 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Code for description and introspection of values and structures.  Used
   to allow for automated formatters and parsers to be built.
*/

#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include <set>
#include "jml/arch/exception.h"
#include "jml/arch/demangle.h"
#include "json_parsing.h"
#include "json_printing.h"
#include "value_description_fwd.h"

namespace Datacratic {

struct JsonParsingContext;
struct JsonPrintingContext;
struct JSConverters;

enum class ValueKind : int32_t {
    // Atomic, ie all or none is replaced
    ATOM,     ///< Generic, atomic type
    INTEGER,
    FLOAT,
    BOOLEAN,
    STRING,
    ENUM,

    // Non-atomic, ie part of them can be mutated
    OPTIONAL,
    ARRAY,
    STRUCTURE,
    TUPLE,
    VARIANT,
    MAP,
    ANY
};

std::ostream & operator << (std::ostream & stream, ValueKind kind);


/*****************************************************************************/
/* VALUE DESCRIPTION                                                         */
/*****************************************************************************/

/** Value Description

    This describes the content of a C++ structure and allows it to be
    manipulated programatically.
*/

struct ValueDescription {
    typedef std::true_type defined;

    ValueDescription(ValueKind kind,
                     const std::type_info * type,
                     const std::string & typeName = "")
        : kind(kind),
          type(type),
          typeName(typeName.empty() ? ML::demangle(type->name()) : typeName),
          jsConverters(nullptr),
          jsConvertersInitialized(false)
    {
    }

    virtual ~ValueDescription() {};
    
    ValueKind kind;
    const std::type_info * const type;
    std::string typeName;

    void setTypeName(const std::string & newName)
    {
        this->typeName = newName;
    }

    virtual void parseJson(void * val, JsonParsingContext & context) const = 0;
    virtual void printJson(const void * val, JsonPrintingContext & context) const = 0;
    virtual bool isDefault(const void * val) const = 0;
    virtual void setDefault(void * val) const = 0;
    virtual void copyValue(const void * from, void * to) const = 0;
    virtual void moveValue(void * from, void * to) const = 0;
    virtual void swapValues(void * from, void * to) const = 0;
    
    virtual void * optionalMakeValue(void * val) const
    {
        throw ML::Exception("type is not optional");
    }

    virtual const void * optionalGetValue(const void * val) const
    {
        throw ML::Exception("type is not optional");
    }

    virtual size_t getArrayLength(void * val) const
    {
        throw ML::Exception("type is not an array");
    }

    virtual void * getArrayElement(void * val, uint32_t element) const
    {
        throw ML::Exception("type is not an array");
    }

    virtual const void * getArrayElement(const void * val, uint32_t element) const
    {
        throw ML::Exception("type is not an array");
    }

    virtual void setArrayLength(void * val, size_t newLength) const
    {
        throw ML::Exception("type is not an array");
    }
    
    virtual const ValueDescription & contained() const
    {
        throw ML::Exception("type does not contain another");
    }

    // Convert from one type to another, making a copy.
    // Default will go through a JSON conversion.
    virtual void convertAndCopy(const void * from,
                                const ValueDescription & fromDesc,
                                void * to) const;

    struct FieldDescription {
        std::string fieldName;
        std::string comment;
        std::unique_ptr<ValueDescription > description;
        int offset;
        int fieldNum;
    };

    virtual size_t getFieldCount(const void * val) const
    {
        throw ML::Exception("type doesn't support fields");
    }

    virtual const FieldDescription *
    hasField(const void * val, const std::string & name) const
    {
        throw ML::Exception("type doesn't support fields");
    }

    virtual void forEachField(const void * val,
                              const std::function<void (const FieldDescription &)> & onField) const
    {
        throw ML::Exception("type doesn't support fields");
    }

    virtual const FieldDescription & 
    getField(const std::string & field) const
    {
        throw ML::Exception("type doesn't support fields");
    }

    // Storage to cache Javascript converters
    mutable JSConverters * jsConverters;
    mutable bool jsConvertersInitialized;

    static ValueDescription * get(std::string const & name);
};

void registerValueDescription(const std::type_info & type,
                              std::function<ValueDescription * ()>,
                              bool isDefault);

template<typename T>
struct RegisterValueDescription {
    RegisterValueDescription()
    {
        registerValueDescription(typeid(T), [] () { return getDefaultDescription((T*)0); }, true);
    }
};

template<typename T, typename Impl>
struct RegisterValueDescriptionI {
    RegisterValueDescriptionI()
        : done(false)
    {
        registerValueDescription(typeid(T), [] () { return new Impl(); }, true);
    }

    bool done;
};

#define REGISTER_VALUE_DESCRIPTION(type)                                \
    namespace {                                                         \
    static const RegisterValueDescription<type> registerValueDescription##type; \
    }


/*****************************************************************************/
/* VALUE DESCRIPTION TEMPLATE                                                */
/*****************************************************************************/

/** Template class for value description.  This is a type-safe version of a
    value description.
*/
    
template<typename T>
struct ValueDescriptionT : public ValueDescription {

    ValueDescriptionT(ValueKind kind = ValueKind::ATOM)
        : ValueDescription(kind, &typeid(T))
    {
    }

    virtual void parseJson(void * val, JsonParsingContext & context) const
    {
        T * val2 = reinterpret_cast<T *>(val);
        return parseJsonTyped(val2, context);
    }

    virtual void parseJsonTyped(T * val, JsonParsingContext & context) const
    {
        return parseJson(val, context);
    }

    virtual void printJson(const void * val, JsonPrintingContext & context) const
    {
        const T * val2 = reinterpret_cast<const T *>(val);
        return printJsonTyped(val2, context);
    }

    virtual void printJsonTyped(const T * val, JsonPrintingContext & context) const
    {
        return printJson(val, context);
    }

    virtual bool isDefault(const void * val) const
    {
        const T * val2 = reinterpret_cast<const T *>(val);
        return isDefaultTyped(val2);
    }

    virtual bool isDefaultTyped(const T * val) const
    {
        return false;
    }

    virtual void setDefault(void * val) const
    {
        T * val2 = reinterpret_cast<T *>(val);
        setDefaultTyped(val2);
    }

    virtual void setDefaultTyped(T * val) const
    {
        *val = T();
    }

    virtual void copyValue(const void * from, void * to) const
    {
        auto from2 = reinterpret_cast<const T *>(from);
        auto to2 = reinterpret_cast<T *>(to);
        if (from2 == to2)
            return;
        *to2 = *from2;
    }

    virtual void moveValue(void * from, void * to) const
    {
        auto from2 = reinterpret_cast<T *>(from);
        auto to2 = reinterpret_cast<T *>(to);
        if (from2 == to2)
            return;
        *to2 = std::move(*from2);
    }

    virtual void swapValues(void * from, void * to) const
    {
        using std::swap;
        auto from2 = reinterpret_cast<T *>(from);
        auto to2 = reinterpret_cast<T *>(to);
        if (from2 == to2)
            return;
        std::swap(*from2, *to2);
    }

    virtual void * optionalMakeValue(void * val) const
    {
        T * val2 = reinterpret_cast<T *>(val);
        return optionalMakeValueTyped(val2);
    }

    virtual void * optionalMakeValueTyped(T * val) const
    {
        throw ML::Exception("type is not optional");
    }

    virtual const void * optionalGetValue(const void * val) const
    {
        const T * val2 = reinterpret_cast<const T *>(val);
        return optionalGetValueTyped(val2);
    }

    virtual const void * optionalGetValueTyped(const T * val) const
    {
        throw ML::Exception("type is not optional");
    }
};

template<typename T>
DefaultDescription<T> *
getDefaultDescription(T * = 0,
                      typename DefaultDescription<T>::defined * = 0)
{
    return new DefaultDescription<T>();
}

/** Template that returns the type of the default description that should
    be instantiated for the given use case.
*/
template<typename T>
struct GetDefaultDescriptionType {
    typedef typename std::remove_reference<decltype(*getDefaultDescription((T*)0))>::type type;
};


/*****************************************************************************/
/* VALUE DESCRIPTION CONCRETE IMPL                                           */
/*****************************************************************************/

/** Used when there is a concrete description of a value we want to register.

    The main thing that this class does is also registers the value description
    as part of construction.
*/

template<typename T, ValueKind kind = ValueKind::ATOM,
         typename Impl = DefaultDescription<T> >
struct ValueDescriptionI : public ValueDescriptionT<T> {

    static RegisterValueDescriptionI<T, Impl> regme;

    ValueDescriptionI()
        : ValueDescriptionT<T>(kind)
    {
        regme.done = true;
    }
};

template<typename T, ValueKind kind, typename Impl>
RegisterValueDescriptionI<T, Impl>
ValueDescriptionI<T, kind, Impl>::
regme;

inline constexpr void * addOffset(void * base, ssize_t offset)
{
    return reinterpret_cast<char *>(base) + offset;
}

inline constexpr const void * addOffset(const void * base, ssize_t offset)
{
    return reinterpret_cast<const char *>(base) + offset;
}


/*****************************************************************************/
/* VALUE DESCRIPTION WITH DEFAULT                                            */
/*****************************************************************************/

/** Provides an adaptor that adapts a given value description and adds
    a default value.
*/

template<typename T,
         typename Base = typename GetDefaultDescriptionType<T>::type>
struct ValueDescriptionWithDefault : public Base {
    ValueDescriptionWithDefault(T defaultValue)
        : defaultValue(defaultValue)
    {
    }
    
    virtual void parseJsonTyped(T * val,
                                JsonParsingContext & context) const
    {
        std::cerr << "parseJsonTyped" << std::endl;
        std::cerr << typeid(Base).name() << std::endl;
        Base::parseJsonTyped(val, context);
    }

    virtual void printJsonTyped(const T * val,
                                JsonPrintingContext & context) const
    {
        context.writeDouble(*val);
    }

    virtual bool isDefaultTyped(const T * val) const
    {
        return *val == defaultValue;
    }

    virtual void setDefaultTyped(T * val) const
    {
        *val = defaultValue;
    }

    T defaultValue;
};



/*****************************************************************************/
/* STRUCTURE DESCRIPTION BASE                                                */
/*****************************************************************************/

/** Base information for a structure description. */

struct StructureDescriptionBase {

    StructureDescriptionBase(const std::type_info * type,
                             const std::string & structName = "",
                             bool nullAccepted = false)
        : type(type),
          structName(structName.empty() ? ML::demangle(type->name()) : structName),
          nullAccepted(nullAccepted)
    {
    }

    const std::type_info * const type;
    const std::string structName;
    bool nullAccepted;

    typedef ValueDescription::FieldDescription FieldDescription;

    // Comparison object to allow const char * objects to be looked up
    // in the map and so for comparisons to be done with no memory
    // allocations.
    struct StrCompare {
        inline bool operator () (const char * s1, const char * s2) const
        {
            char c1 = *s1++, c2 = *s2++;

            if (c1 < c2) return true;
            if (c1 > c2) return false;
            if (c1 == 0) return false;

            c1 = *s1++; c2 = *s2++;
            
            if (c1 < c2) return true;
            if (c1 > c2) return false;
            if (c1 == 0) return false;

            return strcmp(s1, s2) < 0;
        }

    };

    typedef std::map<const char *, FieldDescription, StrCompare> Fields;
    Fields fields;

    std::vector<std::string> fieldNames;

    std::vector<Fields::const_iterator> orderedFields;

    virtual void parseJson(void * output, JsonParsingContext & context) const
    {
        if (!onEntry(output, context)) return;

        if (nullAccepted && context.isNull()) {
            context.expectNull();
            return;
        }
        
        if (!context.isObject())
            context.exception("expected structure of type " + structName);

        auto onMember = [&] ()
            {
                //using namespace std;
                //cerr << "got field " << context.printPath() << endl;

                auto n = context.fieldNamePtr();
                auto it = fields.find(n);
                if (it == fields.end()) {
                    context.onUnknownField();
                }
                else {
                    it->second.description
                    ->parseJson(addOffset(output, it->second.offset),
                                context);
                }
            };

        context.forEachMember(onMember);

        onExit(output, context);
    }

    virtual void printJson(const void * input, JsonPrintingContext & context) const
    {
        context.startObject();

        for (const auto & it: orderedFields) {
            auto & fd = it->second;

            auto mbr = addOffset(input, fd.offset);
            if (fd.description->isDefault(mbr))
                continue;
            context.startMember(it->first);
            fd.description->printJson(mbr, context);
        }
        
        context.endObject();
    }

    virtual bool onEntry(void * output, JsonParsingContext & context) const = 0;
    virtual void onExit(void * output, JsonParsingContext & context) const = 0;
};


/*****************************************************************************/
/* STRUCTURE DESCRIPTION                                                     */
/*****************************************************************************/

/** Class that implements the base of a description of a structure.  Contains
    methods to register all of the member variables of the class.
*/

template<typename Struct, typename Impl>
struct StructureDescriptionImpl
    :  public ValueDescriptionI<Struct, ValueKind::STRUCTURE, Impl>,
       public StructureDescriptionBase {

    StructureDescriptionImpl(bool nullAccepted = false)
        : StructureDescriptionBase(&typeid(Struct), "", nullAccepted)
    {
    }

    /// Function to be called before parsing; if it returns false parsing stops
    std::function<bool (Struct *, JsonParsingContext & context)> onEntryHandler;

    /// Function to be called whenever an unknown field is found
    std::function<void (Struct *, JsonParsingContext & context)> onUnknownField;

    virtual bool onEntry(void * output, JsonParsingContext & context) const
    {
        if (onEntryHandler) {
            if (!onEntryHandler((Struct *)output, context))
                return false;
        }
        
        if (onUnknownField)
            context.onUnknownFieldHandlers.push_back([=,&context] () { this->onUnknownField((Struct *)output, context); });

        return true;
    }
    
    virtual void onExit(void * output, JsonParsingContext & context) const
    {
        if (onUnknownField)
            context.onUnknownFieldHandlers.pop_back();
    }

    template<typename V, typename Base>
    void addField(std::string name,
                  V Base::* field,
                  std::string comment,
                  ValueDescriptionT<V> * description
                  = getDefaultDescription((V *)0))
    {
        if (fields.count(name.c_str()))
            throw ML::Exception("field '" + name + "' added twice");

        fieldNames.push_back(name);
        const char * fieldName = fieldNames.back().c_str();
        
        auto it = fields.insert
            (Fields::value_type(fieldName, std::move(FieldDescription())))
            .first;
        
        FieldDescription & fd = it->second;
        fd.fieldName = fieldName;
        fd.comment = comment;
        fd.description.reset(description);
        Struct * p = nullptr;
        fd.offset = (size_t)&(p->*field);
        fd.fieldNum = fields.size() - 1;
        orderedFields.push_back(it);
        //using namespace std;
        //cerr << "offset = " << fd.offset << endl;
    }

    /** Add a description with a default value. */
    template<typename V, typename Base,
             typename Desc = ValueDescriptionWithDefault<V> >
    void addField(std::string name,
                  V Base::* field,
                  std::string comment,
                  const V & defaultValue)
    {
        addField(name, field, comment, new Desc(defaultValue));
    }

    template<typename V>
    void addParent(ValueDescriptionT<V> * description_
                   = getDefaultDescription((V *)0));

    virtual size_t getFieldCount(const void * val) const
    {
        return fields.size();
    }

    virtual const FieldDescription *
    hasField(const void * val, const std::string & field) const
    {
        auto it = fields.find(field.c_str());
        if (it != fields.end())
            return &it->second;
        return nullptr;
    }

    virtual void forEachField(const void * val,
                              const std::function<void (const FieldDescription &)> & onField) const
    {
        for (auto f: orderedFields) {
            onField(f->second);
        }
    }

    virtual const FieldDescription & 
    getField(const std::string & field) const
    {
        auto it = fields.find(field.c_str());
        if (it != fields.end())
            return it->second;
        throw ML::Exception("structure has no field " + field);
    }

    virtual void parseJson(void * val, JsonParsingContext & context) const
    {
        return StructureDescriptionBase::parseJson(val, context);
    }

    virtual void printJson(const void * val, JsonPrintingContext & context) const
    {
        return StructureDescriptionBase::printJson(val, context);
    }

    void collectUnparseableJson(Json::Value Struct::* member)
    {
        this->onUnknownField = [=] (Struct * obj, JsonParsingContext & context)
            {
                std::function<Json::Value & (int, Json::Value &)> getEntry
                = [&] (int n, Json::Value & curr) -> Json::Value &
                {
                    if (n == context.path.size())
                        return curr;
                    else if (context.path[n].index != -1)
                        return getEntry(n + 1, curr[context.path[n].index]);
                    else return getEntry(n + 1, curr[context.path[n].key]);
                };

                getEntry(0, obj->*member) = context.expectJson();
            };
    }
};

template<typename T>
struct StructureDescription
    : public StructureDescriptionImpl<T, StructureDescription<T> >
{
    StructureDescription(bool nullAccepted = false)
        : StructureDescriptionImpl<T, StructureDescription<T> >(nullAccepted)
    {
    }
    
};

template<typename Struct, typename Impl>
template<typename V>
void StructureDescriptionImpl<Struct, Impl>::addParent(ValueDescriptionT<V> * description_)
{
    StructureDescription<V> * desc2
        = dynamic_cast<StructureDescription<V> *>(description_);
    if (!desc2) {
        delete description_;
        throw ML::Exception("parent description is not a structure");
    }

    std::unique_ptr<StructureDescription<V> > description(desc2);

    Struct * p = nullptr;
    V * p2 = static_cast<V *>(p);

    size_t ofs = (size_t)p2;

    for (auto & oit: description->orderedFields) {
        FieldDescription & ofd = const_cast<FieldDescription &>(oit->second);
        const std::string & name = ofd.fieldName;

        fieldNames.push_back(name);
        const char * fieldName = fieldNames.back().c_str();

        auto it = fields.insert(Fields::value_type(fieldName, std::move(FieldDescription()))).first;
        FieldDescription & fd = it->second;
        fd.fieldName = fieldName;
        fd.comment = ofd.comment;
        fd.description = std::move(ofd.description);
        
        fd.offset = ofd.offset + ofs;
        fd.fieldNum = fields.size() - 1;
        orderedFields.push_back(it);
    }
}


/*****************************************************************************/
/* ENUM DESCRIPTION                                                          */
/*****************************************************************************/

template<typename Enum>
struct EnumDescription: public ValueDescriptionI<Enum, ValueKind::ENUM,
                                                 EnumDescription<Enum> > {

    EnumDescription()
        : hasDefault(false), defaultValue(Enum(0))
    {
    }

    virtual void parseJsonTyped(Enum * val, JsonParsingContext & context) const
    {
        if (context.isString()) {
            std::string s = context.expectStringAscii();
            auto it = parse.find(s);
            if (it == parse.end())
                context.exception("unknown value for " + this->typeName
                                  + ": " + s);
            *val = it->second;
            return;
        }

        *val = (Enum)context.expectInt();
    }

    virtual void printJsonTyped(const Enum * val, JsonPrintingContext & context) const
    {
        auto it = print.find(*val);
        if (it == print.end())
            context.writeInt((int)*val);
        else context.writeString(it->second);
    }
    
    virtual bool isDefaultTyped(const Enum * val) const
    {
        if (!hasDefault)
            return false;
        return *val == defaultValue;
    }

    virtual void setDefaultTyped(Enum * val) const
    {
        *val = defaultValue;
    }

    bool hasDefault;
    Enum defaultValue;

    void setDefaultValue(Enum value)
    {
        this->hasDefault = true;
        this->defaultValue = value;
    }

    void addValue(const std::string & name, Enum value)
    {
        if (!parse.insert(make_pair(name, value)).second)
            throw ML::Exception("double added name to enum");
        print.insert(make_pair(value, name));
    }

    std::unordered_map<std::string, Enum> parse;
    std::unordered_map<Enum, std::string> print;
};


/*****************************************************************************/
/* LIST DESCRIPTION                                                          */
/*****************************************************************************/

template<typename T>
struct ListDescriptionBase {

    ListDescriptionBase(ValueDescriptionT<T> * inner = getDefaultDescription((T *)0))
        : inner(inner)
    {
    }

    std::unique_ptr<ValueDescriptionT<T> > inner;

    template<typename List>
    void parseJsonTypedList(List * val, JsonParsingContext & context) const
    {
        val->clear();

        if (!context.isArray())
            context.exception("expected array of " + inner->typeName);
        
        auto onElement = [&] ()
            {
                T el;
                inner->parseJsonTyped(&el, context);
                val->emplace_back(std::move(el));
            };
        
        context.forEachElement(onElement);
    }

    template<typename List>
    void parseJsonTypedSet(List * val, JsonParsingContext & context) const
    {
        val->clear();

        if (!context.isArray())
            context.exception("expected array of " + inner->typeName);
        
        auto onElement = [&] ()
            {
                T el;
                inner->parseJsonTyped(&el, context);
                val->insert(std::move(el));
            };
        
        context.forEachElement(onElement);
    }

    template<typename List>
    void printJsonTypedList(const List * val, JsonPrintingContext & context) const
    {
        context.startArray(val->size());

        auto it = val->begin();
        for (unsigned i = 0;  i < val->size();  ++i, ++it) {
            context.newArrayElement();
            inner->printJsonTyped(&(*it), context);
        }
        
        context.endArray();
    }
};


/*****************************************************************************/
/* DEFAULT DESCRIPTION FOR VECTOR                                            */
/*****************************************************************************/

template<typename T>
struct DefaultDescription<std::vector<T> >
    : public ValueDescriptionI<std::vector<T>, ValueKind::ARRAY>,
      public ListDescriptionBase<T> {

    DefaultDescription(ValueDescriptionT<T> * inner
                      = getDefaultDescription((T *)0))
        : ListDescriptionBase<T>(inner)
    {
    }

    virtual void parseJson(void * val, JsonParsingContext & context) const
    {
        std::vector<T> * val2 = reinterpret_cast<std::vector<T> *>(val);
        return parseJsonTyped(val2, context);
    }

    virtual void parseJsonTyped(std::vector<T> * val, JsonParsingContext & context) const
    {
        this->parseJsonTypedList(val, context);
    }

    virtual void printJson(const void * val, JsonPrintingContext & context) const
    {
        const std::vector<T> * val2 = reinterpret_cast<const std::vector<T> *>(val);
        return printJsonTyped(val2, context);
    }

    virtual void printJsonTyped(const std::vector<T> * val, JsonPrintingContext & context) const
    {
        this->printJsonTypedList(val, context);
    }

    virtual bool isDefault(const void * val) const
    {
        const std::vector<T> * val2 = reinterpret_cast<const std::vector<T> *>(val);
        return isDefaultTyped(val2);
    }

    virtual bool isDefaultTyped(const std::vector<T> * val) const
    {
        return val->empty();
    }

    virtual size_t getArrayLength(void * val) const
    {
        const std::vector<T> * val2 = reinterpret_cast<const std::vector<T> *>(val);
        return val2->size();
    }

    virtual void * getArrayElement(void * val, uint32_t element) const
    {
        std::vector<T> * val2 = reinterpret_cast<std::vector<T> *>(val);
        return &val2->at(element);
    }

    virtual const void * getArrayElement(const void * val, uint32_t element) const
    {
        const std::vector<T> * val2 = reinterpret_cast<const std::vector<T> *>(val);
        return &val2->at(element);
    }

    virtual void setArrayLength(void * val, size_t newLength) const
    {
        std::vector<T> * val2 = reinterpret_cast<std::vector<T> *>(val);
        val2->resize(newLength);
    }
    
    virtual const ValueDescription & contained() const
    {
        return *this->inner;
    }
};


/*****************************************************************************/
/* DEFAULT DESCRIPTION FOR SET                                               */
/*****************************************************************************/

template<typename T>
struct DefaultDescription<std::set<T> >
    : public ValueDescriptionI<std::set<T>, ValueKind::ARRAY>,
      public ListDescriptionBase<T> {

    DefaultDescription(ValueDescriptionT<T> * inner
                      = getDefaultDescription((T *)0))
        : ListDescriptionBase<T>(inner)
    {
    }

    virtual void parseJson(void * val, JsonParsingContext & context) const
    {
        std::set<T> * val2 = reinterpret_cast<std::set<T> *>(val);
        return parseJsonTyped(val2, context);
    }

    virtual void parseJsonTyped(std::set<T> * val, JsonParsingContext & context) const
    {
        this->parseJsonTypedSet(val, context);
    }

    virtual void printJson(const void * val, JsonPrintingContext & context) const
    {
        const std::set<T> * val2 = reinterpret_cast<const std::set<T> *>(val);
        return printJsonTyped(val2, context);
    }

    virtual void printJsonTyped(const std::set<T> * val, JsonPrintingContext & context) const
    {
        this->printJsonTypedList(val, context);
    }

    virtual bool isDefault(const void * val) const
    {
        const std::set<T> * val2 = reinterpret_cast<const std::set<T> *>(val);
        return isDefaultTyped(val2);
    }

    virtual bool isDefaultTyped(const std::set<T> * val) const
    {
        return val->empty();
    }

    virtual size_t getArrayLength(void * val) const
    {
        const std::set<T> * val2 = reinterpret_cast<const std::set<T> *>(val);
        return val2->size();
    }

    virtual void * getArrayElement(void * val, uint32_t element) const
    {
        throw ML::Exception("can't mutate set elements");
    }

    virtual const void * getArrayElement(const void * val, uint32_t element) const
    {
        const std::set<T> * val2 = reinterpret_cast<const std::set<T> *>(val);
        if (element >= val2->size())
            throw ML::Exception("Invalid set element number");
        auto it = val2->begin();
        for (unsigned i = 0;  i < element;  ++i, ++i) ;
        return &*it;
    }

    virtual void setArrayLength(void * val, size_t newLength) const
    {
        throw ML::Exception("cannot adjust length of a set");
    }
    
    virtual const ValueDescription & contained() const
    {
        return *this->inner;
    }
};


/*****************************************************************************/
/* DEFAULT DESCRIPTION FOR MAP                                               */
/*****************************************************************************/

template<typename T>
struct DefaultDescription<std::map<std::string, T> >
    : public ValueDescriptionI<std::map<std::string, T>, ValueKind::MAP> {

    DefaultDescription(ValueDescriptionT<T> * inner
                      = getDefaultDescription((T *)0))
        : inner(inner)
    {
    }

    std::unique_ptr<ValueDescriptionT<T> > inner;

    typedef ValueDescription::FieldDescription FieldDescription;

    virtual void parseJson(void * val, JsonParsingContext & context) const
    {
        auto * val2 = reinterpret_cast<std::map<std::string, T> *>(val);
        return parseJsonTyped(val2, context);
    }

    virtual void parseJsonTyped(std::map<std::string, T> * val, JsonParsingContext & context) const
    {
        std::map<std::string, T> res;

        auto onMember = [&] ()
            {
                inner->parseJsonTyped(&res[context.fieldName()], context);
            };

        context.forEachMember(onMember);

        val->swap(res);
    }

    virtual void printJson(const void * val, JsonPrintingContext & context) const
    {
        auto * val2 = reinterpret_cast<const std::map<std::string, T> *>(val);
        return printJsonTyped(val2, context);
    }

    virtual void printJsonTyped(const std::map<std::string, T> * val,
                                JsonPrintingContext & context) const
    {
        context.startObject();
        for (auto & v: *val) {
            context.startMember(v.first);
            inner->printJsonTyped(&v.second, context);
        }
        context.endObject();
    }

    virtual bool isDefault(const void * val) const
    {
        auto * val2 = reinterpret_cast<const std::map<std::string, T> *>(val);
        return isDefaultTyped(val2);
    }

    virtual bool isDefaultTyped(const std::map<std::string, T> * val) const
    {
        return val->empty();
    }

    virtual size_t getFieldCount(const void * val) const
    {
        auto * val2 = reinterpret_cast<const std::map<std::string, T> *>(val);
        return val2->size();
    }

    virtual const FieldDescription *
    hasField(const void * val, const std::string & name) const
    {
        throw ML::Exception("map hasField: needs work");
        //auto * val2 = reinterpret_cast<const std::map<std::string, T> *>(val);
        //return val2->count(name);
    }

    virtual void forEachField(const void * val,
                              const std::function<void (const FieldDescription &)> & onField) const
    {
        throw ML::Exception("map forEachField: needs work");
    }

    virtual const FieldDescription & 
    getField(const std::string & field) const
    {
        throw ML::Exception("map getField: needs work");
    }
    
    virtual const ValueDescription & contained() const
    {
        return *this->inner;
    }
};


/*****************************************************************************/
/* DESCRIPTION FROM BASE                                                     */
/*****************************************************************************/

/** This class is used for when you want to create a value description for
    a class that derives from a base class that provides most or all of its
    functionality (eg, a vector).  It forwards all of the methods to the
    base value description.
*/

template<typename T, typename Base,
         typename BaseDescription
             = typename GetDefaultDescriptionType<Base>::type>
struct DescriptionFromBase
    : public ValueDescriptionT<T> {

    DescriptionFromBase(BaseDescription * inner
                      = getDefaultDescription((Base *)0))
        : inner(inner)
    {
    }

    std::unique_ptr<BaseDescription> inner;

    constexpr ssize_t offset() const
    {
        return (ssize_t)(static_cast<Base *>((T *)0));
    }

    void * fixPtr(void * ptr) const
    {
        return addOffset(ptr, offset());
    }

    const void * fixPtr(const void * ptr) const
    {
        return addOffset(ptr, offset());
    }

    virtual void parseJson(void * val, JsonParsingContext & context) const
    {
        inner->parseJson(fixPtr(val), context);
    }

    virtual void parseJsonTyped(T * val, JsonParsingContext & context) const
    {
        inner->parseJson(fixPtr(val), context);
    }

    virtual void printJson(const void * val, JsonPrintingContext & context) const
    {
        inner->printJson(fixPtr(val), context);
    }

    virtual void printJsonTyped(const T * val, JsonPrintingContext & context) const
    {
        inner->printJson(fixPtr(val), context);
    }

    virtual bool isDefault(const void * val) const
    {
        return inner->isDefault(fixPtr(val));
    }

    virtual bool isDefaultTyped(const T * val) const
    {
        return inner->isDefault(fixPtr(val));
    }

    virtual size_t getArrayLength(void * val) const
    {
        return inner->getArrayLength(fixPtr(val));
    }

    virtual void * getArrayElement(void * val, uint32_t element) const
    {
        return inner->getArrayElement(fixPtr(val), element);
    }

    virtual const void * getArrayElement(const void * val, uint32_t element) const
    {
        return inner->getArrayElement(fixPtr(val), element);
    }

    virtual void setArrayLength(void * val, size_t newLength) const
    {
        inner->setArrayLength(fixPtr(val), newLength);
    }
    
    virtual const ValueDescription & contained() const
    {
        return inner->contained();
    }
};


/*****************************************************************************/
/* CONVERSION FUNCTIONS                                                      */
/*****************************************************************************/


// Template set for which hasToJson<T>::value is true if and only if it has a function
// Json::Value T::toJson() const
template<typename T, typename Enable = void>
struct hasToJson {
    enum { value = false };
};

template<typename T>
struct hasToJson<T, typename std::enable_if<std::is_convertible<Json::Value, decltype(std::declval<const T>().toJson())>::value>::type> {
    enum { value = true };
};

// Template set for which hasFromJson<T>::value is true if and only if it has a function
// static T T::fromJson(Json::Value)
template<typename T, typename Enable = void>
struct hasFromJson {
    enum { value = false };
};

template<typename T>
struct hasFromJson<T, typename std::enable_if<std::is_convertible<T, decltype(T::fromJson(std::declval<Json::Value>()))>::value>::type> {
    enum { value = true };
};

// jsonDecode implementation for any type which:
// 1) has a default description;
// 2) does NOT have a fromJson() function (there is a simpler overload for this case)
template<typename T>
T jsonDecode(const Json::Value & json, T * = 0,
             decltype(getDefaultDescription((T *)0)) * = 0,
             typename std::enable_if<!hasFromJson<T>::value>::type * = 0)
{
    T result;

    static std::unique_ptr<ValueDescription> desc
        (getDefaultDescription((T *)0));
    StructuredJsonParsingContext context(json);
    desc->parseJson(&result, context);
    return result;
}

// jsonDecode implementation for any type which:
// 1) has a default description;
// 2) does NOT have a fromJson() function (there is a simpler overload for this case)
template<typename T>
T jsonDecodeStr(const std::string & json, T * = 0,
                decltype(getDefaultDescription((T *)0)) * = 0,
                typename std::enable_if<!hasFromJson<T>::value>::type * = 0)
{
    T result;

    static std::unique_ptr<ValueDescription> desc
        (getDefaultDescription((T *)0));
    StreamingJsonParsingContext context(json, json.c_str(), json.c_str() + json.size());
    desc->parseJson(&result, context);
    return result;
}

// In-place json decoding
template<typename T, typename V>
void jsonDecode(V && json, T & val)
{
    val = std::move(jsonDecode(json, (T *)0));
}

// In-place json decoding
template<typename T>
void jsonDecodeStr(const std::string & json, T & val)
{
    val = std::move(jsonDecodeStr(json, (T *)0));
}

// jsonEncode implementation for any type which:
// 1) has a default description;
// 2) does NOT have a toJson() function (there is a simpler overload for this case)
template<typename T>
Json::Value jsonEncode(const T & obj,
                       decltype(getDefaultDescription((T *)0)) * = 0,
                       typename std::enable_if<!hasToJson<T>::value>::type * = 0)
{
    static std::unique_ptr<ValueDescription> desc
        (getDefaultDescription((T *)0));
    StructuredJsonPrintingContext context;
    desc->printJson(&obj, context);
    return std::move(context.output);
}

// jsonEncode implementation for any type which:
// 1) has a default description;
// 2) does NOT have a toJson() function (there is a simpler overload for this case)
template<typename T>
std::string jsonEncodeStr(const T & obj,
                          decltype(getDefaultDescription((T *)0)) * = 0,
                          typename std::enable_if<!hasToJson<T>::value>::type * = 0)
{
    static std::unique_ptr<ValueDescription> desc
        (getDefaultDescription((T *)0));
    std::ostringstream stream;
    StreamJsonPrintingContext context(stream);
    desc->printJson(&obj, context);
    return std::move(stream.str());
}

inline Json::Value jsonEncode(const char * str)
{
    return str;
}

} // namespace Datacratic



/// Macro to introduce a class TypeDescription that is a structure
/// description for that type, and a getDefaultDescription()
/// overload for it.  The constructor still needs to be done.
#define CREATE_STRUCTURE_DESCRIPTION_NAMED(Name, Type)          \
    struct Name                                                 \
        : public Datacratic::StructureDescriptionImpl<Type, Name> { \
        Name();                                                 \
    };                                                          \
                                                                \
    inline Name *                                               \
    getDefaultDescription(Type *)                               \
    {                                                           \
        return new Name();                                      \
    }                                                          

#define CREATE_STRUCTURE_DESCRIPTION(Type)                      \
    CREATE_STRUCTURE_DESCRIPTION_NAMED(Type##Description, Type)

#define CREATE_CLASS_DESCRIPTION_NAMED(Name, Type)              \
    struct Name                                                 \
        : public Datacratic::StructureDescriptionImpl<Type, Name> { \
        Name() {                                                \
            Type::createDescription(*this);                     \
        }                                                       \
    };                                                          \
                                                                \
    inline Name *                                               \
    getDefaultDescription(Type *)                               \
    {                                                           \
        return new Name();                                      \
    }

#define CREATE_CLASS_DESCRIPTION(Type)                          \
    CREATE_CLASS_DESCRIPTION_NAMED(Type##Description, Type)

#define CREATE_ENUM_DESCRIPTION_NAMED(Name, Type)          \
    struct Name                                                 \
        : public Datacratic::EnumDescription<Type> { \
        Name();                                                 \
    };                                                          \
                                                                \
    inline Name *                                               \
    getDefaultDescription(Type *)                               \
    {                                                           \
        return new Name();                                      \
    }                                                          

#define CREATE_ENUM_DESCRIPTION(Type)                      \
    CREATE_ENUM_DESCRIPTION_NAMED(Type##Description, Type)

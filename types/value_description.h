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
#include "jml/arch/exception.h"
#include "jml/arch/demangle.h"
#include "json_parsing.h"
#include "json_printing.h"
#include "value_description_fwd.h"

namespace Datacratic {

struct JsonParsingContext;
struct JsonPrintingContext;

struct ValueDescription {
    typedef std::true_type defined;

    ValueDescription(const std::type_info * type,
                     const std::string & typeName = "")
        : type(type),
          typeName(typeName.empty() ? type->name() : typeName)
    {
    }
    
    const std::type_info * const type;
    const std::string typeName;

    virtual void parseJson(void * val, JsonParsingContext & context) const = 0;
    virtual void printJson(const void * val, JsonPrintingContext & context) const = 0;
    virtual bool isDefault(const void * val) const = 0;

    // Serialization and reconstitution in the JML (boost serialization)
    // framework
    //virtual void jmlSerialize(const void * val, ML::DB::Store_Writer & store) const = 0;
    //virtual void jmlReconstitute(void * val, ML::DB::Store_Reader & store) const = 0;
};

template<typename T>
struct ValueDescriptionT : public ValueDescription {

    ValueDescriptionT()
        : ValueDescription(&typeid(T))
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
};

template<typename T>
ValueDescriptionT<T> * getDefaultDescription(T * = 0,
                                            typename DefaultDescription<T>::defined * = 0)
{
    return new DefaultDescription<T>();
}

template<class Struct>
struct StructureDescription;

inline void * addOffset(void * base, ssize_t offset)
{
    char * c = reinterpret_cast<char *>(base);
    return c + offset;
}

inline const void * addOffset(const void * base, ssize_t offset)
{
    const char * c = reinterpret_cast<const char *>(base);
    return c + offset;
}


/*****************************************************************************/
/* STRUCTURE DESCRIPTION BASE                                                */
/*****************************************************************************/

/** Base information for a structure description. */

struct StructureDescriptionBase {

    StructureDescriptionBase(const std::type_info * type,
                             const std::string & typeName = "",
                             bool nullAccepted = false)
        : type(type),
          typeName(typeName.empty() ? ML::demangle(type->name()) : typeName),
          nullAccepted(nullAccepted)
    {
    }

    const std::type_info * const type;
    const std::string typeName;
    bool nullAccepted;

    struct FieldDescription {
        std::string fieldName;
        std::string comment;
        std::unique_ptr<ValueDescription > description;
        int offset;
        int fieldNum;
    };

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
            context.exception("expected structure of type " + typeName);

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

template<class Struct>
struct StructureDescription
    :  public ValueDescriptionT<Struct>,
       public StructureDescriptionBase {

    StructureDescription(bool nullAccepted = false)
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

    template<typename V>
    void addParent(ValueDescriptionT<V> * description_
                   = getDefaultDescription((V *)0))
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

    virtual void parseJson(void * val, JsonParsingContext & context) const
    {
        return StructureDescriptionBase::parseJson(val, context);
    }

    virtual void printJson(const void * val, JsonPrintingContext & context) const
    {
        return StructureDescriptionBase::printJson(val, context);
    }

    virtual const FieldDescription & 
    getField(const std::string & field) const
    {
        auto it = fields.find(field.c_str());
        if (it != fields.end())
            return it->second;
        throw ML::Exception("structure has no field " + field);
    }
};

template<typename Enum>
struct EnumDescription: public ValueDescriptionT<Enum> {

    struct Value {
        int value;
        std::string name;
    };

    std::unordered_map<std::string, int> parse;
    std::unordered_map<int, Value> print;
};

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
    void printJsonTypedList(const List * val, JsonPrintingContext & context) const
    {
        context.startArray(val->size());

        for (unsigned i = 0;  i < val->size();  ++i) {
            context.newArrayElement();
            inner->printJsonTyped(&(*val)[i], context);
        }
        
        context.endArray();
    }
};

template<typename T>
struct ValueDescriptionT<std::vector<T> >
    : public ValueDescription,
      public ListDescriptionBase<T> {

    ValueDescriptionT(ValueDescriptionT<T> * inner
                     = getDefaultDescription((T *)0))
        : ValueDescription(&typeid(std::vector<T>)),
          ListDescriptionBase<T>(inner)
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

    virtual void jmlSerialize(ML::DB::Store_Writer & store) const
    {
    }

    virtual void jmlReconstitute(ML::DB::Store_Reader & store) const
    {
    }
};


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


} // namespace Datacratic

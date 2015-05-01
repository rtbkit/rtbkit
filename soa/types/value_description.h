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
#include "jml/arch/demangle.h"
#include "jml/utils/exc_assert.h"
#include "jml/utils/filter_streams.h"
#include "jml/utils/smart_ptr_utils.h"
#include "json_parsing.h"
#include "json_printing.h"
#include "value_description_fwd.h"
#include "soa/utils/type_traits.h"

namespace Datacratic {

/** Tag structure to indicate that we want to only construct a
    description, not initialize it.  Initialization can be
    performed later.
*/

struct ConstructOnly {
};

static const ConstructOnly constructOnly;

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
    LINK,
    ARRAY,
    STRUCTURE,
    TUPLE,
    VARIANT,
    MAP,
    ANY
};

enum class OwnershipModel : int32_t {
    NONE,
    UNIQUE,
    SHARED
};

std::ostream & operator << (std::ostream & stream, ValueKind kind);


struct ValueDescription;

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
    const std::type_info * type;
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
    virtual void * constructDefault() const = 0;
    virtual void destroy(void *) const = 0;

    
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

    /** Return the value description for the nth array element.  This is
        necessary for tuple types, which don't have the same type for each
        element.
    */
    virtual const ValueDescription &
    getArrayElementDescription(const void * val, uint32_t element)
    {
        return contained();
    }

    virtual void setArrayLength(void * val, size_t newLength) const
    {
        throw ML::Exception("type is not an array");
    }
    
    virtual const ValueDescription & contained() const
    {
        throw ML::Exception("type does not contain another");
    }

    virtual OwnershipModel getOwnershipModel() const
    {
        throw ML::Exception("type does not define an ownership type");
    }

    virtual void* getLink(void* obj) const
    {
        throw ML::Exception("type is not a link");
    }

    virtual void set(
            void* obj, void* value, const ValueDescription* valueDesc) const
    {
        throw ML::Exception("type can't be written to");
    }


    // Convert from one type to another, making a copy.
    // Default will go through a JSON conversion.
    virtual void convertAndCopy(const void * from,
                                const ValueDescription & fromDesc,
                                void * to) const;

    struct FieldDescription {
        std::string fieldName;
        std::string comment;
        std::shared_ptr<const ValueDescription > description;
        int offset;
        int fieldNum;

        void* getFieldPtr(void* obj) const
        {
            return ((char*) obj) + offset;
        }
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

    virtual const std::vector<std::string> getEnumKeys() const {
        throw ML::Exception("type is not an enum");
    }

    // Storage to cache Javascript converters
    mutable JSConverters * jsConverters;
    mutable bool jsConvertersInitialized;

    /** Get the value description for a type name */
    static std::shared_ptr<const ValueDescription>
    get(std::string const & name);

    /** Get the value description for a type */
    static std::shared_ptr<const ValueDescription>
    get(const std::type_info & type);

    /** Get the value description for a type */
    template<typename T>
    static std::shared_ptr<const ValueDescriptionT<T> >
    getType()
    {
        auto base = get(typeid(T));

        if (!base)
            return nullptr;

        auto res
            = std::dynamic_pointer_cast<const ValueDescriptionT<T> >(base);
        if (!res)
            throw ML::Exception("logic error in registry: wrong type: "
                                + ML::type_name(*base) + " not convertible to "
                                + ML::type_name<const ValueDescriptionT<T>>());
        return res;
    }

    /** Get the value description for an object */
    template<typename T>
    static std::shared_ptr<const ValueDescriptionT<T> >
    getType(const T * val)
    {
        // TODO: support polymorphic objects
        return getType<T>();
    }

    bool isSame(const ValueDescription* other) const
    {
        return type == other->type;
    }

    void checkSame(const ValueDescription* other) const
    {
        ExcCheck(isSame(other),
                "Wrong object type: "
                "expected <" + typeName + "> got <" + other->typeName + ">");
    }


    bool isChildOf(const ValueDescription* base) const
    {
        if (isSame(base)) return true;

        for (const auto& parent : parents)
            if (parent->isChildOf(base))
                return true;

        return false;
    }

    void checkChildOf(const ValueDescription* base) const
    {
        ExcCheck(isChildOf(base),
                "value of type " + typeName +
                " is not convertible to type " + typeName);
    }


    std::vector< std::shared_ptr<ValueDescription> > parents;

};

/** Register the given value description with the system under the given
    type name.
*/
void registerValueDescription(const std::type_info & type,
                              std::function<ValueDescription * ()>,
                              bool isDefault);

/** Register the value description with a two phase create then
    initialize protocol.  This is needed for recursive structures.
*/
void registerValueDescription(const std::type_info & type,
                              std::function<ValueDescription * ()> createFn,
                              std::function<void (ValueDescription &)> initFn,
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
/* PURE VALUE DESCRIPTION                                                    */
/*****************************************************************************/

template<typename T>
struct PureValueDescription : public ValueDescription {
    PureValueDescription() :
        ValueDescription(ValueKind::ATOM, &typeid(T)) {
    }

    virtual void parseJson(void * val, JsonParsingContext & context) const {};
    virtual void printJson(const void * val, JsonPrintingContext & context) const {};
    virtual bool isDefault(const void * val) const { return false; }
    virtual void setDefault(void * val) const {}
    virtual void copyValue(const void * from, void * to) const {}
    virtual void moveValue(void * from, void * to) const {}
    virtual void swapValues(void * from, void * to) const {}
    virtual void * constructDefault() const {return nullptr;}
    virtual void destroy(void *) const {}

};

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
        copyValue(to, from, typename Datacratic::is_copy_assignable<T>::type());
    }

    virtual void moveValue(void * from, void * to) const
    {
        moveValue(to, from, typename Datacratic::is_move_assignable<T>::type());
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

    virtual void * constructDefault() const
    {
        return constructDefault(typename Datacratic::is_default_constructible<T>::type());
    }

    virtual void destroy(void * val) const
    {
        delete (T*)val;
    }

    virtual void set(
            void* obj, void* value, const ValueDescription* valueDesc) const
    {
        checkSame(valueDesc);
        copyValue(value, obj);
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

private:

    void copyValue(void* obj, const void* value, std::true_type) const
    {
        if (obj == value) return;
        *static_cast<T*>(obj) = *static_cast<const T*>(value);
    }

    void copyValue(void* obj, const void* value, std::false_type) const
    {
        throw ML::Exception("type is not copy assignable");
    }


    void moveValue(void* obj, void* value, std::true_type) const
    {
        if (obj == value) return;
        *static_cast<T*>(obj) = std::move(*static_cast<T*>(value));
    }

    void moveValue(void* obj, void* value, std::false_type) const
    {
        throw ML::Exception("type is not move assignable");
    }

    // Template parameter so not instantiated for types that are not
    // default constructible
    template<typename X>
    void * constructDefault(X) const
    {
        return new T();
    }

    void * constructDefault(std::false_type) const
    {
        throw ML::Exception("type is not default constructible");
    }
};

/** Basic function to implement getting a default description for a type.
    The default implementation will work for any type for which
    DefaultDescription<T> is defined.
*/
template<typename T>
inline DefaultDescription<T> *
getDefaultDescription(T * = 0,
                      typename DefaultDescription<T>::defined * = 0)
{
    return new DefaultDescription<T>();
}

/** This function is used to get a default description that is uninitialized.
    It's necessary when working with recursive data types, as the object needs
    to be registered before it can be initialized.
    
    The default implementation simply uses getDefaultDescription().
*/
template<typename T>
inline decltype(getDefaultDescription((T *)0))
getDefaultDescriptionUninitialized(T * = 0)
{
    return getDefaultDescription((T *)0);
}

/** This function is used to initialize a default description that was
    previously uninitialized.  The default does nothing.  This function
    may be overridden for the value descriptions that require it.
*/
template<typename Desc>
inline void
initializeDefaultDescription(Desc & desc)
{
}


/** Template that returns the type of the default description that should
    be instantiated for the given use case.
*/
template<typename T>
struct GetDefaultDescriptionType {
    typedef typename std::remove_reference<decltype(*getDefaultDescription((T*)0))>::type type;
};

// Json::Value T::toJson() const
template<typename T, typename Enable = void>
struct hasDefaultDescription {
    enum { value = false };
    typedef bool is_false;
};

template<typename T>
struct hasDefaultDescription<T, typename DefaultDescription<T>::defined *> {
    enum { value = true };
    typedef bool is_true;
};


/** Return the shared copy of the default description for this value.  This
    will look it up in the registry, and if not found, will create (and
    register) it.
*/
template<typename T>
std::shared_ptr<const typename GetDefaultDescriptionType<T>::type>
getDefaultDescriptionShared(T * = 0)
{
    auto res = ValueDescription::getType<T>();
    if (!res) {
        auto createFn = [] ()
            {
                return getDefaultDescriptionUninitialized((T *)0);
            };

        auto initFn = [] (ValueDescription & desc)
            {
                auto & descTyped
                = dynamic_cast<typename GetDefaultDescriptionType<T>::type &>
                    (desc);
                initializeDefaultDescription(descTyped);
            };
        
        // For now, register it if it wasn't before.  Eventually this should
        // be done elsewhere.
        registerValueDescription(typeid(T), createFn, initFn, true);

        res = ValueDescription::getType<T>();
    }
    ExcAssert(res);

    auto cast = std::dynamic_pointer_cast<const typename GetDefaultDescriptionType<T>::type>(res);

    if (!cast)
        throw ML::Exception("logic error in registry: wrong type: "
                            + ML::type_name(*res) + " not convertible to "
                            + ML::type_name<typename GetDefaultDescriptionType<T>::type>());

    return cast;
}


template<typename T, typename Enable = void>
struct GetDefaultDescriptionMaybe {
    static std::shared_ptr<const ValueDescription> get()
    {
        return nullptr;
    }
};

template<typename T>
struct GetDefaultDescriptionMaybe<T, decltype(getDefaultDescription((T *)0))> {
    static std::shared_ptr<const ValueDescription> get()
    {
        return getDefaultDescriptionShared((T *)0);
    }
};

/** Return the default description for the given type if it exists, or
    otherwise return a null pointer.
*/
    
template<typename T>
inline std::shared_ptr<const ValueDescription>
maybeGetDefaultDescriptionShared(T * = 0)
{
    auto result = GetDefaultDescriptionMaybe<T>::get();
    if (!result) {
        // Look to see if it's registered in the registry so that we can
        // get it
        result = ValueDescription::getType<T>();
    }
    return result;
}


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
        Base::parseJsonTyped(val, context);
    }

    virtual void printJsonTyped(const T * val,
                                JsonPrintingContext & context) const
    {
        Base::printJsonTyped(val, context);
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
                             ValueDescription * owner,
                             const std::string & structName = "",
                             bool nullAccepted = false)
        : type(type),
          structName(structName.empty() ? ML::demangle(type->name()) : structName),
          nullAccepted(nullAccepted),
          owner(owner)
    {
    }

    const std::type_info * type;
    std::string structName;
    bool nullAccepted;
    ValueDescription * owner;

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

    struct Exception: public ML::Exception {
        Exception(JsonParsingContext & context,
                  const std::string & message)
            : ML::Exception("at " + context.printPath() + ": " + message)
        {
        }

        virtual ~Exception() throw ()
        {
        }
    };

    virtual void parseJson(void * output, JsonParsingContext & context) const
    {
        try {

            if (!onEntry(output, context)) return;

            if (nullAccepted && context.isNull()) {
                context.expectNull();
                return;
            }
        
            if (!context.isObject())
                context.exception("expected structure of type " + structName);

            auto onMember = [&] ()
                {
                    try {
                        auto n = context.fieldNamePtr();
                        auto it = fields.find(n);
                        if (it == fields.end()) {
                            context.onUnknownField(owner);
                        }
                        else {
                            it->second.description
                                ->parseJson(addOffset(output,
                                                      it->second.offset),
                                            context);
                        }
                    }
                    catch (const Exception & exc) {
                        throw;
                    }
                    catch (const std::exception & exc) {
                        throw Exception(context, exc.what());
                    }
                    catch (...) {
                        throw;
                    }
                };

            context.forEachMember(onMember);

            onExit(output, context);
        }
        catch (const Exception & exc) {
            throw;
        }
        catch (const std::exception & exc) {
            throw Exception(context, exc.what());
        }
        catch (...) {
            throw;
        }
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

template<typename Struct>
struct StructureDescription
    : public ValueDescriptionT<Struct>,
      public StructureDescriptionBase {
    StructureDescription(bool nullAccepted = false,
                         const std::string & structName = "")
        : ValueDescriptionT<Struct>(ValueKind::STRUCTURE),
          StructureDescriptionBase(&typeid(Struct), this, structName,
                                   nullAccepted)
    {
    }

    /// Function to be called before parsing; if it returns false parsing stops
    std::function<bool (Struct *, JsonParsingContext & context)> onEntryHandler;

    /// Function to be called whenever an unknown field is found
    std::function<void (Struct *, JsonParsingContext & context)> onUnknownField;

    /// Function to be called after parsing and validation
    std::function<void (Struct *, JsonParsingContext & context)> onPostValidate;

    virtual bool onEntry(void * output, JsonParsingContext & context) const
    {
        if (onEntryHandler) {
            if (!onEntryHandler((Struct *)output, context))
                return false;
        }
        
        if (onUnknownField)
            context.onUnknownFieldHandlers.push_back([=,&context] (const ValueDescription *) { this->onUnknownField((Struct *)output, context); });

        return true;
    }
    
    virtual void onExit(void * output, JsonParsingContext & context) const
    {
        if (onUnknownField)
            context.onUnknownFieldHandlers.pop_back();
        postValidate(output, context);
        StructureDescription * structParent;
        for (auto parent: parents) {
            structParent = static_cast<StructureDescription *>(parent.get());
            structParent->postValidate(output, context);
        }
    }

    virtual void postValidate(void * output, JsonParsingContext & context) const
    {
        if (onPostValidate) {
            Struct * structOutput = static_cast<Struct *>(output);
            onPostValidate(structOutput, context);
        }
    }

    template<typename V, typename Base>
    void addField(std::string name,
                  V Base::* field,
                  std::string comment,
                  ValueDescriptionT<V> * description)
    {
        addField(name, field, comment,
                 std::shared_ptr<const ValueDescriptionT<V> >(description));
    }

    template<typename V, typename Base>
    void addField(std::string name,
                  V Base::* field,
                  std::string comment,
                  std::shared_ptr<const ValueDescriptionT<V> > description
                      = getDefaultDescriptionShared<V>())
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
        fd.description = description;
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

    using ValueDescriptionT<Struct>::parents;

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
                    else return getEntry(n + 1, curr[context.path[n].fieldName()]);
                };

                getEntry(0, obj->*member) = context.expectJson();
            };
    }
};

/** Base class for an implementation of a structure description.  It
    derives from StructureDescription<Struct>, and also registers
    itself.
*/

template<typename Struct, typename Impl>
struct StructureDescriptionImpl
    :  public StructureDescription<Struct> {
    
    StructureDescriptionImpl(bool nullAccepted = false)
        : StructureDescription<Struct>(nullAccepted)
    {
        regme.done = true;
    }
    
    static RegisterValueDescriptionI<Struct, Impl> regme;
};

template<typename Struct, typename Impl>
RegisterValueDescriptionI<Struct, Impl>
StructureDescriptionImpl<Struct, Impl>::
regme;


template<typename Struct>
template<typename V>
void StructureDescription<Struct>::
addParent(ValueDescriptionT<V> * description_)
{
    StructureDescription<V> * desc2
        = dynamic_cast<StructureDescription<V> *>(description_);
    if (!desc2) {
        delete description_;
        throw ML::Exception("parent description is not a structure");
    }

    std::shared_ptr<StructureDescription<V> > description(desc2);
    parents.push_back(description);

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
struct EnumDescription: public ValueDescriptionT<Enum> {

    EnumDescription()
        : ValueDescriptionT<Enum>(ValueKind::ENUM),
          hasDefault(false), defaultValue(Enum(0))
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

    void addValue(const std::string & name, Enum value,
                  const std::string & description)
    {
        if (!parse.insert(make_pair(name, value)).second)
            throw ML::Exception("double added name to enum");
        print.insert(make_pair(value, name));

        // TODO: description
    }

    virtual const std::vector<std::string> getEnumKeys() const {
        std::vector<std::string> res;
        for (const auto it: print) {
            res.push_back(it.second);
        }
        return res;
    }

    std::unordered_map<std::string, Enum> parse;
    std::map<Enum, std::string> print;
};


/*****************************************************************************/
/* LIST DESCRIPTION                                                          */
/*****************************************************************************/

template<typename T>
struct ListDescriptionBase {

    ListDescriptionBase(ValueDescriptionT<T> * inner)
        : inner(inner)
    {
    }

    ListDescriptionBase(std::shared_ptr<const ValueDescriptionT<T> > inner
                        = getDefaultDescriptionShared((T *)0))
        : inner(inner)
    {
    }

    std::shared_ptr<const ValueDescriptionT<T> > inner;

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

    DefaultDescription(ValueDescriptionT<T> * inner)
        : ListDescriptionBase<T>(inner)
    {
    }

    DefaultDescription(std::shared_ptr<const ValueDescriptionT<T> > inner
                       = getDefaultDescriptionShared((T *)0))
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

    DefaultDescription(ValueDescriptionT<T> * inner)
        : ListDescriptionBase<T>(inner)
    {
    }

    DefaultDescription(std::shared_ptr<const ValueDescriptionT<T> > inner
                      = getDefaultDescriptionShared((T *)0))
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

inline std::string stringToKey(const std::string & str, std::string *) { return str; }
inline std::string keyToString(const std::string & str) { return str; }

template<typename T>
inline T stringToKey(const std::string & str, T *) { return boost::lexical_cast<T>(str); }

template<typename T>
inline std::string keyToString(const T & key)
{
    using std::to_string;
    return to_string(key);
}

template<typename T, typename Enable = void>
struct FreeFunctionKeyCodec {
    static T decode(const std::string & s, T *) { return stringToKey(s, (T *)0); }
    static std::string encode(const T & t) { return keyToString(t); }
};

template<typename K, typename T, typename KeyCodec = FreeFunctionKeyCodec<K> >
struct MapValueDescription
    : public ValueDescriptionI<std::map<K, T>, ValueKind::MAP> {

    MapValueDescription(ConstructOnly)
    {
    }

    MapValueDescription(ValueDescriptionT<T> * inner
                      = getDefaultDescription((T *)0))
        : inner(inner)
    {
    }

    MapValueDescription(std::shared_ptr<const ValueDescriptionT<T> > inner
                        = getDefaultDescriptionShared((T *)0))
        : inner(inner)
    {
    }

    std::shared_ptr<const ValueDescriptionT<T> > inner;

    typedef ValueDescription::FieldDescription FieldDescription;

    virtual void parseJson(void * val, JsonParsingContext & context) const
    {
        auto * val2 = reinterpret_cast<std::map<K, T> *>(val);
        return parseJsonTyped(val2, context);
    }

    virtual void parseJsonTyped(std::map<K, T> * val, JsonParsingContext & context) const
    {
        std::map<K, T> res;

        auto onMember = [&] ()
            {
                K key = KeyCodec::decode(context.fieldName(), (K *)0);
                inner->parseJsonTyped(&res[key], context);
            };

        context.forEachMember(onMember);

        val->swap(res);
    }

    virtual void printJson(const void * val, JsonPrintingContext & context) const
    {
        auto * val2 = reinterpret_cast<const std::map<K, T> *>(val);
        return printJsonTyped(val2, context);
    }

    virtual void printJsonTyped(const std::map<K, T> * val,
                                JsonPrintingContext & context) const
    {
        context.startObject();
        for (auto & v: *val) {
            context.startMember(KeyCodec::encode(v.first));
            inner->printJsonTyped(&v.second, context);
        }
        context.endObject();
    }

    virtual bool isDefault(const void * val) const
    {
        auto * val2 = reinterpret_cast<const std::map<K, T> *>(val);
        return isDefaultTyped(val2);
    }

    virtual bool isDefaultTyped(const std::map<K, T> * val) const
    {
        return val->empty();
    }

    virtual size_t getFieldCount(const void * val) const
    {
        auto * val2 = reinterpret_cast<const std::map<K, T> *>(val);
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

template<typename Key, typename Value>
struct DefaultDescription<std::map<Key, Value> >
    : public MapValueDescription<Key, Value> {
    DefaultDescription(ValueDescriptionT<Value> * inner)
        : MapValueDescription<Key, Value>(inner)
    {
    }

    DefaultDescription(std::shared_ptr<const ValueDescriptionT<Value> > inner
                       = getDefaultDescriptionShared((Value *)0))
        : MapValueDescription<Key, Value>(inner)
    {
    }

    DefaultDescription(ConstructOnly)
        : MapValueDescription<Key, Value>(constructOnly)
    {
    }
};

/** These functions allow it to be used to hold recursive data types. */

template<typename Key, typename Value>
DefaultDescription<std::map<Key, Value> > *
getDefaultDescriptionUninitialized(std::map<Key, Value> * = 0)
{
    return new DefaultDescription<std::map<Key, Value> >(constructOnly);
}

template<typename Key, typename Value>
void
initializeDefaultDescription(DefaultDescription<std::map<Key, Value> > & desc)
{
    desc = std::move(DefaultDescription<std::map<Key, Value> >());
}


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

    DescriptionFromBase(BaseDescription * inner)
        : inner(inner)
    {
    }

    DescriptionFromBase(std::shared_ptr<const BaseDescription> inner
                      = getDefaultDescriptionShared((Base *)0))
        : inner(inner)
    {
    }

    std::shared_ptr<const BaseDescription> inner;

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

    static auto desc = getDefaultDescriptionShared<T>();
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

    static auto desc = getDefaultDescriptionShared<T>();
    StreamingJsonParsingContext context(json, json.c_str(), json.c_str() + json.size());
    desc->parseJson(&result, context);
    return result;
}

// jsonDecode implementation for any type which:
// 1) has a default description;
// 2) does NOT have a fromJson() function (there is a simpler overload for this case)
template<typename T>
T jsonDecodeStream(std::istream & stream, T * = 0,
                   decltype(getDefaultDescription((T *)0)) * = 0,
                   typename std::enable_if<!hasFromJson<T>::value>::type * = 0)
{
    T result;

    static auto desc = getDefaultDescriptionShared<T>();
    StreamingJsonParsingContext context("<<input stream>>", stream);
    desc->parseJson(&result, context);
    return result;
}

// jsonDecode implementation for any type which:
// 1) has a default description;
// 2) does NOT have a fromJson() function (there is a simpler overload for this case)
template<typename T>
T jsonDecodeFile(const std::string & filename, T * = 0,
                 decltype(getDefaultDescription((T *)0)) * = 0,
                 typename std::enable_if<!hasFromJson<T>::value>::type * = 0)
{
    T result;
    
    ML::filter_istream stream(filename);
    
    static auto desc = getDefaultDescriptionShared<T>();
    StreamingJsonParsingContext context(filename, stream);
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

// In-place json decoding
template<typename T>
void jsonDecodeStream(std::istream & stream, T & val)
{
    val = std::move(jsonDecodeStream(stream, (T *)0));
}

// In-place json decoding
template<typename T>
void jsonDecodeFile(const std::string & filename, T & val)
{
    val = std::move(jsonDecodeFile(filename, (T *)0));
}

// jsonEncode implementation for any type which:
// 1) has a default description;
// 2) does NOT have a toJson() function (there is a simpler overload for this case)
template<typename T>
Json::Value jsonEncode(const T & obj,
                       decltype(getDefaultDescription((T *)0)) * = 0,
                       typename std::enable_if<!hasToJson<T>::value>::type * = 0)
{
    static auto desc = getDefaultDescriptionShared<T>();
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
    static auto desc = getDefaultDescriptionShared<T>();
    std::ostringstream stream;
    StreamJsonPrintingContext context(stream);
    desc->printJson(&obj, context);
    return std::move(stream.str());
}

// jsonEncode implementation for any type which:
// 1) has a default description;
// 2) does NOT have a toJson() function (there is a simpler overload for this case)
template<typename T>
std::ostream & jsonEncodeToStream(const T & obj,
                                  std::ostream & stream,
                                  decltype(getDefaultDescription((T *)0)) * = 0,
                                  typename std::enable_if<!hasToJson<T>::value>::type * = 0)
{
    static auto desc = getDefaultDescriptionShared<T>();
    StreamJsonPrintingContext context(stream);
    desc->printJson(&obj, context);
    return stream;
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
                                                                \
        Name(const Datacratic::ConstructOnly &)                 \
        {                                                       \
        }                                                       \
    };                                                          \
                                                                \
    inline Name *                                               \
    getDefaultDescription(Type *)                               \
    {                                                           \
        return new Name();                                      \
    }                                                           \
                                                                \
    inline Name *                                               \
    getDefaultDescriptionUninitialized(Type *)                  \
    {                                                           \
        return new Name(Datacratic::ConstructOnly());           \
    }                                                           \
                                                                \
    inline void initializeDefaultDescription(Name & desc)       \
    {                                                           \
        Name newDesc;                                           \
        desc = std::move(newDesc);                              \
    }                                                           \
                                                                \


#define DECLARE_STRUCTURE_DESCRIPTION_NAMED(Name, Type)         \
    struct Name                                                 \
        :  public StructureDescription<Type> {                  \
        Name();                                                 \
                                                                \
        Name(const Datacratic::ConstructOnly &);                \
                                                                \
        struct Regme;                                           \
        static Regme regme;                                     \
    };                                                          \
                                                                \
    Name *                                                      \
    getDefaultDescription(Type *);                              \
                                                                \
    Name *                                                      \
    getDefaultDescriptionUninitialized(Type *);                 \
                                                                \
    void initializeDefaultDescription(Name & desc);             \
                                                                \

#define DEFINE_STRUCTURE_DESCRIPTION_NAMED(Name, Type)          \
                                                                \
    struct Name::Regme {                                                \
        bool done;                                                      \
        Regme()                                                         \
            : done(false)                                               \
        {                                                               \
            Datacratic::registerValueDescription(typeid(Type), [] () { return new Name(); }, true); \
        }                                                               \
    };                                                                  \
                                                                        \
    Name::Name(const Datacratic::ConstructOnly &)                       \
    {                                                           \
        regme.done = true;                                      \
    }                                                           \
                                                                \
    Name *                                                      \
    getDefaultDescription(Type *)                               \
    {                                                           \
        return new Name();                                      \
    }                                                           \
                                                                \
    Name *                                                      \
    getDefaultDescriptionUninitialized(Type *)                  \
    {                                                           \
        return new Name(Datacratic::ConstructOnly());           \
    }                                                           \
                                                                \
    void initializeDefaultDescription(Name & desc)                      \
    {                                                                   \
        Name newDesc;                                                   \
        desc = std::move(newDesc);                                      \
    }                                                                   \
                                                                        \
    Name::Regme Name::regme;                                            \


    
#define CREATE_STRUCTURE_DESCRIPTION(Type)                      \
    CREATE_STRUCTURE_DESCRIPTION_NAMED(Type##Description, Type)

#define DECLARE_STRUCTURE_DESCRIPTION(Type)                      \
    DECLARE_STRUCTURE_DESCRIPTION_NAMED(Type##Description, Type)

#define DEFINE_STRUCTURE_DESCRIPTION(Type)                      \
    DEFINE_STRUCTURE_DESCRIPTION_NAMED(Type##Description, Type)


#define CREATE_CLASS_DESCRIPTION_NAMED(Name, Type)              \
    struct Name                                                 \
        : public Datacratic::StructureDescriptionImpl<Type, Name> { \
        Name() {                                                \
            Type::createDescription(*this);                     \
        }                                                       \
                                                                \
        Name(const Datacratic::ConstructOnly &)                 \
        {                                                       \
        }                                                       \
    };                                                          \
                                                                \
    inline Name *                                               \
    getDefaultDescription(Type *)                               \
    {                                                           \
        return new Name();                                      \
    }                                                           \
                                                                \
    inline Name *                                               \
    getDefaultDescriptionUninitialized(Type *)                  \
    {                                                           \
        return new Name(Datacratic::ConstructOnly());           \
    }                                                           \
                                                                \
    inline void initializeDefaultDescription(Name & desc)       \
    {                                                           \
        Name newDesc;                                           \
        desc = std::move(newDesc);                              \
    }                                                           \

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

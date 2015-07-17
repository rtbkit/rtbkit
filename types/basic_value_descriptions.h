/* basic_value_descriptions.h                                      -*- C++ -*-
   Jeremy Barnes, 20 March 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

*/

#pragma once

#include <limits>

#include "value_description.h"
#include "soa/types/url.h"
#include "soa/types/date.h"
#include "jml/utils/compact_vector.h"

namespace Datacratic {

/*****************************************************************************/
/* DEFAULT DESCRIPTIONS FOR BASIC TYPES                                      */
/*****************************************************************************/

template<>
struct DefaultDescription<Datacratic::Id>
    : public ValueDescriptionI<Datacratic::Id, ValueKind::ATOM> {

    virtual void parseJsonTyped(Datacratic::Id * val,
                                JsonParsingContext & context) const
    {
        Datacratic::parseJson(val, context);
    }

    virtual void printJsonTyped(const Datacratic::Id * val,
                                JsonPrintingContext & context) const
    {
        if (val->type == Id::Type::BIGDEC &&
            val->val2 == 0 && val->val1 <= std::numeric_limits<int32_t>::max()) {
            context.writeInt(val->val1);
        } else {
            context.writeStringUtf8(Utf8String(val->toString()));
        }
    }

    virtual bool isDefaultTyped(const Datacratic::Id * val) const
    {
        return !val->notNull();
    }
};

struct StringIdDescription: public DefaultDescription<Datacratic::Id> {

    virtual void printJsonTyped(const Datacratic::Id * val,
                                JsonPrintingContext & context) const
    {
        context.writeStringUtf8(Utf8String(val->toString()));
    }
};


template<>
struct DefaultDescription<std::string>
    : public ValueDescriptionI<std::string, ValueKind::STRING> {

    virtual void parseJsonTyped(std::string * val,
                                JsonParsingContext & context) const
    {
        *val = context.expectStringAscii();
    }

    virtual void printJsonTyped(const std::string * val,
                                JsonPrintingContext & context) const
    {
        context.writeString(*val);
    }

    virtual bool isDefaultTyped(const std::string * val) const
    {
        return val->empty();
    }
};

template<>
struct DefaultDescription<Utf8String>
    : public ValueDescriptionI<Utf8String, ValueKind::STRING> {

    virtual void parseJsonTyped(Utf8String * val,
                                JsonParsingContext & context) const
    {
        *val = context.expectStringUtf8();
    }

    virtual void printJsonTyped(const Utf8String * val,
                                JsonPrintingContext & context) const
    {
        context.writeStringUtf8(*val);
    }

    virtual bool isDefaultTyped(const Utf8String * val) const
    {
        return val->empty();
    }
};

template<>
struct DefaultDescription<Utf32String> :
    public ValueDescriptionI<Utf32String, ValueKind::STRING> {
        virtual void parseJsonTyped(Utf32String *val,
                                    JsonParsingContext & context) const
        {
            auto utf8Str = context.expectStringUtf8();
            *val = Utf32String::fromUtf8(utf8Str);
        }

        virtual void printJsonTyped(const Utf32String *val,
                                     JsonPrintingContext & context) const
        {
            std::string utf8Str;
            utf8::utf32to8(val->begin(), val->end(), std::back_inserter(utf8Str));
            context.writeStringUtf8(Utf8String { utf8Str });
        }

        virtual bool isDefaultTyped(const Utf32String *val) const
        {
            return val->empty();
        }
};

template<>
struct DefaultDescription<Url>
    : public ValueDescriptionI<Url, ValueKind::ATOM> {

    virtual void parseJsonTyped(Url * val,
                                JsonParsingContext & context) const
    {
        *val = Url(context.expectStringUtf8());
    }

    virtual void printJsonTyped(const Url * val,
                                JsonPrintingContext & context) const
    {
        context.writeStringUtf8(val->toUtf8String());
    }

    virtual bool isDefaultTyped(const Url * val) const
    {
        return val->empty();
    }
};

template<>
struct DefaultDescription<signed int>
    : public ValueDescriptionI<signed int, ValueKind::INTEGER> {

    virtual void parseJsonTyped(signed int * val,
                                JsonParsingContext & context) const
    {
        *val = context.expectInt();
    }
    
    virtual void printJsonTyped(const signed int * val,
                                JsonPrintingContext & context) const
    {
        context.writeInt(*val);
    }
};

template<>
struct DefaultDescription<unsigned int>
    : public ValueDescriptionI<unsigned int, ValueKind::INTEGER> {

    virtual void parseJsonTyped(unsigned int * val,
                                JsonParsingContext & context) const
    {
        *val = context.expectInt();
    }
    
    virtual void printJsonTyped(const unsigned int * val,
                                JsonPrintingContext & context) const
    {
        context.writeInt(*val);
    }
};

template<>
struct DefaultDescription<signed long>
    : public ValueDescriptionI<signed long, ValueKind::INTEGER> {

    virtual void parseJsonTyped(signed long * val,
                                JsonParsingContext & context) const
    {
        *val = context.expectLong();
    }
    
    virtual void printJsonTyped(const signed long * val,
                                JsonPrintingContext & context) const
    {
        context.writeLong(*val);
    }
};

template<>
struct DefaultDescription<unsigned long>
    : public ValueDescriptionI<unsigned long, ValueKind::INTEGER> {

    virtual void parseJsonTyped(unsigned long * val,
                                JsonParsingContext & context) const
    {
        *val = context.expectUnsignedLong();
    }
    
    virtual void printJsonTyped(const unsigned long * val,
                                JsonPrintingContext & context) const
    {
        context.writeUnsignedLong(*val);
    }
};

template<>
struct DefaultDescription<signed long long>
    : public ValueDescriptionI<signed long long, ValueKind::INTEGER> {

    virtual void parseJsonTyped(signed long long * val,
                                JsonParsingContext & context) const
    {
        *val = context.expectLongLong();
    }
    
    virtual void printJsonTyped(const signed long long * val,
                                JsonPrintingContext & context) const
    {
        context.writeLongLong(*val);
    }
};

template<>
struct DefaultDescription<unsigned long long>
    : public ValueDescriptionI<unsigned long long, ValueKind::INTEGER> {

    virtual void parseJsonTyped(unsigned long long * val,
                                JsonParsingContext & context) const
    {
        *val = context.expectUnsignedLongLong();
    }
    
    virtual void printJsonTyped(const unsigned long long * val,
                                JsonPrintingContext & context) const
    {
        context.writeUnsignedLongLong(*val);
    }
};

struct FloatValueDescription
    : public ValueDescriptionI<float, ValueKind::FLOAT> {

    virtual void parseJsonTyped(float * val,
                                JsonParsingContext & context) const
    {
        *val = context.expectFloat();
    }

    virtual void parseJson(void * val,
                           JsonParsingContext & context) const
    {
        *(float *)val = context.expectFloat();
    }

    virtual void printJsonTyped(const float * val,
                                JsonPrintingContext & context) const
    {
        context.writeFloat(*val);
    }
};

template<>
struct DefaultDescription<float>: public FloatValueDescription {
};

struct DoubleValueDescription
    : public ValueDescriptionI<double, ValueKind::FLOAT> {

    virtual void parseJsonTyped(double * val,
                                JsonParsingContext & context) const
    {
        *val = context.expectDouble();
    }

    virtual void printJsonTyped(const double * val,
                                JsonPrintingContext & context) const
    {
        context.writeDouble(*val);
    }
};

template<>
struct DefaultDescription<double>: public DoubleValueDescription {
};

#if 0
template<typename T>
struct DefaultDescription<std::vector<T> >
    : public ValueDescriptionI<std::vector<T>, ValueKind::ARRAY> {
};
#endif


template<typename T>
struct DefaultDescription<T*>
    : public ValueDescriptionI<T*, ValueKind::LINK> {

    DefaultDescription(ValueDescriptionT<T> * inner)
        : inner(inner)
    {
    }

    DefaultDescription(std::shared_ptr<const ValueDescriptionT<T> > inner
                       = getDefaultDescriptionShared((T *)0))
        : inner(inner)
    {
    }

    std::shared_ptr<const ValueDescriptionT<T> > inner;

    virtual void parseJsonTyped(T** val, JsonParsingContext & context) const
    {
        *val = new T();
        inner->parseJsonTyped(*val, context);
    }

    virtual void printJsonTyped(T* const* val, JsonPrintingContext & context) const
    {
        if (!*val)
            context.skip();
        else inner->printJsonTyped(*val, context);
    }

    virtual bool isDefaultTyped(T* const* val) const
    {
        return !*val;
    }

    virtual const ValueDescription & contained() const
    {
        return *this->inner;
    }


    virtual OwnershipModel getOwnershipModel() const
    {
        return OwnershipModel::NONE;
    }

    static T*& cast(void* obj)
    {
        return *static_cast<T**>(obj);
    }

    virtual void* getLink(void* obj) const
    {
        return cast(obj);
    }

    virtual void set(
            void* obj, void* value, const ValueDescription* valueDesc) const
    {
        if (valueDesc->kind != ValueKind::LINK)
            throw ML::Exception("assignment of non-link type to link type");

        valueDesc->contained().checkChildOf(&contained());
        cast(obj) = static_cast<T*>(valueDesc->getLink(value));
    }
};

template<typename T>
struct DefaultDescription<std::unique_ptr<T> >
    : public ValueDescriptionI<std::unique_ptr<T>, ValueKind::LINK> {

    DefaultDescription(ValueDescriptionT<T> * inner)
        : inner(inner)
    {
    }

    DefaultDescription(std::shared_ptr<const ValueDescriptionT<T> > inner
                       = getDefaultDescriptionShared((T *)0))
        : inner(inner)
    {
    }

    std::shared_ptr<const ValueDescriptionT<T> > inner;

    virtual void parseJsonTyped(std::unique_ptr<T> * val,
                                JsonParsingContext & context) const
    {
        val->reset(new T());
        inner->parseJsonTyped(val->get(), context);
    }

    virtual void printJsonTyped(const std::unique_ptr<T> * val,
                                JsonPrintingContext & context) const
    {
        if (!val->get())
            context.skip();
        else inner->printJsonTyped(val->get(), context);
    }

    virtual bool isDefaultTyped(const std::unique_ptr<T> * val) const
    {
        return !val->get();
    }

    virtual const ValueDescription & contained() const
    {
        return *this->inner;
    }

    virtual OwnershipModel getOwnershipModel() const
    {
        return OwnershipModel::UNIQUE;
    }

    static std::unique_ptr<T>& cast(void* obj)
    {
        return *static_cast< std::unique_ptr<T>* >(obj);
    }

    virtual void* getLink(void* obj) const
    {
        return cast(obj).get();
    }

    virtual void set(
            void* obj, void* value, const ValueDescription* valueDesc) const
    {
        if (valueDesc->kind != ValueKind::LINK)
            throw ML::Exception("assignment of non-link type to link type");

        if (valueDesc->getOwnershipModel() != OwnershipModel::NONE)
            throw ML::Exception("unsafe link assignement");

        valueDesc->contained().checkChildOf(&contained());
        cast(obj).reset(static_cast<T*>(valueDesc->getLink(value)));
    }
};

template<typename T>
struct DefaultDescription<std::shared_ptr<T> >
    : public ValueDescriptionI<std::shared_ptr<T>, ValueKind::LINK> {

    DefaultDescription(std::shared_ptr<const ValueDescriptionT<T> > inner
                       = getDefaultDescriptionShared((T *)0))
        : inner(inner)
    {
    }

    DefaultDescription(ValueDescriptionT<T> * inner)
        : inner(inner)
    {
    }

    std::shared_ptr<const ValueDescriptionT<T> > inner;

    virtual void parseJsonTyped(std::shared_ptr<T> * val,
                                JsonParsingContext & context) const
    {
        if (context.isNull()) {
            val->reset();
            context.expectNull();
            return;
        }
        val->reset(new T());
        inner->parseJsonTyped(val->get(), context);
    }

    virtual void printJsonTyped(const std::shared_ptr<T> * val,
                                JsonPrintingContext & context) const
    {
        if (!val->get())
            context.skip();
        else inner->printJsonTyped(val->get(), context);
    }

    virtual bool isDefaultTyped(const std::shared_ptr<T> * val) const
    {
        return !val->get();
    }

    virtual const ValueDescription & contained() const
    {
        return *this->inner;
    }

    virtual OwnershipModel getOwnershipModel() const
    {
        return OwnershipModel::SHARED;
    }


    static std::shared_ptr<T>& cast(void* obj)
    {
        return *static_cast< std::shared_ptr<T>* >(obj);
    }

    virtual void* getLink(void* obj) const
    {
        return cast(obj).get();
    }

    virtual void set(
            void* obj, void* value, const ValueDescription* valueDesc) const
    {
        if (valueDesc->kind != ValueKind::LINK)
            throw ML::Exception("assignment of non-link type to link type");

        if (valueDesc->getOwnershipModel() == OwnershipModel::UNIQUE)
            throw ML::Exception("unsafe link assignement");

        valueDesc->contained().checkChildOf(&contained());

        // Casting is necessary to make sure the ref count is incremented.
        if (valueDesc->getOwnershipModel() == OwnershipModel::SHARED)
            cast(obj) = cast(value);
        else cast(obj).reset(static_cast<T*>(valueDesc->getLink(value)));
    }

};

template<>
struct DefaultDescription<Json::Value>
    : public ValueDescriptionI<Json::Value, ValueKind::ANY> {

    virtual void parseJsonTyped(Json::Value * val,
                                JsonParsingContext & context) const
    {
        Datacratic::parseJson(val, context);
    }

    virtual void printJsonTyped(const Json::Value * val,
                                JsonPrintingContext & context) const
    {
        context.writeJson(*val);
    }

    virtual bool isDefaultTyped(const Json::Value * val) const
    {
        return val->isNull();
    }
};

template<>
struct DefaultDescription<bool>
    : public ValueDescriptionI<bool, ValueKind::BOOLEAN> {

    virtual void parseJsonTyped(bool * val,
                                JsonParsingContext & context) const
    {
        *val = context.expectBool();
    }

    virtual void printJsonTyped(const bool * val,
                                JsonPrintingContext & context) const
    {
        context.writeBool(*val);
    }

    virtual bool isDefaultTyped(const bool * val) const
    {
        return false;
    }
};

template<>
struct DefaultDescription<Date>
    : public ValueDescriptionI<Date, ValueKind::ATOM> {

    virtual void parseJsonTyped(Date * val,
                                JsonParsingContext & context) const
    {
        if (context.isNumber())
            *val = Date::fromSecondsSinceEpoch(context.expectDouble());
        else if (context.isString()) {
            std::string s = context.expectStringAscii();
            if (s.length() >= 11
                && s[4] == '-'
                && s[7] == '-'
                && (s[s.size() - 1] == 'Z'
                    || s[s.size() - 3] == ':'))
                *val = Date::parseIso8601DateTime(s);
            else *val = Date::parseDefaultUtc(s);
        }
        else context.exception("expected date");
    }

    virtual void printJsonTyped(const Date * val,
                                JsonPrintingContext & context) const
    {
        context.writeJson(val->printIso8601());//val->secondsSinceEpoch());
    }

    virtual bool isDefaultTyped(const Date * val) const
    {
        return *val == Date();
    }
};

struct JavaTimestampValueDescription: public DefaultDescription<Date> {

    virtual void parseJsonTyped(Date * val,
                                JsonParsingContext & context) const
    {
        *val = Date::fromSecondsSinceEpoch(context.expectDouble() * 0.001);
    }

    virtual void printJsonTyped(const Date * val,
                                JsonPrintingContext & context) const
    {
        context.writeJson((uint64_t)(val->secondsSinceEpoch() * 1000));
    }
};

struct Iso8601TimestampValueDescription: public DefaultDescription<Date> {

    virtual void parseJsonTyped(Date * val,
                                JsonParsingContext & context) const
    {
        if (context.isNumber())
            *val = Date::fromSecondsSinceEpoch(context.expectDouble());
        else if (context.isString())
            *val = Date::parseIso8601DateTime(context.expectStringAscii());
        else context.exception("expected date");
    }

    virtual void printJsonTyped(const Date * val,
                                JsonPrintingContext & context) const
    {
        context.writeString(val->printIso8601());
    }
};

template<typename T, typename U>
struct DefaultDescription<std::pair<T, U> >
    : public ValueDescriptionI<std::pair<T, U>, ValueKind::ARRAY> {

    DefaultDescription(ValueDescriptionT<T> * inner1,
                       ValueDescriptionT<U> * inner2)
        : inner1(inner1), inner2(inner2)
    {
    }

    DefaultDescription(std::shared_ptr<const ValueDescriptionT<T> > inner1
                       = getDefaultDescriptionShared((T *)0),
                       std::shared_ptr<const ValueDescriptionT<U> > inner2
                       = getDefaultDescriptionShared((U *)0))
        : inner1(inner1), inner2(inner2)
    {
    }

    std::shared_ptr<const ValueDescriptionT<T> > inner1;
    std::shared_ptr<const ValueDescriptionT<U> > inner2;

    virtual void parseJsonTyped(std::pair<T, U> * val,
                                JsonParsingContext & context) const
    {
        int el = 0;
        auto onElement = [&] ()
            {
                if (el == 0)
                    inner1->parseJsonTyped(&val->first, context);
                else if (el == 1)
                    inner2->parseJsonTyped(&val->second, context);
                else context.exception("expected 2 element array");

                ++el;
            };

        context.forEachElement(onElement);

        if (el != 2)
            context.exception("expected 2 element array");
    }

    virtual void printJsonTyped(const std::pair<T, U> * val,
                                JsonPrintingContext & context) const
    {
        context.startArray(2);
        context.newArrayElement();
        inner1->printJsonTyped(&val->first, context);
        context.newArrayElement();
        inner2->printJsonTyped(&val->second, context);
        context.endArray();
    }

    virtual bool isDefaultTyped(const std::pair<T, U> * val) const
    {
        return inner1->isDefaultTyped(&val->first)
            && inner2->isDefaultTyped(&val->second);
    }
};

template<typename T>
struct Optional: public std::unique_ptr<T> {
    Optional()
    {
    }
    
    Optional(Optional && other)
        : std::unique_ptr<T>(std::move(other))
    {
    }

    Optional(const Optional & other)
    {
        if (other)
            this->reset(new T(*other));
    }

    Optional & operator = (const Optional & other)
    {
        Optional newMe(other);
        swap(newMe);
        return *this;
    }

    Optional & operator = (Optional && other)
    {
        Optional newMe(other);
        swap(newMe);
        return *this;
    }

    bool operator == (Optional && other) const
    {
        return (this->get() == other.get()
                || (this->get() != nullptr
                    && other.get != nullptr
                    && *this == *other));
    }

    void swap(Optional & other)
    {
        std::unique_ptr<T>::swap(other);
    }

    template<typename... Args>
    void emplace(Args&&... args)
    {
        this->reset(new T(std::forward<Args>(args)...));
    }
};

template<typename Cls, int defValue = -1>
struct TaggedEnum {
    TaggedEnum(int v = defValue)
        : val(v)
    {
    }

    int val;

    int value() const
    {
        return val;
    }

    /// This typedef allows it to be picked up by a default value description
    typedef void isTaggedEnumType;

#if 0
    operator typename Cls::Vals () const
    {
        return static_cast<typename Cls::Vals>(val);
    }
#endif
};

template<typename E, int def>
bool operator == (const TaggedEnum<E, def> & e1, const TaggedEnum<E, def> & e2)
{
    return e1.val == e2.val;
}

template<typename E, int def>
bool operator != (const TaggedEnum<E, def> & e1, const TaggedEnum<E, def> & e2)
{
    return e1.val != e2.val;
}

template<typename E, int def>
bool operator > (const TaggedEnum<E, def> & e1, const TaggedEnum<E, def> & e2)
{
    return e1.val > e2.val;
}

template<typename E, int def>
bool operator < (const TaggedEnum<E, def> & e1, const TaggedEnum<E, def> & e2)
{
    return e1.val < e2.val;
}

template<typename E, int def>
inline Json::Value jsonPrint(const TaggedEnum<E, def> & e)
{
    return e.val;
}

template<typename E, int def>
inline void jsonParse(const Json::Value & j, TaggedEnum<E, def> & e)
{
    e.val = j.asInt();
}

struct TaggedBool {
    TaggedBool()
        : val(-1)
    {
    }

    TaggedBool(bool v)
        : val(v)
    {
    }

    int val;
};

template<int defValue = -1>
struct TaggedBoolDef : public TaggedBool {
    TaggedBoolDef()
        : val(defValue)
    {
    }

    TaggedBoolDef(bool v)
        : val(v)
    {
    }

    int val;
};

struct TaggedInt {
    TaggedInt(int v = -1)
        : val(v)
    {
    }

    int value() const { return val; }

    int val;
};

template<int defValue = -1>
struct TaggedIntDef : TaggedInt {
    TaggedIntDef(int v = defValue)
        : val(v)
    {
    }

    int val;
};

struct TaggedInt64 {
    TaggedInt64(int64_t v = -1)
        : val(v)
    {
    }

    int64_t value() const { return val; }

    int64_t val;
};

template<int64_t defValue = -1>
struct TaggedInt64Def : TaggedInt {
    TaggedInt64Def(int64_t v = defValue)
        : val(v)
    {
    }

    int64_t val;
};

struct TaggedFloat {
    TaggedFloat(float v = std::numeric_limits<float>::quiet_NaN())
        : val(v)
    {
    }

    float val;
};

template<int num = -1, int den = 1>
struct TaggedFloatDef : public TaggedFloat {
    TaggedFloatDef(float v = 1.0f * num / den)
        : val(v)
    {
    }

    float val;
};

struct TaggedDouble {
    TaggedDouble(double v = std::numeric_limits<double>::quiet_NaN())
        : val(v)
    {
    }

    double val;
};

template<int num = -1, int den = 1>
struct TaggedDoubleDef : public TaggedDouble {
    TaggedDoubleDef(double v = 1.0 * num / den)
        : val(v)
    {
    }

    double val;
};

template<>
struct DefaultDescription<TaggedBool>
    : public ValueDescriptionI<TaggedBool, ValueKind::BOOLEAN> {
  
    virtual void parseJsonTyped(TaggedBool * val,
                                JsonParsingContext & context) const
    {
        if (context.isBool())
            val->val = context.expectBool();
        else val->val = context.expectInt();
    }

    virtual void printJsonTyped(const TaggedBool * val,
                                JsonPrintingContext & context) const
    {
        context.writeInt(val->val);
    }

    virtual bool isDefaultTyped(const TaggedBool * val)
        const
    {
        return val->val == -1;
    }
};

template<int defValue>
struct DefaultDescription<TaggedBoolDef<defValue> >
    : public ValueDescriptionI<TaggedBoolDef<defValue>,
                               ValueKind::BOOLEAN> {
  
    virtual void parseJsonTyped(TaggedBoolDef<defValue> * val,
                                JsonParsingContext & context) const
    {
        if (context.isBool())
            val->val = context.expectBool();
        else val->val = context.expectInt();
    }

    virtual void printJsonTyped(const TaggedBoolDef<defValue> * val,
                                JsonPrintingContext & context) const
    {
        context.writeInt(val->val);
    }

    virtual bool isDefaultTyped(const TaggedBoolDef<defValue> * val)
        const
    {
        return val->val == defValue;
    }
};

template<>
struct DefaultDescription<TaggedInt>
    : public ValueDescriptionI<TaggedInt,
                               ValueKind::INTEGER,
                               DefaultDescription<TaggedInt> > {

    virtual void parseJsonTyped(TaggedInt * val,
                                JsonParsingContext & context) const
    {
        if (context.isString()) {
            std::string s = context.expectStringAscii();
            val->val = boost::lexical_cast<int>(s);
        }
        else val->val = context.expectInt();
    }

    virtual void printJsonTyped(const TaggedInt * val,
                                JsonPrintingContext & context) const
    {
        context.writeInt(val->val);
    }

    virtual bool isDefaultTyped(const TaggedInt * val)
        const
    {
        return val->val == -1;
    }
};

template<int defValue>
struct DefaultDescription<TaggedIntDef<defValue> >
    : public ValueDescriptionI<TaggedIntDef<defValue>,
                               ValueKind::INTEGER> {

    virtual void parseJsonTyped(TaggedIntDef<defValue> * val,
                                JsonParsingContext & context) const
    {
        if (context.isString()) {
            std::string s = context.expectStringAscii();
            val->val = boost::lexical_cast<int>(s);
        }
        else val->val = context.expectInt();
    }

    virtual void printJsonTyped(const TaggedIntDef<defValue> * val,
                                JsonPrintingContext & context) const
    {
        context.writeInt(val->val);
    }

    virtual bool isDefaultTyped(const TaggedIntDef<defValue> * val)
        const
    {
        return val->val == defValue;
    }
};

template<>
struct DefaultDescription<TaggedInt64>
: public ValueDescriptionI<TaggedInt64,
                           ValueKind::INTEGER,
                           DefaultDescription<TaggedInt64> > {
    
    virtual void parseJsonTyped(TaggedInt64 * val,
                                JsonParsingContext & context) const
    {
        if (context.isString()) {
            std::string s = context.expectStringAscii();
            val->val = boost::lexical_cast<int64_t>(s);
        }
        else val->val = context.expectLongLong();
    }

    virtual void printJsonTyped(const TaggedInt64 * val,
                                JsonPrintingContext & context) const
    {
        context.writeLongLong(val->val);
    }

    virtual bool isDefaultTyped(const TaggedInt64 * val) const
    {
        return val->val == -1;
    }
};

template<int64_t defValue>
struct DefaultDescription<TaggedInt64Def<defValue> >
  : public ValueDescriptionI<TaggedInt64Def<defValue>, ValueKind::INTEGER > {

    virtual void parseJsonTyped(TaggedInt64Def<defValue> * val,
                                JsonParsingContext & context) const
    {
        if (context.isString()) {
            std::string s = context.expectStringAscii();
            val->val = boost::lexical_cast<int64_t>(s);
        }
        else val->val = context.expectLongLong();
    }

    virtual void printJsonTyped(const TaggedInt64Def<defValue> * val,
                                JsonPrintingContext & context) const
    {
        context.writeLongLong(val->val);
    }

    virtual bool isDefaultTyped(const TaggedInt64Def<defValue> * val) const
    {
        return val->val == defValue;
    }
};

template<>
struct DefaultDescription<TaggedFloat>
    : public ValueDescriptionI<TaggedFloat,
                               ValueKind::FLOAT> {

    virtual void parseJsonTyped(TaggedFloat * val,
                                JsonParsingContext & context) const
    {
        val->val = context.expectFloat();
    }

    virtual void printJsonTyped(const TaggedFloat * val,
                                JsonPrintingContext & context) const
    {
        context.writeDouble(val->val);
    }

    virtual bool isDefaultTyped(const TaggedFloat * val) const
    {
        return isnan(val->val);
    }
};

template<int num, int den>
struct DefaultDescription<TaggedFloatDef<num, den> >
    : public ValueDescriptionI<TaggedFloatDef<num, den>,
                               ValueKind::FLOAT> {

    virtual void parseJsonTyped(TaggedFloatDef<num, den> * val,
                                JsonParsingContext & context) const
    {
        val->val = context.expectFloat();
    }

    virtual void printJsonTyped(const TaggedFloatDef<num, den> * val,
                                JsonPrintingContext & context) const
    {
        context.writeFloat(val->val);
    }

    virtual bool isDefaultTyped(const TaggedFloatDef<num, den> * val) const
    {
        return val->val == (float)num / den;
    }
};

template<>
struct DefaultDescription<TaggedDouble>
    : public ValueDescriptionI<TaggedDouble,
                               ValueKind::FLOAT> {

    virtual void parseJsonTyped(TaggedDouble * val,
                                JsonParsingContext & context) const
    {
        val->val = context.expectDouble();
    }

    virtual void printJsonTyped(const TaggedDouble * val,
                                JsonPrintingContext & context) const
    {
        context.writeDouble(val->val);
    }

    virtual bool isDefaultTyped(const TaggedDouble * val) const
    {
        return std::isnan(val->val);
    }
};

template<int num, int den>
struct DefaultDescription<TaggedDoubleDef<num, den> >
    : public ValueDescriptionI<TaggedDoubleDef<num, den>,
                               ValueKind::FLOAT> {

    virtual void parseJsonTyped(TaggedDoubleDef<num, den> * val,
                                JsonParsingContext & context) const
    {
        val->val = context.expectDouble();
    }

    virtual void printJsonTyped(const TaggedDoubleDef<num, den> * val,
                                JsonPrintingContext & context) const
    {
        context.writeDouble(val->val);
    }

    virtual bool isDefaultTyped(const TaggedDoubleDef<num, den> * val) const
    {
        return val->val == (double)num / den;
    }
};

template<class Enum>
struct TaggedEnumDescription
    : public ValueDescriptionT<Enum> {

    TaggedEnumDescription()
        : ValueDescriptionT<Enum>(ValueKind::ENUM)
    {
    }

    virtual void parseJsonTyped(Enum * val,
                                JsonParsingContext & context) const
    {
        int index = context.expectInt();
        val->val = index;
    }

    virtual void printJsonTyped(const Enum * val,
                                JsonPrintingContext & context) const
    {
        context.writeInt(val->val);
    }

    virtual bool isDefaultTyped(const Enum * val) const
    {
        return val->val == -1;
    }
};

/** Default description for anything that has a tagged enum but no default
    description.
*/

template<typename Enum>
TaggedEnumDescription<Enum> *
getDefaultDescription(Enum *,
                      typename Enum::isTaggedEnumType * = 0,
                      typename DefaultDescription<Enum>::not_defined * = 0)
{
    return new TaggedEnumDescription<Enum>();
}

/*****************************************************************************/
/* DEFAULT DESCRIPTION FOR COMPACT VECTOR                                    */
/*****************************************************************************/

template<typename T, int Internal>
struct DefaultDescription<ML::compact_vector<T, Internal> >
    : public ValueDescriptionI<ML::compact_vector<T, Internal>, ValueKind::ARRAY>,
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
        ML::compact_vector<T, Internal> * val2 = reinterpret_cast<ML::compact_vector<T, Internal> *>(val);
        return parseJsonTyped(val2, context);
    }

    virtual void parseJsonTyped(ML::compact_vector<T, Internal> * val, JsonParsingContext & context) const
    {
        this->parseJsonTypedList(val, context);
    }

    virtual void printJson(const void * val, JsonPrintingContext & context) const
    {
        const ML::compact_vector<T, Internal> * val2 = reinterpret_cast<const ML::compact_vector<T, Internal> *>(val);
        return printJsonTyped(val2, context);
    }

    virtual void printJsonTyped(const ML::compact_vector<T, Internal> * val, JsonPrintingContext & context) const
    {
        this->printJsonTypedList(val, context);
    }

    virtual bool isDefault(const void * val) const
    {
        const ML::compact_vector<T, Internal> * val2 = reinterpret_cast<const ML::compact_vector<T, Internal> *>(val);
        return isDefaultTyped(val2);
    }

    virtual bool isDefaultTyped(const ML::compact_vector<T, Internal> * val) const
    {
        return val->empty();
    }

    virtual size_t getArrayLength(void * val) const
    {
        const ML::compact_vector<T, Internal> * val2 = reinterpret_cast<const ML::compact_vector<T, Internal> *>(val);
        return val2->size();
    }

    virtual void * getArrayElement(void * val, uint32_t element) const
    {
        ML::compact_vector<T, Internal> * val2 = reinterpret_cast<ML::compact_vector<T, Internal> *>(val);
        return &val2->at(element);
    }

    virtual const void * getArrayElement(const void * val, uint32_t element) const
    {
        const ML::compact_vector<T, Internal> * val2 = reinterpret_cast<const ML::compact_vector<T, Internal> *>(val);
        return &val2->at(element);
    }

    virtual void setArrayLength(void * val, size_t newLength) const
    {
        ML::compact_vector<T, Internal> * val2 = reinterpret_cast<ML::compact_vector<T, Internal> *>(val);
        val2->resize(newLength);
    }
    
    virtual const ValueDescription & contained() const
    {
        return *this->inner;
    }
};

typedef Utf8String CSList;  // comma-separated list

template<typename T>
struct List: public ML::compact_vector<T, 3> {
};

/// This can either be:
/// single format: "w": 123
/// multiple formats: "w": [ 123, 456 ]
struct FormatListDescription
    : public ValueDescriptionI<List<int> >,
      public ListDescriptionBase<int> {

    virtual void parseJsonTyped(List<int> * val,
                                JsonParsingContext & context) const
    {
        if (context.isArray()) {
            auto onElement = [&] ()
                {
                    val->push_back(context.expectInt());
                };
            context.forEachElement(onElement);
        }
        else {
            val->push_back(context.expectInt());
        }
    }

    virtual void printJsonTyped(const List<int> * val,
                                JsonPrintingContext & context) const
    {
        if (val->size() == 1) {
            this->inner->printJsonTyped(&(*val)[0], context);
        }
        else
            printJsonTypedList(val, context);
    }

    virtual bool isDefaultTyped(const List<int> * val) const
    {
        return val->empty();
    }

};

struct CommaSeparatedListDescription
    : public ValueDescriptionI<std::string, ValueKind::STRING> {

    virtual void parseJsonTyped(std::string * val,
                                JsonParsingContext & context) const
    {
        if (context.isArray()) {
            std::string res;
            auto onElement = [&] ()
                {
                    std::string s = context.expectStringAscii();
                    if (!res.empty())
                        res += ", ";
                    res += s;
                        
                };
            context.forEachElement(onElement);
            *val = res;
        }
        else {
            *val = context.expectStringAscii();
        }
    }

    virtual void printJsonTyped(const std::string * val,
                                JsonPrintingContext & context) const
    {
        context.writeString(*val);
    }

    virtual bool isDefaultTyped(const std::string * val) const
    {
        return val->empty();
    }

};


struct Utf8CommaSeparatedListDescription
    : public ValueDescriptionI<Utf8String, ValueKind::STRING> {

    virtual void parseJsonTyped(Utf8String * val,
                                JsonParsingContext & context) const
    {
        if (context.isArray()) {
            Utf8String res;
            auto onElement = [&] ()
                {
                    Utf8String s = context.expectStringUtf8();
                    if (!res.empty())
                        res += ", ";
                    res += s;
                };

            context.forEachElement(onElement);
            *val = res;
        }
        else {
            *val = context.expectStringUtf8();
        }
    }

    virtual void printJsonTyped(const Utf8String * val,
                                JsonPrintingContext & context) const
    {
        context.writeStringUtf8(*val);
    }

    virtual bool isDefaultTyped(const Utf8String * val) const
    {
        return val->empty();
    }

};

template<typename T>
struct DefaultDescription<Optional<T> >
    : public ValueDescriptionI<Optional<T>, ValueKind::OPTIONAL> {

    DefaultDescription(ValueDescriptionT<T> * inner)
        : inner(inner)
    {
    }

    DefaultDescription(std::shared_ptr<const ValueDescriptionT<T> > inner
                       = getDefaultDescriptionShared((T *)0))
        : inner(inner)
    {
    }

    std::shared_ptr<const ValueDescriptionT<T> > inner;

    virtual void parseJsonTyped(Optional<T> * val,
                                JsonParsingContext & context) const
    {
        if (context.isNull()) {
            context.expectNull();
            val->reset();
            return;
        }
        val->reset(new T());
        inner->parseJsonTyped(val->get(), context);
    }

    virtual void printJsonTyped(const Optional<T> * val,
                                JsonPrintingContext & context) const
    {
        if (!val->get())
            context.skip();
        else inner->printJsonTyped(val->get(), context);
    }

    virtual bool isDefaultTyped(const Optional<T> * val) const
    {
        return !val->get();
    }

    virtual void * optionalMakeValueTyped(Optional<T> * val) const
    {
        if (!val->get())
            val->reset(new T());
        return val->get();
    }

    virtual const void * optionalGetValueTyped(const Optional<T> * val) const
    {
        if (!val->get())
            throw ML::Exception("no value in optional field");
        return val->get();
    }

    virtual const ValueDescription & contained() const
    {
        return *inner;
    }
};

template<typename T>
struct DefaultDescription<List<T> >
    : public ValueDescriptionI<List<T>, ValueKind::ARRAY>,
      public ListDescriptionBase<T> {

    virtual void parseJsonTyped(List<T> * val,
                                JsonParsingContext & context) const
    {
        this->parseJsonTypedList(val, context);
    }

    virtual void printJsonTyped(const List<T> * val,
                                JsonPrintingContext & context) const
    {
        this->printJsonTypedList(val, context);
    }

    virtual bool isDefaultTyped(const List<T> * val) const
    {
        return val->empty();
    }

    virtual size_t getArrayLength(void * val) const
    {
        const List<T> * val2 = reinterpret_cast<const List<T> *>(val);
        return val2->size();
    }

    virtual void * getArrayElement(void * val, uint32_t element) const
    {
        List<T> * val2 = reinterpret_cast<List<T> *>(val);
        return &val2->at(element);
    }

    virtual const void * getArrayElement(const void * val, uint32_t element) const
    {
        const List<T> * val2 = reinterpret_cast<const List<T> *>(val);
        return &val2->at(element);
    }

    virtual void setArrayLength(void * val, size_t newLength) const
    {
        List<T> * val2 = reinterpret_cast<List<T> *>(val);
        val2->resize(newLength);
    }
    
    virtual const ValueDescription & contained() const
    {
        return *this->inner;
    }
};

} // namespace Datacratic

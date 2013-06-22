/* fbx_parsing.h                                               -*- C++ -*-
   Jean-Sebastien Bejeau, 19 June 2013

   Code to parse FBX bid requests.
*/

#pragma once

#include "soa/types/value_description.h"
#include "soa/types/basic_value_descriptions.h"
#include "soa/types/json_parsing.h"
#include "fbx.h"
#include <boost/lexical_cast.hpp>

namespace Datacratic {

template<>
struct DefaultDescription<FBX::TaggedBool>
    : public ValueDescriptionI<FBX::TaggedBool, ValueKind::BOOLEAN> {
  
    virtual void parseJsonTyped(FBX::TaggedBool * val,
                                JsonParsingContext & context) const
    {
        if (context.isBool())
            val->val = context.expectBool();
        else val->val = context.expectInt();
    }

    virtual void printJsonTyped(const FBX::TaggedBool * val,
                                JsonPrintingContext & context) const
    {
        context.writeInt(val->val);
    }

    virtual bool isDefaultTyped(const FBX::TaggedBool * val)
        const
    {
        return val->val == -1;
    }
};

template<int defValue>
struct DefaultDescription<FBX::TaggedBoolDef<defValue> >
    : public ValueDescriptionI<FBX::TaggedBoolDef<defValue>,
                               ValueKind::BOOLEAN> {
  
    virtual void parseJsonTyped(FBX::TaggedBoolDef<defValue> * val,
                                JsonParsingContext & context) const
    {
        if (context.isBool())
            val->val = context.expectBool();
        else val->val = context.expectInt();
    }

    virtual void printJsonTyped(const FBX::TaggedBoolDef<defValue> * val,
                                JsonPrintingContext & context) const
    {
        context.writeInt(val->val);
    }

    virtual bool isDefaultTyped(const FBX::TaggedBoolDef<defValue> * val)
        const
    {
        return val->val == defValue;
    }
};

template<>
struct DefaultDescription<FBX::TaggedInt>
    : public ValueDescriptionI<FBX::TaggedInt,
                               ValueKind::INTEGER,
                               DefaultDescription<FBX::TaggedInt> > {

    virtual void parseJsonTyped(FBX::TaggedInt * val,
                                JsonParsingContext & context) const
    {
        if (context.isString()) {
            std::string s = context.expectStringAscii();
            val->val = boost::lexical_cast<int>(s);
        }
        else val->val = context.expectInt();
    }

    virtual void printJsonTyped(const FBX::TaggedInt * val,
                                JsonPrintingContext & context) const
    {
        context.writeInt(val->val);
    }

    virtual bool isDefaultTyped(const FBX::TaggedInt * val)
        const
    {
        return val->val == -1;
    }
};

template<int defValue>
struct DefaultDescription<FBX::TaggedIntDef<defValue> >
    : public ValueDescriptionI<FBX::TaggedIntDef<defValue>,
                               ValueKind::INTEGER> {

    virtual void parseJsonTyped(FBX::TaggedIntDef<defValue> * val,
                                JsonParsingContext & context) const
    {
        if (context.isString()) {
            std::string s = context.expectStringAscii();
            val->val = boost::lexical_cast<int>(s);
        }
        else val->val = context.expectInt();
    }

    virtual void printJsonTyped(const FBX::TaggedIntDef<defValue> * val,
                                JsonPrintingContext & context) const
    {
        context.writeInt(val->val);
    }

    virtual bool isDefaultTyped(const FBX::TaggedIntDef<defValue> * val)
        const
    {
        return val->val == defValue;
    }
};


template<class Enum>
struct TaggedEnumDescription
    : public ValueDescriptionI<Enum, ValueKind::ENUM,
                               TaggedEnumDescription<Enum> > {

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

template<typename T>
struct DefaultDescription<FBX::Optional<T> >
    : public ValueDescriptionI<FBX::Optional<T>, ValueKind::OPTIONAL> {

    DefaultDescription(ValueDescriptionT<T> * inner
                       = getDefaultDescription((T *)0))
        : inner(inner)
    {
    }

    std::unique_ptr<ValueDescriptionT<T> > inner;

    virtual void parseJsonTyped(FBX::Optional<T> * val,
                                JsonParsingContext & context) const
    {
        if (context.isNull()) {
            context.expectNull();
            val->reset();
        }
        val->reset(new T());
        inner->parseJsonTyped(val->get(), context);
    }

    virtual void printJsonTyped(const FBX::Optional<T> * val,
                                JsonPrintingContext & context) const
    {
        if (!val->get())
            context.skip();
        else inner->printJsonTyped(val->get(), context);
    }

    virtual bool isDefaultTyped(const FBX::Optional<T> * val) const
    {
        return !val->get();
    }

    virtual void * optionalMakeValueTyped(FBX::Optional<T> * val) const
    {
        if (!val->get())
            val->reset(new T());
        return val->get();
    }

    virtual const void * optionalGetValueTyped(const FBX::Optional<T> * val) const
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
struct DefaultDescription<FBX::List<T> >
    : public ValueDescriptionI<FBX::List<T>, ValueKind::ARRAY>,
      public ListDescriptionBase<T> {

    virtual void parseJsonTyped(FBX::List<T> * val,
                                JsonParsingContext & context) const
    {
        this->parseJsonTypedList(val, context);
    }

    virtual void printJsonTyped(const FBX::List<T> * val,
                                JsonPrintingContext & context) const
    {
        this->printJsonTypedList(val, context);
    }

    virtual bool isDefaultTyped(const FBX::List<T> * val) const
    {
        return val->empty();
    }

    virtual size_t getArrayLength(void * val) const
    {
        const FBX::List<T> * val2 = reinterpret_cast<const FBX::List<T> *>(val);
        return val2->size();
    }

    virtual void * getArrayElement(void * val, uint32_t element) const
    {
        FBX::List<T> * val2 = reinterpret_cast<FBX::List<T> *>(val);
        return &val2->at(element);
    }

    virtual const void * getArrayElement(const void * val, uint32_t element) const
    {
        const FBX::List<T> * val2 = reinterpret_cast<const FBX::List<T> *>(val);
        return &val2->at(element);
    }

    virtual void setArrayLength(void * val, size_t newLength) const
    {
        FBX::List<T> * val2 = reinterpret_cast<FBX::List<T> *>(val);
        val2->resize(newLength);
    }
    
    virtual const ValueDescription & contained() const
    {
        return *this->inner;
    }

};


template<>
struct DefaultDescription<FBX::BidRequest>
    : public StructureDescription<FBX::BidRequest> {
    DefaultDescription();
};

template<>
struct DefaultDescription<FBX::PageTypeCode>
    : public StructureDescription<FBX::PageTypeCode> {
    DefaultDescription()
    {
    }
};

template<>
struct DefaultDescription<FBX::RtbUserContext>
    : public StructureDescription<FBX::RtbUserContext> {
    DefaultDescription();
};

template<>
struct DefaultDescription<FBX::RtbPageContext>
    : public StructureDescription<FBX::RtbPageContext> {
    DefaultDescription();
};



} // namespace Datacratic

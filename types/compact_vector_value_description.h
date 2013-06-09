
#include "value_description.h"
#include "jml/utils/compact_vector.h"

#pragma once

namespace Datacratic {

template<typename T, size_t I, typename S, bool Sf, typename P, typename A>
struct DefaultDescription<ML::compact_vector<T, I, S, Sf, P, A> >
    : public ValueDescriptionI<ML::compact_vector<T, I, S, Sf, P, A>, ValueKind::ARRAY>,
      public ListDescriptionBase<T> {

    DefaultDescription(ValueDescriptionT<T> * inner
                      = getDefaultDescription((T *)0))
        : ListDescriptionBase<T>(inner)
    {
    }

    virtual void parseJson(void * val, JsonParsingContext & context) const
    {
        ML::compact_vector<T, I, S, Sf, P, A> * val2 = reinterpret_cast<ML::compact_vector<T, I, S, Sf, P, A> *>(val);
        return parseJsonTyped(val2, context);
    }

    virtual void parseJsonTyped(ML::compact_vector<T, I, S, Sf, P, A> * val, JsonParsingContext & context) const
    {
        this->parseJsonTypedList(val, context);
    }

    virtual void printJson(const void * val, JsonPrintingContext & context) const
    {
        const ML::compact_vector<T, I, S, Sf, P, A> * val2 = reinterpret_cast<const ML::compact_vector<T, I, S, Sf, P, A> *>(val);
        return printJsonTyped(val2, context);
    }

    virtual void printJsonTyped(const ML::compact_vector<T, I, S, Sf, P, A> * val, JsonPrintingContext & context) const
    {
        this->printJsonTypedList(val, context);
    }

    virtual bool isDefault(const void * val) const
    {
        const ML::compact_vector<T, I, S, Sf, P, A> * val2 = reinterpret_cast<const ML::compact_vector<T, I, S, Sf, P, A> *>(val);
        return isDefaultTyped(val2);
    }

    virtual bool isDefaultTyped(const ML::compact_vector<T, I, S, Sf, P, A> * val) const
    {
        return val->empty();
    }

    virtual size_t getArrayLength(void * val) const
    {
        const ML::compact_vector<T, I, S, Sf, P, A> * val2 = reinterpret_cast<const ML::compact_vector<T, I, S, Sf, P, A> *>(val);
        return val2->size();
    }

    virtual void * getArrayElement(void * val, uint32_t element) const
    {
        ML::compact_vector<T, I, S, Sf, P, A> * val2 = reinterpret_cast<ML::compact_vector<T, I, S, Sf, P, A> *>(val);
        return &val2->at(element);
    }

    virtual const void * getArrayElement(const void * val, uint32_t element) const
    {
        const ML::compact_vector<T, I, S, Sf, P, A> * val2 = reinterpret_cast<const ML::compact_vector<T, I, S, Sf, P, A> *>(val);
        return &val2->at(element);
    }

    virtual void setArrayLength(void * val, size_t newLength) const
    {
        ML::compact_vector<T, I, S, Sf, P, A> * val2 = reinterpret_cast<ML::compact_vector<T, I, S, Sf, P, A> *>(val);
        val2->resize(newLength);
    }
    
    virtual const ValueDescription & contained() const
    {
        return *this->inner;
    }
};

} // namespace Datacratic

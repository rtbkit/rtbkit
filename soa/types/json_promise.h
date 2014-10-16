/* json_promise.h                                                  -*- C++ -*-
   Jeremy Barnes, 22 January 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

   A class that contains something that is promised to be JSON.  Used to
   defer JSON processing until needed to avoid unnecessary transformations.
*/

#pragma once

namespace Datacratic {


/*****************************************************************************/
/* JSON PROMISE                                                              */
/*****************************************************************************/

struct JsonPromise {

    JsonPromise() noexcept
        : structured(0), serialized(0), value(0)
    {
    }

    JsonPromise(const Json::Value & val)
        : structured(new Json::Value(val)), serialized(0), value(0)
    {
    }

    JsonPromise(Json::Value && val)
        : structured(new Json::Value(std::move(val))), serialized(0), value(0)
    {
    }

    JsonPromise(const std::string & str)
        : structured(0), serialized(new std::string(str)), value(0)
    {
    }

    JsonPromise(std::string && str)
        : structured(0), serialized(new std::string(std::move(str))), value(0)
    {
    }

    JsonPromise(const JsonPromise & other) = delete;
    void operator = (const JsonPromise & other) = delete;

    JsonPromise(JsonPromise && other) noexcept
        : structured(other.structured),
          serialized(other.serialized),
          value(other.value)
    {
        other.structured = nullptr;
        other.serialized = nullptr;
        other.value = nullptr;
    }

    JsonPromise & operator = (JsonPromise && other) noexcept
    {
        JsonPromise newMe(std::move(other));
        swap(newMe);
        return *this;
    }

    void swap(JsonPromise & other) noexcept
    {
        std::swap(structured, other.structured);
        std::swap(serialized, other.serialized);
        std::swap(value, other.value);
    }

    template<typename T>
    static void copiedDeleter(const ValueDescription * desc,
                              void * obj_) noexcept
    {
        T * obj = reinterpret_cast<T *>(obj_);
        delete obj;
    }

    template<typename T>
    JsonPromise(const T & val,
                const ValueDescription<T> & desc
                    = *getDefaultDescriptionShared<T>())
        : structured(0), serialized(0),
          value(new Value(new T(val), &desc, &copiedDeleter<T>))
    {
    }
    
    template<typename T>
    JsonPromise(T && val,
                const ValueDescription<T> & desc
                    = *getDefaultDescriptionShared<T>())
        : structured(0), serialized(0),
          value(new Value(new T(std::move(val(), &desc, &copiedDeleter<T>))))
    {
    }
    
    ~JsonPromise()
    {
        if (structured)
            delete structured;
        if (serialized)
            delete serialized;
        if (value) {
            if (value->deleter)
                value->deleter(value->desc, value->val);
            delete value;
        }
    }

    std::ostream & dump(std::ostream & stream)
    {
        if (serialized)
            return stream << *serialized;
        else if (value) {
            StreamingJsonPrintingContext context(stream);
            value->desc->printJson(value->val, context);
            return stream << *value;
        }
        else if (structured)
            return stream << *structured;
        else return stream << "null";
    }

    const std::string & asString() const
    {
        if (serialized)
            return *serialized;
        if (value) {
            std::ostringstream stream;
            StreamingJsonPrintingContext context(stream);
            value->desc->printJson(value->val, context);
            serialized = new std::string(std::move(stream.str()));
            return *serialized;
        }
        if (structured) {
            serialized = new std::string(std::move(structured->toString()));
            return *serialized;
        }
        return "null";
    }

    const Json::Value & asJson() const
    {
        if (structured)
            return *structured;
        if (value) {
            StructuredJsonPrintingContext context;
            value->desc->printJson(value->val, context);
            structured = new Json::Value(std::move(context.value));
            return *structured;
        }
        if (serialized) {
            structured = new Json::Value(std::move(Json::parse(*serialized)));
            return *structured;
        }
        
        return Json::Value();  // null
    }
    
    template<typename T>
    const T & asValue(const ValueDescription<T> & desc
                      = *getDescriptionShared<T>())
    {
        if (!value)
            createValue(desc);
        ExcAssert(value);
        ExcAssert(value->val);
        return *(T *)(value->val);
    }
    
    operator Json::Value ()
    {
        return asJson();
    }

    operator std::string ()
    {
        return asString();
    }

    template<typename T>
    operator T ()
    {
        return asValue<T>();
    }

private:
    std::unique_ptr<Json::Value> structured;
    std::unique_ptr<std::string> serialized;

    void createValue(const ValueDescription & desc)
    {
        if (value) return;
        if (structured) {
            StructuredJsonParsingContext context(*structured);
            std::unique_ptr<T> obj(new T());
            desc.parseJson(obj.get(), context);

            value.reset(new Value(&desc, obj.release(), &copiedDeleter<T>()));
            Value value;
            value.desc = desc;
            value.deleter = &copiedDeleter<T>;
            value.val = obj.release();
        }
        if (serialized) {
        }
    }
    
    struct Value {
        ~Value()
        {
        }
        const ValueDescription * desc;
        void * val;
        void (const ValueDescription *, void *) deleter;
    };
    
    std::unique_ptr<Value> value;
};

} // namespace Datacratic

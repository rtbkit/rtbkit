/* json_parsing.h                                                  -*- C++ -*-
   Jeremy Barnes, 22 February 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

*/

#pragma once

#include "soa/jsoncpp/json.h"
#include "jml/utils/json_parsing.h"
#include "soa/types/id.h"

namespace Datacratic {

template<typename T>
struct ValueDescription;

struct JsonParsingContext;

struct JsonParser {
    virtual ~JsonParser();
    virtual void parse(void * output, JsonParsingContext & context) const = 0;
};

/** For any value description, build a JSON parser for it. */
JsonParser * createJsonParser(const ValueDescription<void> & desc);

struct JsonPathEntry {
    JsonPathEntry(int index)
        : index(index), fieldNumber(0)
    {
    }
    
    JsonPathEntry(std::string key)
        : index(-1), key(key), fieldNumber(0)
    {
    }
    
    int index;
    std::string key;
    int fieldNumber;

};

struct JsonPath: public std::vector<JsonPathEntry> {
    std::string print() const
    {
        std::string result;
        for (auto & e: *this) {
            if (e.index == -1)
                result += "." + e.key;
            else result += '[' + std::to_string(e.index) + ']';
        }
        return result;
    }

    std::string fieldName() const
    {
        return this->back().key;
    }

    void push(const JsonPathEntry & entry)
    {
        this->push_back(entry);
    }

    void replace(const JsonPathEntry & entry)
    {
        int newFieldNumber = this->back().fieldNumber + 1;
        this->back() = entry;
        this->back().fieldNumber = newFieldNumber;
    }

    void pop()
    {
        this->pop_back();
    }

};

struct JsonParsingContext {

    JsonPath path;
    std::vector<JsonParser *> parsers;

    std::string printPath() const
    {
        return path.print();
    }

    std::string fieldName() const
    {
        return path.fieldName();
    }

    void pushPath(const JsonPathEntry & entry)
    {
        path.push(entry);
    }

    void replacePath(const JsonPathEntry & entry)
    {
        path.replace(entry);
    }

    void popPath()
    {
        path.pop();
    }

    typedef std::function<void ()> OnUnknownField;

    std::vector<OnUnknownField> onUnknownFieldHandlers;

    void onUnknownField()
    {
        if (!onUnknownFieldHandlers.empty())
            onUnknownFieldHandlers.back()();
        else exception("unknown field " + printPath());
    }

    /** Handler for when we get an undexpected field. */

    virtual void exception(const std::string & message) = 0;
    
    virtual int expectInt() = 0;
    virtual float expectFloat() = 0;
    virtual float expectBool() = 0;
    virtual bool matchUnsignedLongLong(unsigned long long & val) = 0;
    virtual bool matchLongLong(long long & val) = 0;
    virtual std::string expectStringAscii() = 0;
    virtual Json::Value expectJson() = 0;
    virtual bool isObject() const = 0;
    virtual bool isString() const = 0;
    virtual bool isArray() const = 0;
    virtual bool isBool() const = 0;
#if 0
    virtual bool isNull() const = 0;
    virtual bool isNumber() const = 0;
    virtual bool isInt() const = 0;
#endif
    virtual void skip() = 0;

    virtual void forEachMember(std::function<void ()> fn) = 0;
    virtual void forEachElement(std::function<void ()> fn) = 0;
};

struct StreamingJsonParsingContext
    : public JsonParsingContext  {

    StreamingJsonParsingContext()
    {
    }

    template<typename... Args>
    StreamingJsonParsingContext(Args &&... args)
    {
        init(std::forward<Args>(args)...);
    }

    template<typename... Args>
    void init(Args &&... args)
    {
        ownedContext.reset(new ML::Parse_Context(std::forward<Args>(args)...));
        context = ownedContext.get();
    }

    void init(ML::Parse_Context & context)
    {
        this->context = &context;
        ownedContext.reset();
    }

    ML::Parse_Context * context;
    std::unique_ptr<ML::Parse_Context> ownedContext;

    template<typename Fn>
    void forEachMember(Fn fn)
    {
        bool first = true;

        auto onMember = [&] (const std::string & memberName,
                             ML::Parse_Context &)
            {
                if (first)
                    pushPath(memberName);
                else replacePath(memberName);

                fn();

                first = false;
            };
        
        expectJsonObject(*context, onMember);

        if (!first)
            popPath();
    }

    virtual void forEachMember(std::function<void ()> fn)
    {
        return forEachMember<std::function<void ()> >(fn);
    }

    template<typename Fn>
    void forEachElement(Fn fn)
    {
        bool first = true;

        auto onElement = [&] (int index, ML::Parse_Context &)
            {
                if (first)
                    pushPath(index);
                else replacePath(index);

                fn();

                first = false;
            };
        
        expectJsonArray(*context, onElement);

        if (!first)
            popPath();
    }

    virtual void forEachElement(std::function<void ()> fn)
    {
        return forEachElement<std::function<void ()> >(fn);
    }

    void skip()
    {
        ML::expectJson(*context);
    }

    virtual int expectInt()
    {
        return context->expect_int();
    }

    virtual float expectFloat()
    {
        return context->expect_float();
    }

    virtual float expectBool()
    {
        return ML::expectJsonBool(*context);
    }

    virtual bool matchUnsignedLongLong(unsigned long long & val)
    {
        return context->match_unsigned_long_long(val);
    }

    virtual bool matchLongLong(long long & val)
    {
        return context->match_long_long(val);
    }

    virtual std::string expectStringAscii()
    {
        return expectJsonStringAscii(*context);
    }

    virtual bool isObject() const
    {
        char c = *(*context);
        return c == '{';
    }

    virtual bool isString() const
    {
        char c = *(*context);
        return c == '\"';
    }

    virtual bool isArray() const
    {
        char c = *(*context);
        return c == '[';
    }

    virtual bool isBool() const
    {
        char c = *(*context);
        return c == 't' || c == 'f';
        
    }

    virtual void exception(const std::string & message)
    {
        context->exception(message);
    }

#if 0
    virtual bool isNull() const
    {
        Revert_Token token(*context);
        if (match_literal("null"))
            return true;
        return false;
    }

    virtual bool isNumber() const
    {
        Revert_Token token(*context);
        double d;
        if (match_double(d))
            return true;
        return false;
    }
    
    virtual bool isInt() const
    {
        Revert_Token token(*context);
        long long l;
        if (match_long_long(l))
            return true;
        return false;
    }
#endif

    virtual Json::Value expectJson()
    {
        return ML::expectJson(*context);
    }
};

struct StructuredJsonParsingContext: public JsonParsingContext {
    Json::Value toParse;
};


template<typename Context>
void parseJson(int * output, Context & context)
{
    *output = context.expect_int();
}

template<typename Context>
void parseJson(float * output, Context & context)
{
    *output = context.expect_float();
}

template<typename Context>
void parseJson(double * output, Context & context)
{
    *output = context.expect_double();
}

template<typename Context>
void parseJson(Id * output, Context & context)
{
    using namespace std;

    unsigned long long i;
    if (context.matchUnsignedLongLong(i)) {
        cerr << "got unsigned " << i << endl;
        *output = Id(i);
        return;
    }

    signed long long l;
    if (context.matchLongLong(l)) {
        cerr << "got signed " << l << endl;
        *output = Id(l);
        return;
    }

    std::string s = context.expectStringAscii();
    *output = Id(s);
}

template<typename Context, typename T>
void parseJson(std::vector<T> * output, Context & context)
{
    throw ML::Exception("vector not done");
}

template<typename Context>
void parseJson(Json::Value * output, Context & context)
{
    *output = context.expectJson();
}

} // namespace Datacratic

/* json_parsing.h                                                  -*- C++ -*-
   Jeremy Barnes, 22 February 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

*/

#pragma once

#include "soa/jsoncpp/json.h"
#include "jml/utils/json_parsing.h"
#include "jml/utils/parse_context.h"
#include "jml/utils/compact_vector.h"
#include "soa/types/id.h"
#include "soa/types/string.h"
#include <boost/algorithm/string.hpp>


namespace Datacratic {

struct JsonParsingContext;
struct ValueDescription;

struct JsonPathEntry {
    JsonPathEntry(int index)
        : index(index), keyStr(0), keyPtr(0), fieldNumber(0)
    {
    }
    
    JsonPathEntry(const std::string & key)
        : index(-1), keyStr(new std::string(key)), keyPtr(keyStr->c_str()),
          fieldNumber(0)
    {
    }
    
    JsonPathEntry(const char * keyPtr)
        : index(-1), keyStr(nullptr), keyPtr(keyPtr), fieldNumber(0)
    {
    }

    JsonPathEntry(JsonPathEntry && other) noexcept
    {
        *this = std::move(other);
    }

    JsonPathEntry & operator = (JsonPathEntry && other) noexcept
    {
        index = other.index;
        keyPtr = other.keyPtr;
        keyStr = other.keyStr;
        fieldNumber = other.fieldNumber;
        other.keyStr = nullptr;
        other.keyPtr = nullptr;
        return *this;
    }

    ~JsonPathEntry()
    {
        if (keyStr)
            delete keyStr;
    }

    int index;
    std::string * keyStr;
    const char * keyPtr;
    int fieldNumber;

    std::string fieldName() const
    {
        return keyStr ? *keyStr : std::string(keyPtr);
    }

    const char * fieldNamePtr() const
    {
        return keyPtr;
    }

    // Needed for compilers that don't support move_if_noexcept
    JsonPathEntry & operator = (const JsonPathEntry & other)
    {
        index = other.index;
        if (other.keyStr) {
            keyStr = new std::string(*other.keyStr);
            keyPtr = keyStr->c_str();
        }
        else {
            keyPtr = other.keyPtr;
            keyStr = nullptr;
        }
        fieldNumber = other.fieldNumber;
        return *this;
    }

    // Needed for compilers that don't support move_if_noexcept
    JsonPathEntry(const JsonPathEntry & other)
    {
        *this = other;
    }
};

struct JsonPath: public ML::compact_vector<JsonPathEntry, 8> {
    JsonPath()
    {
    }

    std::string print() const
    {
        std::string result;
        for (auto & e: *this) {
            if (e.index == -1)
                result += "." + std::string(e.fieldName());
            else result += '[' + std::to_string(e.index) + ']';
        }
        return result;
    }

    std::string fieldName() const
    {
        return this->back().fieldName();
    }

    const char * fieldNamePtr() const
    {
        return this->back().fieldNamePtr();
    }

    void push(JsonPathEntry entry, int fieldNum = 0)
    {
        entry.fieldNumber = fieldNum;
        this->emplace_back(std::move(entry));
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

/*****************************************************************************/
/* JSON PARSING CONTEXT                                                      */
/*****************************************************************************/

struct JsonParsingContext {

    JsonPath path;

    std::string printPath() const
    {
        return path.print();
    }

    std::string fieldName() const
    {
        return path.fieldName();
    }

    const char * fieldNamePtr() const
    {
        return path.fieldNamePtr();
    }

    void pushPath(const JsonPathEntry & entry, int memberNumber = 0)
    {
        path.push(entry, memberNumber);
    }

    void replacePath(const JsonPathEntry & entry)
    {
        path.replace(entry);
    }

    void popPath()
    {
        path.pop();
    }

    typedef std::function<void (const ValueDescription * desc)> OnUnknownField;

    std::vector<OnUnknownField> onUnknownFieldHandlers;

    void onUnknownField(const ValueDescription * desc = 0);

    /** Handler for when we get an undexpected field. */

    virtual void exception(const std::string & message) = 0;

    /** Return a string that gives the context of where the parsing is
        at, for example line number and column.
    */
    virtual std::string getContext() const = 0;
    
    virtual int expectInt() = 0;
    virtual unsigned int expectUnsignedInt() = 0;
    virtual long expectLong() = 0;
    virtual unsigned long expectUnsignedLong() = 0;
    virtual long long expectLongLong() = 0;
    virtual unsigned long long expectUnsignedLongLong() = 0;

    virtual float expectFloat() = 0;
    virtual double expectDouble() = 0;
    virtual bool expectBool() = 0;
    virtual bool matchUnsignedLongLong(unsigned long long & val) = 0;
    virtual bool matchLongLong(long long & val) = 0;
    virtual bool matchDouble(double & val) = 0;
    virtual std::string expectStringAscii() = 0;
    virtual ssize_t expectStringAscii(char * value, size_t maxLen) = 0;
    virtual Utf8String expectStringUtf8() = 0;
    virtual Json::Value expectJson() = 0;
    virtual void expectNull() = 0;
    virtual bool isObject() const = 0;
    virtual bool isString() const = 0;
    virtual bool isArray() const = 0;
    virtual bool isBool() const = 0;
    virtual bool isNumber() const = 0;
    virtual bool isNull() const = 0;
#if 0
    virtual bool isInt() const = 0;
#endif
    virtual void skip() = 0;

    /** For debugging: print out what is the currently being parsed
        element.  No guarantees about what it actually prints; that
        depends on the .
    */
    virtual std::string printCurrent() = 0;
    
    virtual void forEachMember(const std::function<void ()> & fn) = 0;
    virtual void forEachElement(const std::function<void ()> & fn) = 0;
};


/*****************************************************************************/
/* STREAMING JSON PARSING CONTEXT                                            */
/*****************************************************************************/

/** This object allows you to parse a stream (string, file, std::istream)
    containing JSON data into an object without performing an intermediate
    translation into a structured JSON format.  This tends to be a lot
    faster as far fewer memory allocations are required.
*/

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
    void forEachMember(const Fn & fn)
    {
        int memberNum = 0;

        auto onMember = [&] (const char * memberName,
                             ML::Parse_Context &)
            {
                // This structure takes care of pushing and popping our
                // path entry.  It will make sure the member is always
                // popped no matter what
                struct PathPusher {
                    PathPusher(const char * memberName,
                               int memberNum,
                               StreamingJsonParsingContext * context)
                        : context(context)
                    {
                        context->pushPath(memberName, memberNum);
                    }

                    ~PathPusher()
                    {
                        context->popPath();
                    }

                    StreamingJsonParsingContext * const context;
                } pusher(memberName, memberNum++, this);

                fn();
            };
        
        expectJsonObjectAscii(*context, onMember);
    }

    virtual void forEachMember(const std::function<void ()> & fn)
    {
        return forEachMember<std::function<void ()> >(fn);
    }

    template<typename Fn>
    void forEachElement(const Fn & fn)
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

    virtual void forEachElement(const std::function<void ()> & fn)
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

    virtual unsigned int expectUnsignedInt()
    {
        return context->expect_unsigned();
    }

    virtual long expectLong()
    {
        return context->expect_long();
    }

    virtual unsigned long expectUnsignedLong()
    {
        return context->expect_unsigned_long();
    }

    virtual long long expectLongLong()
    {
        return context->expect_long_long();
    }

    virtual unsigned long long expectUnsignedLongLong()
    {
        return context->expect_unsigned_long_long();
    }

    virtual float expectFloat()
    {
        return context->expect_float();
    }

    virtual double expectDouble()
    {
        return context->expect_double();
    }

    virtual bool expectBool()
    {
        return ML::expectJsonBool(*context);
    }

    virtual void expectNull()
    {
        context->expect_literal("null");
    }

    virtual bool matchUnsignedLongLong(unsigned long long & val)
    {
        return context->match_unsigned_long_long(val);
    }

    virtual bool matchLongLong(long long & val)
    {
        return context->match_long_long(val);
    }

    virtual bool matchDouble(double & val)
    {
        return context->match_double(val);
    }

    virtual std::string expectStringAscii()
    {
        return expectJsonStringAscii(*context);
    }

    virtual ssize_t expectStringAscii(char * value, size_t maxLen)
    {
        return expectJsonStringAscii(*context, value, maxLen);
    }

    virtual Utf8String expectStringUtf8();

    virtual bool isObject() const
    {
        skipJsonWhitespace(*context);
        char c = *(*context);
        return c == '{';
    }

    virtual bool isString() const
    {
        skipJsonWhitespace(*context);
        char c = *(*context);
        return c == '\"';
    }

    virtual bool isArray() const
    {
        skipJsonWhitespace(*context);
        char c = *(*context);
        return c == '[';
    }

    virtual bool isBool() const
    {
        skipJsonWhitespace(*context);
        char c = *(*context);
        return c == 't' || c == 'f';
        
    }

    virtual bool isNumber() const
    {
        skipJsonWhitespace(*context);
        ML::Parse_Context::Revert_Token token(*context);
        double d;
        if (context->match_double(d))
            return true;
        return false;
    }

    virtual bool isNull() const
    {
        skipJsonWhitespace(*context);
        ML::Parse_Context::Revert_Token token(*context);
        if (context->match_literal("null"))
            return true;
        return false;
    }

#if 0    
    virtual bool isNumber() const
    {
        char c = *(*context);
        if (c >= '0' && c <= '9')
            return true;
        if (c == '.' || c == '+' || c == '-')
            return true;
        if (c == 'N' || c == 'I')  // NaN or Inf
            return true;
        return false;
    }
#endif

    virtual void exception(const std::string & message)
    {
        context->exception("at " + printPath() + ": " + message);
    }

    virtual std::string getContext() const
    {
        return context->where() + " at " + printPath();
    }

#if 0
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

    virtual std::string printCurrent()
    {
        try {
            ML::Parse_Context::Revert_Token token(*context);
            return boost::trim_copy(expectJson().toString());
        } catch (const std::exception & exc) {
            ML::Parse_Context::Revert_Token token(*context);
            return context->expect_text("\n");
        }
    }
};

struct StructuredJsonParsingContext: public JsonParsingContext {

    StructuredJsonParsingContext(const Json::Value & val)
        : current(&val), top(&val)
    {
    }

    const Json::Value * current;
    const Json::Value * top;

    virtual void exception(const std::string & message)
    {
        //using namespace std;
        //cerr << *current << endl;
        //cerr << *top << endl;
        throw ML::Exception("At path " + printPath() + ": "
                            + message + " parsing "
                            + boost::trim_copy(top->toString()));
    }
    
    virtual std::string getContext() const
    {
        return printPath();
    }

    virtual int expectInt()
    {
        return current->asInt();
    }

    virtual unsigned int expectUnsignedInt()
    {
        return current->asUInt();
    }

    virtual long expectLong()
    {
        return current->asInt();
    }

    virtual unsigned long expectUnsignedLong()
    {
        return current->asUInt();
    }

    virtual long long expectLongLong()
    {
        return current->asInt();
    }

    virtual unsigned long long expectUnsignedLongLong()
    {
        return current->asUInt();
    }

    virtual float expectFloat()
    {
        return current->asDouble();
    }

    virtual double expectDouble()
    {
        return current->asDouble();
    }

    virtual bool expectBool()
    {
        return current->asBool();
    }

    virtual void expectNull()
    {
        if (!current->isNull())
            exception("expected null value");
    }

    virtual bool matchUnsignedLongLong(unsigned long long & val)
    {
        if (current->isIntegral()) {
            val = current->asUInt();
            return true;
        }
        if (current->isNumeric()) {
            unsigned long long v = current->asDouble();
            if (v == current->asDouble()) {
                val = v;
                return true;
            }
        }
        return false;
    }

    virtual bool matchLongLong(long long & val)
    {
        if (current->isIntegral()) {
            val = current->asInt();
            return true;
        }
        if (current->isNumeric()) {
            long long v = current->asDouble();
            if (v == current->asDouble()) {
                val = v;
                return true;
            }
        }
        return false;
    }

    virtual bool matchDouble(double & val)
    {
        if (current->isNumeric()) {
            val = current->asDouble();
            return true;
        }
        return false;
    }

    virtual std::string expectStringAscii()
    {
        return current->asString();
    }

    virtual ssize_t expectStringAscii(char * value, size_t maxLen)
    {
        const std::string & strValue = current->asString();
        ssize_t realSize = strValue.size();
        if (realSize >= maxLen) {
            return -1;
        }
        memcpy(value, strValue.c_str(), realSize);
        value[realSize] = '\0';
        return realSize;
    }

    virtual Utf8String expectStringUtf8()
    {
        return Utf8String(current->asString());
    }

    virtual Json::Value expectJson()
    {
        return *current;
    }

    virtual bool isObject() const
    {
        return current->type() == Json::objectValue;
    }

    virtual bool isString() const
    {
        return current->type() == Json::stringValue;
    }

    virtual bool isArray() const
    {
        return current->type() == Json::arrayValue;
    }

    virtual bool isBool() const
    {
        return current->type() == Json::booleanValue;
    }

    virtual bool isNumber() const
    {
        return current->isNumeric();
    }

    virtual bool isNull() const
    {
        return current->isNull();
    }

    virtual void skip()
    {
    }

    virtual void forEachMember(const std::function<void ()> & fn)
    {
        if (!isObject())
            exception("expected an object");

        const Json::Value * oldCurrent = current;
        int memberNum = 0;

        for (auto it = current->begin(), end = current->end();
             it != end;  ++it) {

            // This structure takes care of pushing and popping our
            // path entry.  It will make sure the member is always
            // popped no matter what
            struct PathPusher {
                PathPusher(const std::string & memberName,
                           int memberNum,
                           StructuredJsonParsingContext * context)
                    : context(context)
                {
                    context->pushPath(memberName, memberNum);
                }

                ~PathPusher()
                {
                    context->popPath();
                }

                StructuredJsonParsingContext * const context;
            } pusher(it.memberName(), memberNum++, this);
            
            current = &(*it);
            fn();
        }
        
        current = oldCurrent;
    }

    virtual void forEachElement(const std::function<void ()> & fn)
    {
        if (!isArray())
            exception("expected an array");

        const Json::Value * oldCurrent = current;

        for (unsigned i = 0;  i < oldCurrent->size();  ++i) {
            if (i == 0)
                pushPath(i);
            else replacePath(i);

            current = &(*oldCurrent)[i];

            fn();
        }

        if (oldCurrent->size() != 0)
            popPath();
        
        current = oldCurrent;
    }

    virtual std::string printCurrent()
    {
        return boost::trim_copy(current->toString());
    }
};


/*****************************************************************************/
/* STRING JSON PARSING CONTEXT                                               */
/*****************************************************************************/

struct StringJsonParsingContext
    : public StreamingJsonParsingContext  {

    StringJsonParsingContext(std::string str_,
                             const std::string & filename = "<<internal>>")
        : str(std::move(str_))
    {
        init(filename, str.c_str(), str.c_str() + str.size());
    }

    std::string str;
};


/*****************************************************************************/
/* UTILITIES                                                                 */
/*****************************************************************************/

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

    if (context.isString()) {
        Utf8String value = context.expectStringUtf8();
        *output = Id(value.rawString());
        return;
    }

    unsigned long long i;
    if (context.matchUnsignedLongLong(i)) {
        // cerr << "got unsigned " << i << endl;
        *output = Id(i);
        return;
    }

    signed long long l;
    if (context.matchLongLong(l)) {
        // cerr << "got signed " << l << endl;
        *output = Id(l);
        return;
    }

    double d;
    if (context.matchDouble(d)) {
        if ((long long)d != d)
            context.exception("IDs must be integers");
        *output = Id((long long)d);
        return;
    }

    if (context.isNull()) {
        context.expectNull();
        *output = Id();
        output->type = Id::NULLID;
        return;
    }

    std::cerr << context.expectJson() << endl;

    throw ML::Exception("unhandled id conversion type");
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

/* json_printing.h                                                 -*- C++ -*-
   Jeremy Barnes, 26 February 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Context to print out JSON.
*/

#pragma once

#include <string>
#include <ostream>

#include "jml/utils/exc_assert.h"
#include "jml/utils/json_parsing.h"

#include "soa/jsoncpp/value.h"
#include "soa/types/string.h"


namespace Datacratic {


/*****************************************************************************/
/* JSON PRINTING CONTEXT                                                     */
/*****************************************************************************/

struct JsonPrintingContext {

    virtual ~JsonPrintingContext()
    {
    }

    virtual void startObject() = 0;
    virtual void startMember(const std::string & memberName) = 0;
    virtual void endObject() = 0;

    virtual void startArray(int knownSize = -1) = 0;
    virtual void newArrayElement() = 0;
    virtual void endArray() = 0;

    virtual void writeInt(int i) = 0;
    virtual void writeUnsignedInt(unsigned i) = 0;
    virtual void writeLong(long i) = 0;
    virtual void writeUnsignedLong(unsigned long i) = 0;
    virtual void writeLongLong(long long i) = 0;
    virtual void writeUnsignedLongLong(unsigned long long i) = 0;
    virtual void writeFloat(float f) = 0;
    virtual void writeDouble(double d) = 0;
    virtual void writeString(const std::string & s) = 0;
    virtual void writeStringUtf8(const Utf8String & s) = 0;
    virtual void writeBool(bool b) = 0;
    virtual void writeNull() = 0;

    virtual void writeJson(const Json::Value & val) = 0;
    virtual void skip() = 0;
};


/*****************************************************************************/
/* STREAM JSON PRINTING CONTEXT                                              */
/*****************************************************************************/

struct StreamJsonPrintingContext
    : public JsonPrintingContext {

    StreamJsonPrintingContext(std::ostream & stream)
        : stream(stream), writeUtf8(true)
    {
    }

    std::ostream & stream;
    bool writeUtf8;          ///< If true, utf8 chars in binary.  False: escaped ASCII

    struct PathEntry {
        PathEntry(bool isObject)
            : isObject(isObject), memberNum(-1)
        {
        }

        bool isObject;
        std::string memberName;
        int memberNum;
    };

    std::vector<PathEntry> path;

    virtual void startObject()
    {
        path.push_back(true /* isObject */);
        stream << "{";
    }

    virtual void startMember(const std::string & memberName)
    {
        ExcAssert(path.back().isObject);
        //path.back().memberName = memberName;
        ++path.back().memberNum;
        if (path.back().memberNum != 0)
            stream << ",";
        stream << '\"';
        ML::jsonEscape(memberName, stream);
        stream << "\":";
    }

    virtual void endObject()
    {
        ExcAssert(path.back().isObject);
        path.pop_back();
        stream << "}";
    }

    virtual void startArray(int knownSize = -1)
    {
        path.push_back(false /* isObject */);
        stream << "[";
    }

    virtual void newArrayElement()
    {
        ExcAssert(!path.back().isObject);
        ++path.back().memberNum;
        if (path.back().memberNum != 0)
            stream << ",";
    }

    virtual void endArray()
    {
        ExcAssert(!path.back().isObject);
        path.pop_back();
        stream << "]";
    }
    
    virtual void skip()
    {
        stream << "null";
    }

    virtual void writeNull()
    {
        stream << "null";
    }

    virtual void writeInt(int i)
    {
        stream << i;
    }

    virtual void writeUnsignedInt(unsigned int i)
    {
        stream << i;
    }

    virtual void writeLong(long int i)
    {
        stream << i;
    }

    virtual void writeUnsignedLong(unsigned long int i)
    {
        stream << i;
    }

    virtual void writeLongLong(long long int i)
    {
        stream << i;
    }

    virtual void writeUnsignedLongLong(unsigned long long int i)
    {
        stream << i;
    }

    virtual void writeFloat(float f)
    {
        if (std::isfinite(f))
            stream << f;
        else stream << "\"" << f << "\"";
    }

    virtual void writeDouble(double d)
    {
        if (std::isfinite(d))
            stream << d;
        else stream << "\"" << d << "\"";
    }

    virtual void writeString(const std::string & s)
    {
        stream << '\"';
        ML::jsonEscape(s, stream);
        stream << '\"';
    }

    virtual void writeStringUtf8(const Utf8String & s);

    virtual void writeJson(const Json::Value & val)
    {
        stream << val.toStringNoNewLine();
    }

    virtual void writeBool(bool b)
    {
        stream << (b ? "true": "false");
    }

};


/*****************************************************************************/
/* STRUCTURED JSON PRINTING CONTEXT                                          */
/*****************************************************************************/

/** JSON printing context that puts things into a structure. */

struct StructuredJsonPrintingContext
    : public JsonPrintingContext {

    Json::Value output;
    Json::Value * current;

    StructuredJsonPrintingContext()
        : current(&output)
    {
    }

    std::vector<Json::Value *> path;

    virtual void startObject()
    {
        *current = Json::Value(Json::objectValue);
        path.push_back(current);
    }

    virtual void startMember(const std::string & memberName)
    {
        current = &(*path.back())[memberName];
    }

    virtual void endObject()
    {
        path.pop_back();
    }

    virtual void startArray(int knownSize = -1)
    {
        *current = Json::Value(Json::arrayValue);
        path.push_back(current);
    }

    virtual void newArrayElement()
    {
        Json::Value & b = *path.back();
        current = &b[b.size()];
    }

    virtual void endArray()
    {
        path.pop_back();
    }
    
    virtual void skip()
    {
        *current = Json::Value();
    }

    virtual void writeNull()
    {
        *current = Json::Value();
    }

    virtual void writeInt(int i)
    {
        *current = i;
    }

    virtual void writeUnsignedInt(unsigned int i)
    {
        *current = i;
    }

    virtual void writeLong(long int i)
    {
        *current = i;
    }

    virtual void writeUnsignedLong(unsigned long int i)
    {
        *current = i;
    }

    virtual void writeLongLong(long long int i)
    {
        *current = i;
    }

    virtual void writeUnsignedLongLong(unsigned long long int i)
    {
        *current = i;
    }

    virtual void writeFloat(float f)
    {
        *current = f;
    }

    virtual void writeDouble(double d)
    {
        *current = d;
    }

    virtual void writeString(const std::string & s)
    {
        *current = s;
    }

    virtual void writeStringUtf8(const Utf8String & s)
    {
        *current = s;
    }

    virtual void writeJson(const Json::Value & val)
    {
        *current = val;
    }

    virtual void writeBool(bool b)
    {
        *current = b;
    }
};

} // namespace Datacratic


/* json_printing.h                                                 -*- C++ -*-
   Jeremy Barnes, 26 February 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Context to print out JSON.
*/

#pragma once

#include "json_parsing.h"
#include "jml/utils/exc_assert.h"
#include <boost/algorithm/string.hpp>


namespace Datacratic {

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
    virtual void writeFloat(float f) = 0;
    virtual void writeDouble(double d) = 0;
    virtual void writeString(const std::string & s) = 0;

    virtual void writeJson(const Json::Value & val) = 0;
    virtual void skip() = 0;
};

struct StreamJsonPrintingContext
    : public JsonPrintingContext {

    StreamJsonPrintingContext(std::ostream & stream)
        : stream(stream)
    {
    }

    std::ostream & stream;

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
    std::string prefix;

    void writePrefix()
    {
        stream << prefix;
        prefix = "";
    }

    virtual void startObject()
    {
        writePrefix();
        path.push_back(true /* isObject */);
        stream << "{";
    }

    virtual void startMember(const std::string & memberName)
    {
        ExcAssert(path.back().isObject);
        path.back().memberName = memberName;
        ++path.back().memberNum;
        if (path.back().memberNum != 0)
            stream << ",";
        prefix = '\"' + ML::jsonEscape(memberName) + "\":";
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
        writePrefix();
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
        if (!prefix.empty())
            prefix = "";
        else stream << "null";
    }

    virtual void writeInt(int i)
    {
        writePrefix();
        stream << i;
    }

    virtual void writeFloat(float f)
    {
        writePrefix();
        stream << f;
    }

    virtual void writeDouble(double d)
    {
        writePrefix();
        stream << d;
    }

    virtual void writeString(const std::string & s)
    {
        writePrefix();
        stream << '\"' << ML::jsonEscape(s) << '\"';
    }

    virtual void writeJson(const Json::Value & val)
    {
        writePrefix();
        stream << boost::trim_copy(val.toString());
    }
};

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

    virtual void writeInt(int i)
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

    virtual void writeJson(const Json::Value & val)
    {
        *current = val;
    }
};

} // namespace Datacratic


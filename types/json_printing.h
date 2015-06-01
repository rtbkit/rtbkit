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

    StreamJsonPrintingContext(std::ostream & stream);

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

    virtual void startObject();

    virtual void startMember(const std::string & memberName);

    virtual void endObject();

    virtual void startArray(int knownSize = -1);

    virtual void newArrayElement();

    virtual void endArray();
    
    virtual void skip();

    virtual void writeNull();

    virtual void writeInt(int i);

    virtual void writeUnsignedInt(unsigned int i);

    virtual void writeLong(long int i);

    virtual void writeUnsignedLong(unsigned long int i);

    virtual void writeLongLong(long long int i);

    virtual void writeUnsignedLongLong(unsigned long long int i);

    virtual void writeFloat(float f);

    virtual void writeDouble(double d);

    virtual void writeString(const std::string & s);

    virtual void writeStringUtf8(const Utf8String & s);

    virtual void writeJson(const Json::Value & val);

    virtual void writeBool(bool b);
};


/*****************************************************************************/
/* STRUCTURED JSON PRINTING CONTEXT                                          */
/*****************************************************************************/

/** JSON printing context that puts things into a structure. */

struct StructuredJsonPrintingContext
    : public JsonPrintingContext {

    Json::Value output;
    Json::Value * current;

    StructuredJsonPrintingContext();

    std::vector<Json::Value *> path;

    virtual void startObject();

    virtual void startMember(const std::string & memberName);

    virtual void endObject();

    virtual void startArray(int knownSize = -1);

    virtual void newArrayElement();

    virtual void endArray();
    
    virtual void skip();

    virtual void writeNull();

    virtual void writeInt(int i);

    virtual void writeUnsignedInt(unsigned int i);

    virtual void writeLong(long int i);

    virtual void writeUnsignedLong(unsigned long int i);

    virtual void writeLongLong(long long int i);

    virtual void writeUnsignedLongLong(unsigned long long int i);

    virtual void writeFloat(float f);

    virtual void writeDouble(double d);

    virtual void writeString(const std::string & s);

    virtual void writeStringUtf8(const Utf8String & s);

    virtual void writeJson(const Json::Value & val);

    virtual void writeBool(bool b);
};

} // namespace Datacratic


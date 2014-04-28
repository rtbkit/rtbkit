/* messages.h                                                      -*- C++ -*-
   Jeremy Barnes, 31 May 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Functions to aid in messaging.
*/

#ifndef __router__messages_h__
#define __router__messages_h__

#include "soa/types/value_description.h"
#include "soa/service/zmq.hpp"
#include "rtbkit/common/json_holder.h"

namespace Datacratic {

inline zmq::message_t encodeMessage(const Id & id)
{
    return id.toString();
}

} // namespace Datacratic

namespace RTBKIT {

inline zmq::message_t encodeMessage(const JsonHolder & j)
{
    return j.toString();
}

inline int toInt(const std::string & str)
{
    char * end;
    const char * start = str.c_str();
    int result = strtol(start, &end, 10);
    if (end != start + str.length())
        throw ML::Exception("couldn't parse int");
    return result;
}

inline long toLong(const std::string & str)
{
    char * end;
    const char * start = str.c_str();
    long result = strtol(start, &end, 10);
    if (end != start + str.length())
        throw ML::Exception("couldn't parse int");
    return result;
}

inline double toDouble(const std::string & str)
{
    char * end;
    const char * start = str.c_str();
    double result = strtod(start, &end);
    if (end != start + str.length())
        throw ML::Exception("couldn't parse double");
    return result;
}

inline std::string toString(double d)
{
    return ML::format("%f", d);
}

inline std::string toString(int i)
{
    return ML::format("%i", i);
}

inline std::string toString(signed long l)
{
    return ML::format("%li", l);
}

inline std::string toString(unsigned long l)
{
    return ML::format("%ld", l);
}

template<typename T>
struct Message {
    T payload;
    bool ok;

    Message() : ok(false) {
    }

    Message(T value) : payload(std::move(value)), ok(true) {
    }

    explicit operator bool() {
        return ok;
    }

    std::string toString() const {
        using namespace Datacratic;
        static auto desc = getDefaultDescriptionShared((T*) 0);

        std::stringstream stream;
        Datacratic::StreamJsonPrintingContext context(stream);
        desc->printJson(&payload, context);

        return ML::format("{\"%s\":%s}", desc->typeName, stream.str());
    }

    static Message<T> fromString(std::string const & value) {
        Message<T> result;
        ML::Parse_Context source("Message", value.c_str(), value.size());
        expectJsonObject(source, [&](std::string key,
                                     ML::Parse_Context & context) {
            auto desc = Datacratic::ValueDescription::get(key);
            if(desc) {
                Datacratic::StreamingJsonParsingContext json(context);
                desc->parseJson(&result.payload, json);
                result.ok = true;
            }
        });

        return result;
    }
};

} // namespace RTBKIT

#endif /* __router__messages_h__ */

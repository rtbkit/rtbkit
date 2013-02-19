/* messages.h                                                      -*- C++ -*-
   Jeremy Barnes, 31 May 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Functions to aid in messaging.
*/

#ifndef __router__messages_h__
#define __router__messages_h__

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

} // namespace RTBKIT

#endif /* __router__messages_h__ */

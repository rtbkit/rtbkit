/* redis_utils.cc
   Wolfgang Sourdeau, 17 December 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Redis migration class
 */
#include "soa/service/redis.h"

#include "redis_utils.h"


namespace RTBKIT {

using namespace Redis;

bool
GetRedisReplyAsInt(const Reply & reply, long long int & value)
{
    bool result(true);

    switch (reply.type()) {
    case STRING:
        value = atoll(reply.asString().c_str());
        break;
    case INTEGER:
        value = reply.asInt();
        break;
    case NIL:
        value = 0;
        break;
    default:
        result = false;
    }

    return result;
}

} // namespace RTBKIT

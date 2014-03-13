/* redis_utils.h
   Wolfgang Sourdeau, 17 December 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Redis migration class
 */
#ifndef REDIS_UTILS_H
#define REDIS_UTILS_H

namespace RTBKIT {

/* convert redis INT or STRING reply to long long int and validate type */
bool GetRedisReplyAsInt(const Redis::Reply & reply, long long int & value);

} // namespace RTBKIT

#endif /* REDIS_UTILS_H */

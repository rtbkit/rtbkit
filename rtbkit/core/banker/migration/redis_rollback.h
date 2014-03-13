/* redis_rollback.h
   Wolfgang Sourdeau, 7 January 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
   
   Redis rollback class
 */

#ifndef REDIS_ROLLBACK_H
#define REDIS_ROLLBACK_H

namespace Redis {
    class Address;
}

namespace RTBKIT {

struct RedisRollback {
    void perform(const Redis::Address & sourceAddress,
                 const Redis::Address & targetAddress);
};

} // namespace RTBKIT

#endif /* REDIS_ROLLBACK_H */

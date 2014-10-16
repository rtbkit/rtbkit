/* redis_migration.h
   Wolfgang Sourdeau, 17 December 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.
   
   Redis migration class
 */

#ifndef REDIS_MIGRATION_H
#define REDIS_MIGRATION_H

namespace Redis {
    class Address;
}

namespace RTBKIT {

struct RedisMigration {
    void perform(const Redis::Address & sourceAddress,
                 int delta,
                 const Redis::Address & targetAddress);
};

} // namespace RTBKIT

#endif /* REDIS_MIGRATION_H */

/* redis_migration.cc
   Wolfgang Sourdeau, 17 December 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.
   
   Redis migration script from campaign:strategy schema to the new accounts
   schema
 */

#include <iostream>

#include <boost/program_options/options_description.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include "soa/service/redis.h"

#include "redis_migration.h"
#include "redis_rollback.h"


using namespace std;
using namespace boost::program_options;


int main(int argc, char *argv[])
{
    string sourceHost("localhost"), targetHost("");
    int sourcePort(6379), targetPort(0), delta(0);

    options_description migration_options("Redis options");
    migration_options.add_options()
        ("redis-host,h", value<string>(&sourceHost),
         "source Redis host")
        ("redis-port,p", value<int>(&sourcePort),
         "source Redis port")
        ("delta,d", value<int>(&delta),
         "acceptable inconsistency delta between 'budget',"
         " 'transfer' and 'available', in µUSD (0)")
        ("redis-target-host,i", value<string>(&targetHost),
         "target Redis host (optional)")
        ("redis-target-port,q", value<int>(&targetPort),
         "target Redis port (optional)");
    options_description all_opt("redis_migrate");
    all_opt.add(migration_options);
    all_opt.add_options()
        ("rollback,R", "rollback from the new schema to the old one")
        ("help,H", "print this message");

    variables_map vm;
    store(command_line_parser(argc, argv)
          .options(all_opt)
          //.positional(p)
          .run(),
          vm);
    notify(vm);

    if (vm.count("help")) {
        cerr << all_opt << endl;
        return 1;
    }

    bool rollback(vm.count("rollback") != 0);

    Redis::Address sourceAddress = Redis::Address::tcp(sourceHost, sourcePort);
    if (targetHost == "") {
        targetHost = sourceHost;
    }
    if (targetPort == 0) {
        targetPort = sourcePort;
    }

    cerr << sourceHost << ":" << sourcePort << endl;
    cerr << targetHost << ":" << targetPort << endl;
    Redis::Address targetAddress = Redis::Address::tcp(targetHost, targetPort);

    if (rollback) {
        RTBKIT::RedisRollback rollback;
        rollback.perform(sourceAddress, targetAddress);
    }
    else {
        RTBKIT::RedisMigration migration;
        migration.perform(sourceAddress, delta, targetAddress);
    }

    return 0;
}

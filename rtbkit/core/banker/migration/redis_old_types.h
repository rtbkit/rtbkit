/* redis_old_types.h
   Wolfgang Sourdeau, 7 January 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.
 */

#ifndef REDIS_OLD_TYPES_H
#define REDIS_OLD_TYPES_H

namespace Redis {
    class AsyncConnection;
}

namespace RTBKIT {

extern const std::string CampaignsPrefix;

struct Campaign;
struct Strategy;

typedef std::unordered_map<std::string, Campaign> Campaigns;
typedef std::unordered_map<std::string, Strategy> Strategies;

struct Campaign {
    Campaign(const std::string & key = "",
             long long available = -1,
             long long transferred = -1);

    void load(Redis::AsyncConnection & redis);
    void save(Redis::AsyncConnection & redis) const;

    bool validateAll(int acceptedDelta) const;

    std::string key_;

    long long int available_;
    long long int transferred_;

    std::vector<Strategy> strategies;
};

struct Strategy {
    Strategy(const std::string & key,
             const std::string & campaignKey,
             long long available = -1,
             long long spent = -1,
             long long transferred = -1);

    void load(Redis::AsyncConnection & redis, int acceptedDelta);
    void assignToParent(Campaigns & campaigns) const;

    void save(Redis::AsyncConnection & redis) const;

    std::string key_;
    std::string campaignKey_;
    bool valid_;

    long long available_;
    long long spent_;
    long long transferred_;
};

} // namespace RTBKIT

#endif /* REDIS_OLD_TYPES_H */

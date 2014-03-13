#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "soa/service/redis.h"

#include "redis_utils.h"

#include "redis_old_types.h"


namespace {

bool IsMoreOrLess(long long int value, long long int acceptedDelta)
{
    int absolute(acceptedDelta >= 0 ? acceptedDelta : -acceptedDelta);

    return (value >= 0
            ? (value <= absolute)
            : (-value <= absolute));
}

}


namespace RTBKIT {

using namespace std;
using namespace RTBKIT;
using namespace Redis;


const string CampaignsPrefix = "campaigns:";


/* CAMPAIGN */
Campaign::
Campaign(const string & key,
         long long available, long long transferred)
    : key_(key),
      available_(available), transferred_(transferred)
{
}

/* validate campaign numbers with strategies */
bool
Campaign::
validateAll(int acceptedDelta)
    const
{
    long long int transferred(0);
    bool allValid(true);

    for (const Strategy & strategy: strategies) {
        transferred += strategy.transferred_;
        allValid &= strategy.valid_;
    }

    string campaignKey(CampaignsPrefix + key_);
    if (!allValid) {
        cerr << "! campaign '" << campaignKey
             << "' contains invalid strategies"
             << endl;
        return false;
    }
 
    if (!IsMoreOrLess(transferred - transferred_,
                      (acceptedDelta * strategies.size()))) {
        cerr << "! campaign '" << campaignKey
             << "' has invalid 'transferred' value:"
             << endl
             << "    (computed) " << transferred
             << " != (stored) " << transferred_
             << endl;
        return false;
    }

    cerr << "- campaign '" << campaignKey
         << "' is valid and ready for conversion"
         << endl;

    return true;
}

void
Campaign::
load(AsyncConnection & redis)
{
    string redisKey(CampaignsPrefix + key_);
    Command hmget = HMGET(redisKey);
    hmget.addArg("available");
    hmget.addArg("transferred");

    Result result = redis.exec(hmget);
    if (!result.ok()) {
        cerr << "! HMGET " + redisKey
             << ": error fetching result" << endl;
        return;
    }
    const Reply & reply = result.reply();
    if (reply.type() != ARRAY) {
        cerr << "! HMGET " + redisKey
             << ": unexpected reply type" << endl;
        return;
    }
    if (!GetRedisReplyAsInt(reply[0], available_)) {
        cerr << "! HMGET " + redisKey
             << ": value for 'available' cannot be converted to int"
             << endl;
        return;
    }
    if (!GetRedisReplyAsInt(reply[1], transferred_)) {
        cerr << "! HMGET " + redisKey
             << ": value for 'transferred' cannot be converted to int"
             << endl;
        return;
    };
    cerr << "- campaign '" + redisKey + "' properly loaded"
         << endl;
}

void
Campaign::
save(AsyncConnection & redis)
    const
{
    string redisKey(CampaignsPrefix + key_);

    Command hmset = HMSET(redisKey);
    hmset.addArg("available");
    hmset.addArg(available_);
    hmset.addArg("transferred");
    hmset.addArg(transferred_);

    Result result = redis.exec(hmset);
    if (!result.ok()) {
        cerr << "! HMSET " + redisKey
             << ": error storing campaign keys" << endl;
        return;
    }

    cerr << "- campaign '" + redisKey + "' properly saved" << endl;
}

/* STRATEGY */
Strategy::
Strategy(const string & key, const string & campaignKey,
         long long available, long long spent, long long transferred)
    : key_(key), campaignKey_(campaignKey), valid_(false),
      available_(available), spent_(spent),
      transferred_(transferred)
{
}

void
Strategy::
load(AsyncConnection & redis, int acceptedDelta)
{
    string redisKey(CampaignsPrefix + campaignKey_ + ":" + key_);
    Command hmget = HMGET(redisKey);
    hmget.addArg("available");
    hmget.addArg("spent");
    hmget.addArg("transferred");
    Result result = redis.exec(hmget);
    if (!result.ok()) {
        cerr << "! HMGET " + redisKey
             << ": error fetching result" << endl;
        return;
    }
    const Reply & reply = result.reply();
    if (reply.type() != ARRAY) {
        cerr << "! HMGET " + redisKey
             << ": unexpected reply type" << endl;
        return;
    }
    if (!GetRedisReplyAsInt(reply[0], available_)) {
        cerr << "! HMGET " + redisKey
             << ": value for 'available' cannot be converted to int"
             << endl;
        return;
    }
    if (!GetRedisReplyAsInt(reply[1], spent_)) {
        cerr << "! HMGET " + redisKey
             << ": value for 'spent' cannot be converted to int"
             << endl;
        return;
    };
    if (!GetRedisReplyAsInt(reply[2], transferred_)) {
        cerr << "! HMGET " + redisKey
             << ": value for 'transferred' cannot be converted to int"
             << endl;
        return;
    };

    /* validation */
    if (transferred_ < 0
        // || !IsMoreOrLess(available_, acceptedDelta)
        || spent_ < 0) {
        cerr << "! strategy '" + redisKey + "' has one or more negative values:"
             << endl
             << "    transferred = " << transferred_
             << "; available = " << available_
             << "; spent = " << spent_
             << endl;
    }
#if 0
    else if (!IsMoreOrLess((transferred_ - spent_) - available_,
                           acceptedDelta)) {
        long long int delta = (available_ + spent_) - transferred_;
        cerr << "! strategy '" + redisKey + "' is in an inconsistent state:"
             << endl
             << "    transferred = " << transferred_
             << "; available = " << available_
             << "; spent = " << spent_
             << endl
             << "    delta: " << delta
             << endl;
    }
#endif
    else {
        cerr << "- strategy '" + redisKey + "' properly loaded and consistent"
             << endl;
        valid_ = true;
    }
}

void
Strategy::
save(AsyncConnection & redis)
    const
{
    string redisKey(CampaignsPrefix + campaignKey_ + ":" + key_);

    if (!valid_) {
        cerr << "! skipped the saving of 'invalid' strategy '" + redisKey + "'"
             << endl;
        return;
    }

    Command hmset = HMSET(redisKey);
    hmset.addArg("available");
    hmset.addArg(available_);
    hmset.addArg("spent");
    hmset.addArg(spent_);
    hmset.addArg("transferred");
    hmset.addArg(transferred_);

    Result result = redis.exec(hmset);
    if (!result.ok()) {
        cerr << "! HMSET " + redisKey
             << ": error storing campaign keys" << endl;
        return;
    }

    cerr << "- strategy '" + redisKey + "' properly saved" << endl;
}

/* assign strategy to the right campaign */
void Strategy::
assignToParent(Campaigns & campaigns) const
{
    if (campaigns.count(campaignKey_) == 0) {
        cerr << "! ignored strategy '"
             << CampaignsPrefix << campaignKey_ << ":" << key_
             << "' without corresponding campaign"
             << endl;
    }
    else {
        Campaign & campaign = campaigns.at(campaignKey_);
        campaign.strategies.push_back(*this);
    }
}

} // namespace RTBKIT

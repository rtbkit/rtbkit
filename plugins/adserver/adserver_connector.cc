/* adserver_connector.cc
   Jeremy Barnes, 19 December 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

   Base class to connect to an ad server.  We also have an http ad server
   connector that builds on top of this.
*/


#include "adserver_connector.h"

#include "rtbkit/common/auction_events.h"


using namespace std;

namespace RTBKIT {


/*****************************************************************************/
/* POST AUCTION PROXY                                                        */
/*****************************************************************************/

AdServerConnector::
AdServerConnector(const string & serviceName,
                  const shared_ptr<Datacratic::ServiceProxies> & proxy)
    : ServiceBase(serviceName, proxy),
      toPostAuctionService_(proxy->zmqContext)
{
}

AdServerConnector::
~AdServerConnector()
{
}

void
AdServerConnector::
init(shared_ptr<ConfigurationService> config)
{
    shared_ptr<ServiceProxies> services = getServices();

    registerServiceProvider(serviceName_, { "adServer" });
    services->config->removePath(serviceName());

    toPostAuctionService_.init(config, ZMQ_XREQ);
    toPostAuctionService_.connectToServiceClass("rtbPostAuctionService",
                                                "events");

#if 0 // later, for when we have multiple

    toPostAuctionServices_.init(getServices()->config, name);
    toPostAuctionServices_.connectHandler = [=] (const string & connectedTo) {
        cerr << "AdServerConnector is connected to post auction service "
        << connectedTo << endl;
    };
    toPostAuctionServices_.connectAllServiceProviders("rtbPostAuctionService",
                                                      "agents");
    toPostAuctionServices_.connectToServiceClass();
#endif
}

void
AdServerConnector::
start()
{
    startTime_ = Date::now();
    recordHit("up");
}

void
AdServerConnector::
shutdown()
{
}

void
AdServerConnector::
recordUptime()
    const
{
    recordLevel(Date::now().secondsSince(startTime_), "uptime");
}

void
AdServerConnector::
publishWin(const Id & auctionId,
           const Id & adSpotId,
           Amount winPrice,
           Date timestamp,
           const JsonHolder & winMeta,
           const UserIds & ids,
           const AccountKey & account,
           Date bidTimestamp)
{
    recordHit("receivedEvent");
    recordHit("event.WIN");

    PostAuctionEvent event;
    event.type = PAE_WIN;
    event.auctionId = auctionId;
    event.adSpotId = adSpotId;
    event.winPrice = winPrice;
    event.timestamp = timestamp;
    event.metadata = winMeta;
    event.uids = ids;
    event.account = account;
    event.bidTimestamp = bidTimestamp;

    string str = ML::DB::serializeToString(event);
    toPostAuctionService_.sendMessage("WIN", str);
}

void
AdServerConnector::
publishLoss(const Id & auctionId,
            const Id & adSpotId,
            Date timestamp,
            const JsonHolder & lossMeta,
            const AccountKey & account,
            Date bidTimestamp)
{
    recordHit("receivedEvent");
    recordHit("event.LOSS");

    PostAuctionEvent event;
    event.type = PAE_LOSS;
    event.auctionId = auctionId;
    event.adSpotId = adSpotId;
    event.timestamp = timestamp;
    event.metadata = lossMeta;
    event.account = account;
    event.bidTimestamp = bidTimestamp;

    string str = ML::DB::serializeToString(event);
    toPostAuctionService_.sendMessage("LOSS", str);
}

void
AdServerConnector::
publishCampaignEvent(const string & label,
                     const Id & auctionId,
                     const Id & adSpotId,
                     Date timestamp,
                     const JsonHolder & impressionMeta,
                     const UserIds & ids)
{ 
    recordHit("receivedEvent");
    recordHit("event." + label);

    PostAuctionEvent event;
    event.type = PAE_CAMPAIGN_EVENT;
    event.label = label;
    event.auctionId = auctionId;
    event.adSpotId = adSpotId;
    event.timestamp = timestamp;
    event.uids = ids;
    event.metadata = impressionMeta;

    string str = ML::DB::serializeToString(event);
    toPostAuctionService_.sendMessage("EVENT", str);
}

} // namespace RTBKIT

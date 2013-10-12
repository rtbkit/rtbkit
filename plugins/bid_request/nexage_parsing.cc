/* nexage_parsing.cc
    Syed Murtaza Hussain Kazmi, 11 September 2013
   Copyright (c) 2013 Virtual Force.  All rights reserved.


   Structure descriptions for Nexage.
*/

#include "nexage_parsing.h"
#include "soa/types/json_parsing.h"

//using namespace RTBKIT;
using namespace std;

namespace Datacratic {


DefaultDescription<Nexage::BidRequest>::
DefaultDescription()
{
        onUnknownField = [=] (Nexage::BidRequest* br, JsonParsingContext & context)
        {
            //cerr << "got unknown field " << context.printPath() << endl;

            std::function<Json::Value & (int, Json::Value &)> getEntry
            = [&] (int n, Json::Value & curr) -> Json::Value &
            {
                if (n == context.path.size())
                    return curr;
                else if (context.path[n].index != -1)
                    return getEntry(n + 1, curr[context.path[n].index]);
                else return getEntry(n + 1, curr[context.path[n].key]);
            };

            getEntry(0, br->unparseable)
                = context.expectJson();
        };

    
	addField("id", &Nexage::BidRequest::id, "Bid Request ID");
    addField("imp", &Nexage::BidRequest::imp, "Impressions");
    
    addField("site", &Nexage::BidRequest::site, "Information about site the request is on");
    addField("app", &Nexage::BidRequest::app, "Information about the app the request is being shown in");
    addField("device", &Nexage::BidRequest::device, "Information about the device on which the request was made");
    addField("user", &Nexage::BidRequest::user, "Information about the user who is making the request");
    addField("at", &Nexage::BidRequest::at, "Type of auction: 1(st) or 2(nd)");
    addField("tmax", &Nexage::BidRequest::tmax, "Maximum response time (ms)");
    addField("wseat", &Nexage::BidRequest::wseat, "Allowable seats");
    addField("allimps", &Nexage::BidRequest::allimps, "Set to 1 if all impressions on this page are in the bid request");
    addField("cur", &Nexage::BidRequest::cur, "List of acceptable currencies to bid in");
    addField("bcat", &Nexage::BidRequest::bcat, "Blocked advertiser content categories");
    addField("badv", &Nexage::BidRequest::badv, "Blocked adversiser domains");
    addField("ext", &Nexage::BidRequest::ext, "Extended fields outside of protocol");
    addField("unparseable", &Nexage::BidRequest::unparseable, "Unparseable fields are collected here");	
}

DefaultDescription<Nexage::NexageGeo>::
DefaultDescription()
{
    addField("lat", &Nexage::NexageGeo::lat, "Latiture of user in degrees from equator");
    addField("lon", &Nexage::NexageGeo::lon, "Longtitude of user in degrees (-180 to 180)");
    addField("country", &Nexage::NexageGeo::country, "ISO 3166-1 country code");
    addField("region", &Nexage::NexageGeo::region, "ISO 3166-2 Region code");
    addField("regionfips104", &Nexage::NexageGeo::regionfips104, "FIPS 10-4 region code");
    addField("metro", &Nexage::NexageGeo::metro, "Metropolitan region (Google Metro code");
    addField("city", &Nexage::NexageGeo::city, "City name (UN Code for Trade and Transport)");
    addField("zip", &Nexage::NexageGeo::zip, "Zip or postal code");
    addField("type", &Nexage::NexageGeo::type, "Source of location data");
    addField("ext", &Nexage::NexageGeo::ext, "Extensions to the protocol go here");
    /// Datacratic extension
    addField("dma", &Nexage::NexageGeo::dma, "DMA code");
}

DefaultDescription<Nexage::NexageUser>::
DefaultDescription()
{
    addField("id", &Nexage::NexageUser::id, "Exchange-specific user ID");
    addField("buyeruid", &Nexage::NexageUser::buyeruid, "Year of birth");
    addField("gender", &Nexage::NexageUser::gender, "Gender: Male, Female, Other");
    addField("keywords", &Nexage::NexageUser::keywords, "List of keywords of consumer intent");
    addField("customdata", &Nexage::NexageUser::customdata, "Custom data from exchange");
    addField("geo", &Nexage::NexageUser::geo, "Geolocation of user at registration");
    addField("data", &Nexage::NexageUser::data, "User data segments");
    addField("ext", &Nexage::NexageUser::ext, "Extensions go here, new in OpenRTB 2.1");
}


} // namespace Datacratic


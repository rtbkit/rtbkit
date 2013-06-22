/* fbx_parsing.cc
   Jean-Sebastien Bejeau, 19 June 2013

   Structure descriptions for FBX.
*/

#include "fbx_parsing.h"
#include "soa/types/json_parsing.h"

using namespace FBX;
//using namespace RTBKIT;
using namespace std;

namespace Datacratic {

DefaultDescription<BidRequest>::
DefaultDescription()
{
    onUnknownField = [=] (BidRequest * br, JsonParsingContext & context)
        {
            //cerr << "got unknown field " << context.printPath() << endl;

            std::function<Json::Value & (int, Json::Value &)> getEntry
            = [&] (int n, Json::Value & curr) -> Json::Value &
            {
                if (n == context.path.size())
                    return curr;
                else if (context.path[n].index != -1)
                    return getEntry(n + 1, curr[context.path[n].index]);
                else return getEntry(n + 1, curr[context.path[n].fieldName()]);
            };

            getEntry(0, br->unparseable)
                = context.expectJson();
        };

    addField("requestId", &BidRequest::requestId, "Bid request ID");
    addField("partnerMatchId", &BidRequest::partnerMatchId, "Partner’s user ID");
    addField("userContext", &BidRequest::userContext, "An object of type UserContext");
    addField("pageContext", &BidRequest::pageContext, "An object of type PageContext");
    addField("istest", &BidRequest::istest, "Indicates an auction being held purely for debugging purposes");
    addField("allowViewTag", &BidRequest::allowViewTag, "Indicates if view tags are accepted.");
    addField("unparseable", &BidRequest::unparseable, "Unparseable fields are collected here");

}

DefaultDescription<RtbPageContext>::
DefaultDescription()
{
    addField("pageTypeId", &RtbPageContext::pageTypeId, "Page type");
    addField("numSlots", &RtbPageContext::numSlots, "Estimated number of ad slots in the placement");
}

DefaultDescription<RtbUserContext>::
DefaultDescription()
{
    addField("ipAdressMasked", &RtbUserContext::ipAdressMasked, "User IP address");
    addField("userAgent", &RtbUserContext::userAgent, "User agent from the user browser");
    addField("country", &RtbUserContext::country, "Country");
}

} // namespace Datacratic

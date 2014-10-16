/* rest_request_binding.cc                                         -*- C++ -*-
   Jeremy Barnes, 21 May 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

*/

#include "rest_request_binding.h"

namespace Datacratic {

/** These functions turn an argument to the request binding into a function
    that can generate the value required by the handler function.

*/

std::function<std::string
              (const RestServiceEndpoint::ConnectionId & connection,
               const RestRequest & request,
               const RestRequestParsingContext & context)>
createParameterExtractor(Json::Value & argHelp,
                         const StringPayload & p, void *)
{
    Json::Value & v = argHelp["payload"];
    v["description"] = p.description;

    return [=] (const RestServiceEndpoint::ConnectionId & connection,
                const RestRequest & request,
                const RestRequestParsingContext & context)
        {
            return request.payload;
        };
}

/** Pass the connection on */
std::function<const RestServiceEndpoint::ConnectionId &
                     (const RestServiceEndpoint::ConnectionId & connection,
                      const RestRequest & request,
                      const RestRequestParsingContext & context)>
createParameterExtractor(Json::Value & argHelp,
                         const PassConnectionId &, void *)
{
    return [] (const RestServiceEndpoint::ConnectionId & connection,
                const RestRequest & request,
                const RestRequestParsingContext & context)
        -> const RestServiceEndpoint::ConnectionId &
        {
            return connection;
        };
}

/** Pass the connection on */
std::function<const RestRequestParsingContext &
                     (const RestServiceEndpoint::ConnectionId & connection,
                      const RestRequest & request,
                      const RestRequestParsingContext & context)>
createParameterExtractor(Json::Value & argHelp,
                         const PassParsingContext &, void *)
{
    return [] (const RestServiceEndpoint::ConnectionId & connection,
                const RestRequest & request,
                const RestRequestParsingContext & context)
        -> const RestRequestParsingContext &
        {
            return context;
        };
}

/** Pass the connection on */
std::function<const RestRequest &
                     (const RestServiceEndpoint::ConnectionId & connection,
                      const RestRequest & request,
                      const RestRequestParsingContext & context)>
createParameterExtractor(Json::Value & argHelp,
                         const PassRequest &, void *)
{
    return [] (const RestServiceEndpoint::ConnectionId & connection,
               const RestRequest & request,
               const RestRequestParsingContext & context)
        -> const RestRequest &
        {
            return request;
        };
}



} // namespace Datacratic

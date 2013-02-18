/** rest_request_router.h                                          -*- C++ -*-
    Jeremy Barnes, 13 November 2012
    Copyright (c) 2012 Datacratic Inc.  All rights reserved.

*/

#pragma once

#include "soa/service/named_endpoint.h"
#include "soa/service/message_loop.h"
#include "soa/service/rest_service_endpoint.h"
#include "jml/utils/vector_utils.h"
#include "jml/utils/positioned_types.h"
//#include <regex>
#include <boost/regex.hpp>

namespace Datacratic {


/*****************************************************************************/
/* PATH SPEC                                                                 */
/*****************************************************************************/

/** This matches part of a path for a REST URI. */

struct PathSpec {
    enum Type {
        NONE,
        STRING,
        REGEX
    } type;

    PathSpec()
        : type(NONE)
    {
    }
        
    PathSpec(const std::string & fullPath)
        : type(STRING), path(fullPath)
    {
    }

    PathSpec(const char * fullPath)
        : type(STRING), path(fullPath)
    {
    }

    PathSpec(const std::string & str, const boost::regex & rex)
        : type(REGEX),
          path(str),
          rex(rex)
    {
    }

    void getHelp(Json::Value & result)
    {
        switch (type) {
        case STRING:
            result["path"] = path;
            break;
        case REGEX: {
            Json::Value & v = result["path"];
            v["regex"] = path;
            v["desc"] = desc;
            break;
        }
        default:
            throw ML::Exception("unknown path parameter");
        }
    }
    
    std::string getPathDesc() const
    {
        if (!desc.empty())
            return desc;
        return path;
    }

    std::string path;
    boost::regex rex;
    std::string desc;

    bool operator == (const PathSpec & other) const
    {
        return path == other.path;
    }

    bool operator != (const PathSpec & other) const
    {
        return ! operator == (other);
    }

    bool operator < (const PathSpec & other) const
    {
        return path < other.path;
    }
};

struct Rx : public PathSpec {
    Rx(const std::string & regexString, const std::string & desc)
        : PathSpec(regexString, boost::regex(regexString))
    {
        this->desc = desc;
    }
};

std::ostream & operator << (std::ostream & stream, const PathSpec & path);


/*****************************************************************************/
/* REQUEST FILTER                                                            */
/*****************************************************************************/

/** Filter for a REST request by method, etc. */

struct RequestFilter {
    RequestFilter()
    {
    }

    RequestFilter(const std::string & verb)
    {
        verbs.insert(verb);
    }

    RequestFilter(const char * verb)
    {
        verbs.insert(verb);
    }

    RequestFilter(std::set<std::string> verbs)
        : verbs(verbs)
    {
    }

    RequestFilter(const std::initializer_list<std::string> & verbs)
        : verbs(verbs)
    {
    }

    void getHelp(Json::Value & result)
    {
        if (!verbs.empty()) {
            int i = 0;
            for (auto it = verbs.begin(), end = verbs.end(); it != end;  ++it, ++i) {
                result["verbs"][i] = *it;
            }
        }
    }

    std::set<std::string> verbs;
};

std::ostream & operator << (std::ostream & stream, const RequestFilter & filter);

/*****************************************************************************/
/* REQUEST REQUEST PARSING CONTEXT                                           */
/*****************************************************************************/

/** Parsing context for a REST request.  Tracks of how the request path
    is processed so that the entity names can be extracted later.
*/

struct RestRequestParsingContext {
    RestRequestParsingContext(const RestRequest & request)
        : remaining(request.resource)
    {
    }

    std::vector<std::string> resources;
    std::string remaining;
};

std::ostream & operator << (std::ostream & stream,
                            const RestRequestParsingContext & context);


/*****************************************************************************/
/* REST REQUEST ROUTER                                                       */
/*****************************************************************************/

struct RestRequestRouter {

    enum MatchResult {
        MR_NO,     ///< Didn't match but can continue
        MR_YES,    ///< Did match
        MR_ERROR   ///< Error
    };    

    typedef std::function<MatchResult (const RestServiceEndpoint::ConnectionId & connection,
                                       const RestRequest & request,
                                       const RestRequestParsingContext & context)>
         OnProcessRequest;

    RestRequestRouter();

    RestRequestRouter(const OnProcessRequest & processRequest,
                      const std::string & description,
                      bool terminal,
                      const Json::Value & argHelp = Json::Value());

    virtual ~RestRequestRouter();
    
    /** Return a requestHandler that can be assigned to the
        RestServiceEndpoint.
    */
    RestServiceEndpoint::OnHandleRequest requestHandler() const;

    virtual void handleRequest(const RestServiceEndpoint::ConnectionId & connection,
                               const RestRequest & request) const;

    virtual MatchResult
    processRequest(const RestServiceEndpoint::ConnectionId & connection,
                   const RestRequest & request,
                   RestRequestParsingContext & context) const;

    struct Route {
        PathSpec path;
        RequestFilter filter;
        std::shared_ptr<RestRequestRouter> router;

        MatchResult process(const RestRequest & request,
                            const RestRequestParsingContext & context,
                            const RestServiceEndpoint::ConnectionId & connection) const;
    };

    /** Add a route that will match the given path and filter and will
        delegate to the given sub-route.
    */
    void addRoute(PathSpec path, RequestFilter filter,
                  const std::shared_ptr<RestRequestRouter> & handler);

    /** Add a terminal route with the given path and filter that will call
        the given callback.
    */
    void addRoute(PathSpec path, RequestFilter filter,
                  const std::string & description,
                  const OnProcessRequest & cb,
                  const Json::Value & argHelp);

    void addHelpRoute(PathSpec path, RequestFilter filter);

    virtual void getHelp(Json::Value & result,
                         const std::string & currentPath,
                         const std::set<std::string> & verbs);

    RestRequestRouter &
    addSubRouter(PathSpec path, const std::string & description);
    
    OnProcessRequest rootHandler;
    std::vector<Route> subRoutes;
    std::string description;
    bool terminal;
    Json::Value argHelp;
};


} // namespace Datacratic

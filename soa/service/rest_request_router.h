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
#include "jml/arch/rtti_utils.h"
#include "jml/arch/demangle.h"
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
/* REST REQUEST PARSING CONTEXT                                              */
/*****************************************************************************/

/** Parsing context for a REST request.  Tracks of how the request path
    is processed so that the entity names can be extracted later.
*/

struct RestRequestParsingContext {
    RestRequestParsingContext(const RestRequest & request)
        : remaining(request.resource)
    {
    }

    /** Add the given object. */
    template<typename T>
    void addObject(T * obj)
    {
        objects.push_back(std::make_pair(obj, &typeid(T)));
    }

    /** Get the object at the given index on the context (defaults to the
        last), and return it and its type.

        Indexes below zero are interpreted as offsets from the end of the
        array.
    */
    std::pair<void *, const std::type_info *> getObject(int index = -1) const
    {
        if (index == -1)
            index = objects.size() + index;
        if (index < 0 || index >= objects.size())
            throw ML::Exception("Attempt to extract invalid object number");

        auto res = objects[index];
        if (!res.first || !res.second)
            throw ML::Exception("invalid object");

        return res;
    }

    /** Get the object at the given index on the context (defaults to the
        last), and convert it safely to the given type.

        Indexes below zero are interpreted as offsets from the end of the
        array.
    */
    template<typename As>
    As & getObjectAs(int index = -1) const
    {
        auto obj = getObject(index);

        const std::type_info * tp = &typeid(As);
        if (tp == obj.second)
            return *reinterpret_cast<As *>(obj.first);

        void * converted = ML::is_convertible(*obj.second,
                                              *tp,
                                              obj.first);
        if (!converted)
            throw ML::Exception("wanted to get object of type "
                                + ML::type_name<As>()
                                + " from incompatible object of type "
                                + ML::demangle(obj.second->name()));

        return *reinterpret_cast<As *>(converted);
    }

    /// List of resources (url components) in the path
    std::vector<std::string> resources;

    /// List of extracted objects to which path components refer.  Both the
    /// object and its type are stored.
    std::vector<std::pair<void *, const std::type_info *> > objects;

    /// Part of the resource that has not yet been consumed
    std::string remaining;
};

std::ostream & operator << (std::ostream & stream,
                            const RestRequestParsingContext & context);


/*****************************************************************************/
/* REST REQUEST ROUTER                                                       */
/*****************************************************************************/

struct RestRequestRouter {

    typedef RestServiceEndpoint::ConnectionId ConnectionId;

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

    /** Type of a function that is called by the route after matching to extract any
        objects referred to so that they can be added to the context and made
        available to futher event handlers.
    */
    typedef std::function<void(RestRequestParsingContext & context)> ExtractObject;

    template<typename T>
    static ExtractObject addObject(T * obj)
    {
        return [=] (RestRequestParsingContext & context)
            {
                context.addObject(obj);
            };
    }

    struct Route {
        PathSpec path;
        RequestFilter filter;
        std::shared_ptr<RestRequestRouter> router;
        std::function<void(RestRequestParsingContext & context)> extractObject;

        MatchResult process(const RestRequest & request,
                            const RestRequestParsingContext & context,
                            const RestServiceEndpoint::ConnectionId & connection) const;
    };

    /** Add a route that will match the given path and filter and will
        delegate to the given sub-route.
    */
    void addRoute(PathSpec path, RequestFilter filter,
                  const std::shared_ptr<RestRequestRouter> & handler,
                  ExtractObject extractObject = nullptr);

    /** Add a terminal route with the given path and filter that will call
        the given callback.
    */
    void addRoute(PathSpec path, RequestFilter filter,
                  const std::string & description,
                  const OnProcessRequest & cb,
                  const Json::Value & argHelp,
                  ExtractObject extractObject = nullptr);

    void addHelpRoute(PathSpec path, RequestFilter filter);

    virtual void getHelp(Json::Value & result,
                         const std::string & currentPath,
                         const std::set<std::string> & verbs);

    /** Create a generic sub router. */
    RestRequestRouter &
    addSubRouter(PathSpec path, const std::string & description,
                 ExtractObject extractObject = nullptr,
                 std::shared_ptr<RestRequestRouter> subRouter = nullptr);

    /** In the normal case, we don't create an ExtractObject function. */
    static ExtractObject getExtractObject(const void *)
    {
        return nullptr;
    }

    /** Where the class has a getObject() function that takes a RestRequestParsingContext,
        we do create an ExtractObject function.
    */
    template<typename T>
    static ExtractObject getExtractObject(T * val,
                                          decltype(std::declval<T *>()->getObject(std::declval<RestRequestParsingContext>())) * = 0)
    {
        return [=] (RestRequestParsingContext & context) -> int
            {
                return val->getObject(context);
            };
    }

    /** Create a sub router of a specific type. */
    template<typename T, typename... Args>
    T &
    addSubRouter(PathSpec path, const std::string & description, Args &&... args)
    {
        // TODO: check it doesn't exist
        Route route;
        route.path = path;
        auto res = std::make_shared<T>(std::forward<Args>(args)...);
        route.router = res;
        route.router->description = description;
        route.extractObject = getExtractObject(res.get());
        subRoutes.push_back(route);
        return *res;
    }
    
    OnProcessRequest rootHandler;
    std::vector<Route> subRoutes;
    std::string description;
    bool terminal;
    Json::Value argHelp;
};


} // namespace Datacratic

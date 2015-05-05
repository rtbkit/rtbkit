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

    void getHelp(Json::Value & result) const
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

    void getHelp(Json::Value & result) const
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
    void addObject(T * obj,
                   std::function<void (void *)> deleter = nullptr)
    {
        objects.emplace_back(obj, &typeid(T), std::move(deleter));
    }

    /** Add a shared pointer to the given object, incrementing the
        count so that it cannot be freed until this parsing context
        releases it.

        This is useful when an object may be deleted during request
        parsing, to make sure it stays alive until the request has
        completed.
    */
    template<typename T>
    void addSharedPtr(std::shared_ptr<T> ptr)
    {
        addObject(new std::shared_ptr<T>(std::move(ptr)),
                  [] (void * ptr) { delete (std::shared_ptr<T> *)ptr; });
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

        auto & res = objects[index];
        if (!res.obj || !res.type)
            throw ML::Exception("invalid object");

        return std::make_pair(res.obj, res.type);
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

        if (&typeid(std::shared_ptr<As>) == obj.second)
            return *reinterpret_cast<std::shared_ptr<As> *>(obj.first)->get();

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

    template<typename As>
    std::shared_ptr<As> getSharedPtrAs(int index = -1) const
    {
        return getObjectAs<std::shared_ptr<As> >(index);
    }

    /// List of resources (url components) in the path
    std::vector<std::string> resources;

    /// Objects
    struct ObjectEntry {
        ObjectEntry(void * obj = nullptr,
                    const std::type_info * type = nullptr,
                    std::function<void (void *) noexcept> deleter = nullptr)
            : obj(obj), type(type), deleter(std::move(deleter))
        {
        }

        ~ObjectEntry() noexcept
        {
            if (deleter)
                deleter(obj);
        }

        void * obj;
        const std::type_info * type;
        std::function<void (void *) noexcept> deleter;

        //ObjectEntry(const ObjectEntry &) = delete;
        //void operator = (const ObjectEntry &) = delete;

        // Needed for gcc 4.6
        ObjectEntry(const ObjectEntry & other)
            : obj(other.obj), type(other.type)
        {
            ObjectEntry & otherNonConst = (ObjectEntry &)other;
            deleter = std::move(otherNonConst.deleter);
            otherNonConst.obj = nullptr;
            otherNonConst.type = nullptr;
        }

        void operator = (const ObjectEntry & other)
        {
            ObjectEntry newMe(other);
            *this = std::move(newMe);
        }
    };

    /// List of extracted objects to which path components refer.  Both the
    /// object and its type are stored as well as a destructor function.
    /// They are shared pointers as the contexts are copied.
    std::vector<ObjectEntry> objects;

    /// Part of the resource that has not yet been consumed
    std::string remaining;

    /// Used to save the state so that whatever was pushed after can be
    /// removed and the object can get back to its old state (without making
    /// a copy).
    struct State {
        std::string remaining;
        int resourcesLength;
        int objectsLength;
    };

    /// Save the current state, to be restored in restoreState
    State saveState() const
    {
        State result;
        result.remaining = remaining;
        result.resourcesLength = resources.size();
        result.objectsLength = objects.size();
        return result;
    }

    /// Restore the current state
    void restoreState(State && state)
    {
        remaining = std::move(state.remaining);
        ExcAssertGreaterEqual(resources.size(), state.resourcesLength);
        resources.resize(state.resourcesLength);
        ExcAssertGreaterEqual(objects.size(), state.objectsLength);
        while (objects.size() > state.objectsLength)
            objects.pop_back();
    }

    /// Guard object to save the state and restore it on scope exit
    struct StateGuard {
        State state;
        RestRequestParsingContext * obj;

        StateGuard(RestRequestParsingContext * obj)
            : state(std::move(obj->saveState())),
              obj(obj)
        {
        }

        ~StateGuard()
        {
            obj->restoreState(std::move(state));
        }
    };

    RestRequestParsingContext(const RestRequestParsingContext &) = delete;
    void operator = (const RestRequestParsingContext &) = delete;
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
        MR_ERROR,  ///< Error
        MR_ASYNC   ///< Handled, but asynchronously
    };    

    typedef std::function<MatchResult (const RestServiceEndpoint::ConnectionId & connection,
                                       const RestRequest & request,
                                       RestRequestParsingContext & context)>
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

    virtual void options(std::set<std::string> & verbsAccepted,
                         Json::Value & help,
                         const RestRequest & request,
                         RestRequestParsingContext & context) const;

    /** Type of a function that is called by the route after matching to extract any
        objects referred to so that they can be added to the context and made
        available to futher event handlers.
        
        Sample usage:
        
        // Verify that the given subject is indeed present in the behaviour domain
        auto verifySubject = [=] (const RestServiceEndpoint::ConnectionId & connection,
                                  const RestRequest & request,
                                  RestRequestParsingContext & context)
        {
            // Grab the dataset, which was matched previously with an addObject
            Dataset & dataset = context.getObjectAs<Dataset>();

            // Find the subject we just parsed from the path
            string subject = context.resources.back();

            SubjectId sid = SubjectId::fromString(subject);

            bool exists = dataset.knownSubject(sid);

            if (!exists && request.verb != "PUT") {
                Json::Value error;
                error["error"] = "subject '" + subject + "' doesn't exist in dataset";
                connection.sendResponse(404, error);
            }
        };

        auto & subject
            = subjects.addSubRouter(Rx("/([^/]*)", "/<subject>"),
                                    "operations on an individual subject",
                                    verifySubject);
    */
    
    typedef std::function<void(const RestServiceEndpoint::ConnectionId & connection,
                               const RestRequest & request,
                               RestRequestParsingContext & context)> ExtractObject;

    template<typename T>
    static ExtractObject addObject(T * obj)
    {
        return [=] (const RestServiceEndpoint::ConnectionId & connection,
                    const RestRequest & request,
                    RestRequestParsingContext & context)
            {
                context.addObject(obj);
            };
    }

    struct Route {
        PathSpec path;
        RequestFilter filter;
        std::shared_ptr<RestRequestRouter> router;
        ExtractObject extractObject;

        bool matchPath(const RestRequest & request,
                       RestRequestParsingContext & context) const;

        MatchResult process(const RestRequest & request,
                            RestRequestParsingContext & context,
                            const RestServiceEndpoint::ConnectionId & connection) const;

        void
        options(std::set<std::string> & verbsAccepted,
                Json::Value & help,
                const RestRequest & request,
                RestRequestParsingContext & context) const;
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
                         const std::set<std::string> & verbs) const;

    /** Create a generic sub router. */
    RestRequestRouter &
    addSubRouter(PathSpec path, const std::string & description,
                 ExtractObject extractObject = nullptr,
                 std::shared_ptr<RestRequestRouter> subRouter = nullptr);

    OnProcessRequest getStaticRouteHandler(const std::string dir) const;
    void serveStaticDirectory(const std::string & route,
                              const std::string & dir);


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
        return [=] (const ConnectionId & connection,
                    const RestRequest & request,
                    RestRequestParsingContext & context) -> int
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

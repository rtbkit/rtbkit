/* rest_request_router.cc
   Jeremy Barnes, 15 November 2012
   Copyright (c) 2012 Datacratic Inc.  All rights reserved.

*/

#include "rest_request_router.h"
#include "jml/utils/vector_utils.h"
#include "jml/arch/exception_handler.h"
#include "jml/utils/set_utils.h"
#include "jml/utils/file_functions.h"


using namespace std;


namespace Datacratic {


/*****************************************************************************/
/* PATH SPEC                                                                 */
/*****************************************************************************/

std::ostream & operator << (std::ostream & stream, const PathSpec & path)
{
    return stream << path.path;
}


/*****************************************************************************/
/* REQUEST FILTER                                                            */
/*****************************************************************************/

std::ostream & operator << (std::ostream & stream, const RequestFilter & filter)
{
    return stream;
}


/*****************************************************************************/
/* REST REQUEST PARSING CONTEXT                                              */
/*****************************************************************************/

std::ostream & operator << (std::ostream & stream,
                            const RestRequestParsingContext & context)
{
    return stream << context.resources << " " << context.remaining;
}


/*****************************************************************************/
/* REST REQUEST ROUTER                                                       */
/*****************************************************************************/

RestRequestRouter::
RestRequestRouter()
    : terminal(false)
{
}

RestRequestRouter::
RestRequestRouter(const OnProcessRequest & processRequest,
                  const std::string & description,
                  bool terminal,
                  const Json::Value & argHelp)
    : rootHandler(processRequest),
      description(description),
      terminal(terminal),
      argHelp(argHelp)
{
}

RestRequestRouter::
~RestRequestRouter()
{
}
    
RestServiceEndpoint::OnHandleRequest
RestRequestRouter::
requestHandler() const
{
    return std::bind(&RestRequestRouter::handleRequest,
                     this,
                     std::placeholders::_1,
                     std::placeholders::_2);
}

void
RestRequestRouter::
handleRequest(const RestServiceEndpoint::ConnectionId & connection,
              const RestRequest & request) const
{
    //JML_TRACE_EXCEPTIONS(false);

    RestRequestParsingContext context(request);
    MatchResult res = processRequest(connection, request, context);
    if (res == MR_NO) {
        connection.sendErrorResponse(404, "unknown resource " + request.verb + " " + request.resource);
    }
}

static std::string getVerbsStr(const std::set<std::string> & verbs)
{
    string verbsStr;
    for (auto v: verbs) {
        if (!verbsStr.empty())
            verbsStr += ",";
        verbsStr += v;
    }
            
    return verbsStr;
}

RestRequestRouter::
MatchResult
RestRequestRouter::
processRequest(const RestServiceEndpoint::ConnectionId & connection,
               const RestRequest & request,
               RestRequestParsingContext & context) const
{
    bool debug = false;

    if (debug) {
        cerr << "processing request " << request
             << " with context " << context
             << " against route " << description 
             << " with " << subRoutes.size() << " subroutes" << endl;
    }

    if (request.verb == "OPTIONS") {
        Json::Value help;
        std::set<std::string> verbs;

        this->options(verbs, help, request, context);

        RestParams headers = { { "Allow", getVerbsStr(verbs) } };
        
        if (verbs.empty())
            connection.sendHttpResponse(400, "", "", headers);
        else
            connection.sendHttpResponse(200, help.toStyledString(),
                                        "application/json",
                                        headers);
        return MR_YES;
    }

    if (rootHandler && (!terminal || context.remaining.empty()))
        return rootHandler(connection, request, context);

    for (auto & sr: subRoutes) {
        if (debug)
            cerr << "  trying subroute " << sr.router->description << endl;
        try {
            MatchResult mr = sr.process(request, context, connection);
            //cerr << "returned " << mr << endl;
            if (mr == MR_YES || mr == MR_ASYNC || mr == MR_ERROR)
                return mr;
        } catch (const std::exception & exc) {
            connection.sendErrorResponse(500, ML::format("threw exception: %s",
                                                         exc.what()));
        } catch (...) {
            connection.sendErrorResponse(500, "unknown exception");
        }
    }

    return MR_NO;
    //connection.sendErrorResponse(404, "invalid route for "
    //                             + request.resource);
}

void
RestRequestRouter::
options(std::set<std::string> & verbsAccepted,
        Json::Value & help,
        const RestRequest & request,
        RestRequestParsingContext & context) const
{
    for (auto & sr: subRoutes) {
        sr.options(verbsAccepted, help, request, context);
    }
}

bool
RestRequestRouter::Route::
matchPath(const RestRequest & request,
          RestRequestParsingContext & context) const
{
    switch (path.type) {
    case PathSpec::STRING: {
        std::string::size_type pos = context.remaining.find(path.path);
        if (pos == 0) {
            using namespace std;
            //cerr << "context string " << pos << endl;
            context.resources.push_back(path.path);
            context.remaining = string(context.remaining, path.path.size());
            break;
        }
        else return false;
    }
    case PathSpec::REGEX: {
        boost::smatch results;
        bool found
            = boost::regex_search(context.remaining,
                                  results,
                                  path.rex)
            && !results.prefix().matched;  // matches from the start
        
        //cerr << "matching regex " << path.path << " against "
        //     << context.remaining << " with found " << found << endl;
        if (!found)
            return false;
        for (unsigned i = 0;  i < results.size();  ++i)
            context.resources.push_back(results[i]);
        context.remaining = std::string(context.remaining,
                                        results[0].length());
        break;
    }
    case PathSpec::NONE:
    default:
        throw ML::Exception("unknown rest request type");
    }

    return true;
}

RestRequestRouter::MatchResult
RestRequestRouter::Route::
process(const RestRequest & request,
        RestRequestParsingContext & context,
        const RestServiceEndpoint::ConnectionId & connection) const
{
    using namespace std;

    bool debug = false;

    if (debug) {
        cerr << "verb = " << request.verb << " filter.verbs = " << filter.verbs
             << endl;
    }
    if (!filter.verbs.empty()
        && !filter.verbs.count(request.verb))
        return MR_NO;

    // At the end, make sure we put the context back to how it was
    RestRequestParsingContext::StateGuard guard(&context);

    if (!matchPath(request, context))
        return MR_NO;

    if (extractObject)
        extractObject(connection, request, context);

    if (connection.responseSent())
        return MR_YES;

    return router->processRequest(connection, request, context);
}

void
RestRequestRouter::Route::
options(std::set<std::string> & verbsAccepted,
        Json::Value & help,
        const RestRequest & request,
        RestRequestParsingContext & context) const
{
    RestRequestParsingContext::StateGuard guard(&context);

    if (!matchPath(request, context))
        return;

    if (context.remaining.empty()) {
        verbsAccepted.insert(filter.verbs.begin(), filter.verbs.end());

        string path = "";//this->path.getPathDesc();
        Json::Value & sri = help[path + getVerbsStr(filter.verbs)];
        this->path.getHelp(sri);
        filter.getHelp(sri);
        router->getHelp(help, path, filter.verbs);
    }
    router->options(verbsAccepted, help, request, context);
}

void
RestRequestRouter::
addRoute(PathSpec path, RequestFilter filter,
         const std::shared_ptr<RestRequestRouter> & handler,
         ExtractObject extractObject)
{
    if (rootHandler)
        throw ML::Exception("can't add a sub-route to a terminal route");

    Route route;
    route.path = path;
    route.filter = filter;
    route.router = handler;
    route.extractObject = extractObject;

    subRoutes.emplace_back(std::move(route));
}

void
RestRequestRouter::
addRoute(PathSpec path, RequestFilter filter,
         const std::string & description,
         const OnProcessRequest & cb,
         const Json::Value & argHelp,
         ExtractObject extractObject)
{
    addRoute(path, filter,
             std::make_shared<RestRequestRouter>(cb, description, true, argHelp),
             extractObject);
}

void
RestRequestRouter::
addHelpRoute(PathSpec path, RequestFilter filter)
{
    OnProcessRequest helpRoute
        = [=] (const RestServiceEndpoint::ConnectionId & connection,
               const RestRequest & request,
               const RestRequestParsingContext & context)
        {
            Json::Value help;
            getHelp(help, "", set<string>());
            connection.sendResponse(200, help);

            return MR_YES;
        };

    addRoute(path, filter, "Get help on the available API commands",
             helpRoute, Json::Value());
}

void
RestRequestRouter::
getHelp(Json::Value & result, const std::string & currentPath,
        const std::set<std::string> & verbs) const
{
    Json::Value & v = result[(currentPath.empty() ? "" : currentPath + " ")
                             + getVerbsStr(verbs)];

    v["description"] = description;
    if (!argHelp.isNull())
        v["arguments"] = argHelp;
    
    for (unsigned i = 0;  i < subRoutes.size();  ++i) {
        string path = currentPath + subRoutes[i].path.getPathDesc();
        Json::Value & sri = result[(path.empty() ? "" : path + " ")
                                   + getVerbsStr(subRoutes[i].filter.verbs)];
        subRoutes[i].path.getHelp(sri);
        subRoutes[i].filter.getHelp(sri);
        subRoutes[i].router->getHelp(result, path, subRoutes[i].filter.verbs);
    }
}

RestRequestRouter &
RestRequestRouter::
addSubRouter(PathSpec path, const std::string & description, ExtractObject extractObject,
             std::shared_ptr<RestRequestRouter> subRouter)
{
    // TODO: check it doesn't exist
    Route route;
    route.path = path;
    if (subRouter)
        route.router = subRouter;
    else route.router.reset(new RestRequestRouter());

    route.router->description = description;
    route.extractObject = extractObject;

    subRoutes.push_back(route);
    return *route.router;
}

RestRequestRouter::OnProcessRequest
RestRequestRouter::
getStaticRouteHandler(const string dir) const {
    RestRequestRouter::OnProcessRequest staticRoute
        = [dir] (const RestServiceEndpoint::ConnectionId & connection,
                 const RestRequest & request,
                 const RestRequestParsingContext & context) {

        string path = context.resources.back();

        cerr << "static content for " << path << endl;

        if (path.find("..") != string::npos) {
            throw ML::Exception("not dealing with path with .. in it");
        }

        string filename = dir + "/" + path;

        ML::filter_istream stream(filename);
        ML::File_Read_Buffer buf(filename);

        string mimeType = "text/plain";
        if (filename.find(".html") != string::npos) {
            mimeType = "text/html";
        }
        else if (filename.find(".js") != string::npos) {
            mimeType = "application/javascript";
        }
        else if (filename.find(".css") != string::npos) {
            mimeType = "text/css";
        }

        string result(buf.start(), buf.end());
        connection.sendResponse(200, result, mimeType);
        return RestRequestRouter::MR_YES;
    };
    return staticRoute;
}

void RestRequestRouter::
serveStaticDirectory(const std::string & route, const std::string & dir) {
    addRoute(Rx(route + "/(.*)", "<resource>"),
             "GET", "Static content",
             getStaticRouteHandler(dir),
             Json::Value());
}


} // namespace Datacratic

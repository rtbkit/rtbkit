/** rest_request_binding.h                                         -*- C++ -*-
    Jeremy Barnes, 15 November 2012
    Copyright (c) 2012 Datacratic.  All rights reserved.

    Functionality to bind arbitrary functions into REST requests in a
    declarative manner.

    Warning: full of funky variadic template voodoo.
*/

#pragma once

#include "rest_request_router.h"
#include "jml/arch/demangle.h"
#include <boost/lexical_cast.hpp>
#include "json_codec.h"
#include "soa/types/value_description.h"
#include "soa/types/json_printing.h"

namespace Datacratic {


/*****************************************************************************/
/* PARTIAL BIND                                                              */
/*****************************************************************************/

/** These functions bind the first 0 or 1 arguments of a callable and leave
    the rest unbound.

    Useful for when you don't know the number of arguments, so you can't
    write std::placeholder::_1, etc.
*/

template<typename T>
T partial_bind(T fn)
{
    return fn;
}

template<typename R, typename... Args, typename Obj, typename Ptr>
std::function<R (Args...)>
partial_bind(R (Obj::* pmf) (Args...) const,
             Ptr ptr)
{
    return [=] (Args&&... args) -> R
        {
            const Obj & obj = *ptr;
            return ((obj).*(pmf))(std::forward<Args>(args)...);
        };
}

template<typename R, typename... Args, typename Obj, typename Ptr>
std::function<R (Args...)>
partial_bind(R (Obj::* pmf) (Args...),
             Ptr ptr)
{
    return [=] (Args&&... args) -> R
        {
            const Obj & obj = *ptr;
            return ((obj).*(pmf))(std::forward<Args>(args)...);
        };
}

template<typename R, typename Arg1, typename... Args, typename Bind1>
std::function<R (Args...)>
partial_bind(const std::function<R (Arg1, Args...)> & fn, Bind1 && bind1)
{
    return [=] (Args&&... args) -> R
        {
            return fn(bind1, std::forward<Args>(args)...);
        };
}


/*****************************************************************************/
/* REST CODEC                                                                */
/*****************************************************************************/

/** Default parameter encoder / decoder that uses a lexical cast to convert
    to the required type.
*/
template<typename T>
decltype(boost::lexical_cast<T>(std::declval<std::string>()))
restDecode(const std::string & str, T * = 0)
{
    return boost::lexical_cast<T>(str);
}

template<typename T>
std::string restEncode(const T & val,
                       decltype(boost::lexical_cast<std::string>(std::declval<T>())) * = 0)
{
    return boost::lexical_cast<std::string>(val);
}

template<typename T, typename Enable = void>
struct RestCodec {
    static T decode(const std::string & str)
    {
        return restDecode(str, (T *)0);
    }

    static std::string encode(const T & val)
    {
        return restEncode(val);
    }
};

/*****************************************************************************/
/* JSON STR CODEC                                                            */
/*****************************************************************************/

/** Parameter encoder / decoder that takes parameters encoded in JSON as a
    string and converts them into the required type.
*/

template<typename T>
struct JsonStrCodec {
    static T decode(const std::string & str)
    {
        return jsonDecodeStr<T>(str);
    }

    static std::string encode(const T & t)
    {
        return jsonEncodeStr(t);
    }
};


/*****************************************************************************/
/* REST PARAMETER SELECTORS                                                  */
/*****************************************************************************/

/** This indicates that we get a parameter from the query string.  If the
    parameter is not present, we throw an exception.
*/
template<typename T, typename Codec = RestCodec<T> >
struct RestParam {
    RestParam(const std::string & name, const std::string & description)
        : name(name), description(description)
    {
        //std::cerr << "created RestParam with " << name << " at "
        //          << this << std::endl;
    }
    
    RestParam(const RestParam & other)
        : name(other.name), description(other.description)
    {
        //std::cerr << "copied RestParam with " << name << " to "
        //          << this << std::endl;
    }

    std::string name;
    std::string description;

private:
    void operator = (const RestParam & other);
};


/** This indicates that we get a parameter from the query string.  If the
    parameter is not present, we use a default value.
*/
template<typename T, typename Codec = RestCodec<T> >
struct RestParamDefault {
    RestParamDefault(const std::string & name,
                     const std::string & description,
                     T defaultValue,
                     const std::string & defaultValueStr)
        : name(name), description(description), defaultValue(defaultValue),
          defaultValueStr(defaultValueStr)
    {
        //std::cerr << "created RestParam with " << name << " at "
        //          << this << std::endl;
    }

    RestParamDefault(const std::string & name,
                     const std::string & description,
                     T defaultValue = T())
        : name(name), description(description), defaultValue(defaultValue),
          defaultValueStr(boost::lexical_cast<std::string>(defaultValue))
    {
        //std::cerr << "created RestParam with " << name << " at "
        //          << this << std::endl;
    }
    
    RestParamDefault(const RestParamDefault & other)
        : name(other.name), description(other.description),
          defaultValue(other.defaultValue),
          defaultValueStr(other.defaultValueStr)
    {
        //std::cerr << "copied RestParam with " << name << " to "
        //          << this << std::endl;
    }

    std::string name;
    std::string description;
    T defaultValue;
    std::string defaultValueStr;

private:
    void operator = (const RestParamDefault & other);
};


/** This indicates that we get a parameter from the query string and decode
    it from JSON.  If the parameter is not present, we throw an exception.
*/
template<typename T, typename Codec = JsonStrCodec<T> >
struct RestParamJson : public RestParam<T, Codec> {
    RestParamJson(const std::string & name, const std::string & description)
        : RestParam<T, Codec>(name, description)
    {
    }
    
private:
    void operator = (const RestParamJson & other);
};


/** This indicates that we get a parameter from the query string.  If the
    parameter is not present, we use a default value.  Encoding/decoding
    is done through JSON.
*/
template<typename T, typename Codec = JsonStrCodec<T> >
struct RestParamJsonDefault : public RestParamDefault<T, Codec> {
    RestParamJsonDefault(const std::string & name,
                         const std::string & description,
                         T defaultValue = T(),
                         const std::string & defaultValueStr = "")
        : RestParamDefault<T, Codec>(name, description, defaultValue,
                                     defaultValueStr)
    {
    }
    
private:
    void operator = (const RestParamJsonDefault & other);
};


/** This indicates that we get a parameter from the JSON payload.
    If name is empty, then we get the whole payload.  Otherwise we
    get the named value of the payload.
*/
template<typename T>
struct JsonParam {
    JsonParam(const std::string & name, const std::string & description)
        : name(name), description(description)
    {
    }
    
    JsonParam(const JsonParam & other)
        : name(other.name), description(other.description)
    {
    }
    
    std::string name;
    std::string description;
};


/** This indicates that we get a parameter from the path of the request. 
    For example, GET /v1/object/3/value, this would be able to bind "3" to
    a parameter of the request.
*/
template<typename T, class Codec = RestCodec<T> >
struct RequestParam {
    RequestParam(int index, const std::string & name, const std::string & description)
        : index(index), name(name), description(description)
    {
    }

    RequestParam(const RequestParam & other)
        : index(other.index),
          name(other.name),
          description(other.description)
    {
    }

    int index;
    std::string name;
    std::string description;
};


/*****************************************************************************/
/* CREATE PARAMETER EXTRACTOR                                                */
/*****************************************************************************/

/** These functions turn an argument to the request binding into a function
    that can generate the value required by the handler function.

*/

#if 0
/** By default, to create a parameter extractor we simply take a copy of the
    argument and return that.

    This is used to pass through direct parameter values.
*/
template<typename T>
std::function<T (const RestServiceEndpoint::ConnectionId & connection,
                 const RestRequest & request,
                 const RestRequestParsingContext & context)>
createParameterExtractor(Json::Value & argHelp, const T & p, void * = 0)
{
    return [=] (const RestServiceEndpoint::ConnectionId & connection,
                const RestRequest & request,
                const RestRequestParsingContext & context)
        {
            return p;
        };
}
#endif

/** Free function to take the payload and pass it as a string. */
struct StringPayload {
    StringPayload(const std::string & description)
        : description(description)
    {
    }

    std::string description;
};

inline static std::function<std::string
                     (const RestServiceEndpoint::ConnectionId & connection,
                      const RestRequest & request,
                      const RestRequestParsingContext & context)>
createParameterExtractor(Json::Value & argHelp,
                         const StringPayload & p, void * = 0)
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

struct PassConnectionId {
};

/** Pass the connection on */
inline static std::function<const RestServiceEndpoint::ConnectionId &
                     (const RestServiceEndpoint::ConnectionId & connection,
                      const RestRequest & request,
                      const RestRequestParsingContext & context)>
createParameterExtractor(Json::Value & argHelp,
                         const PassConnectionId &, void * = 0)
{
    return [] (const RestServiceEndpoint::ConnectionId & connection,
                const RestRequest & request,
                const RestRequestParsingContext & context)
        -> const RestServiceEndpoint::ConnectionId &
        {
            return connection;
        };
}


struct PassParsingContext {
};

/** Pass the connection on */
inline static std::function<const RestRequestParsingContext &
                     (const RestServiceEndpoint::ConnectionId & connection,
                      const RestRequest & request,
                      const RestRequestParsingContext & context)>
createParameterExtractor(Json::Value & argHelp,
                         const PassParsingContext &, void * = 0)
{
    return [] (const RestServiceEndpoint::ConnectionId & connection,
                const RestRequest & request,
                const RestRequestParsingContext & context)
        -> const RestRequestParsingContext &
        {
            return context;
        };
}

struct PassRequest {
};

/** Pass the connection on */
inline static std::function<const RestRequest &
                     (const RestServiceEndpoint::ConnectionId & connection,
                      const RestRequest & request,
                      const RestRequestParsingContext & context)>
createParameterExtractor(Json::Value & argHelp,
                         const PassRequest &, void * = 0)
{
    return [] (const RestServiceEndpoint::ConnectionId & connection,
               const RestRequest & request,
               const RestRequestParsingContext & context)
        -> const RestRequest &
        {
            return request;
        };
}

/** Free function to be called in order to generate a parameter extractor
    for the given parameter.  See the CreateRestParameterGenerator class for more
    details.
*/
template<typename T, typename Codec>
static std::function<decltype(Codec::decode(std::declval<std::string>()))
                     (const RestServiceEndpoint::ConnectionId & connection,
                      const RestRequest & request,
                      const RestRequestParsingContext & context)>
createParameterExtractor(Json::Value & argHelp,
                         const RestParam<T, Codec> & p, void * = 0)
{
    ExcAssertNotEqual(p.name, "");

    Json::Value & v = argHelp["requestParams"];
    Json::Value & v2 = v[v.size()];
    if (!p.name.empty())
        v2["name"] = p.name;
    v2["description"] = p.description;
    v2["cppType"] = ML::type_name<T>();
    v2["encoding"] = "URI encoded";
    v2["location"] = "query string";

    return [=] (const RestServiceEndpoint::ConnectionId & connection,
                const RestRequest & request,
                const RestRequestParsingContext & context)
        {
            //std::cerr << "getting value of " << p.name << std::endl;
            std::string paramValue = request.params.getValue(p.name);
            return Codec::decode(paramValue);
        };
}

template<typename T, typename Codec>
static std::function<T (const RestServiceEndpoint::ConnectionId & connection,
                        const RestRequest & request,
                        const RestRequestParsingContext & context)>
createParameterExtractor(Json::Value & argHelp,
                         const RestParamDefault<T, Codec> & p, void * = 0)
{
    ExcAssertNotEqual(p.name, "");

    Json::Value & v = argHelp["requestParams"];
    Json::Value & v2 = v[v.size()];
    if (!p.name.empty())
        v2["name"] = p.name;
    v2["description"] = p.description;
    v2["cppType"] = ML::type_name<T>();
    v2["encoding"] = "URI encoded";
    v2["location"] = "query string";
    v2["defaultValue"] = p.defaultValueStr;

    return [=] (const RestServiceEndpoint::ConnectionId & connection,
                const RestRequest & request,
                const RestRequestParsingContext & context)
        {
            //std::cerr << "getting value of " << p.name << std::endl;
            std::string paramValue;
            T result;
            if (request.params.hasValue(p.name)) 
                result = Codec::decode(request.params.getValue(p.name));
            else result = p.defaultValue;
            return result;
        };
}

/** Free function to be called in order to generate a parameter extractor
    for the given parameter.  See the CreateRestParameterGenerator class for more
    details.
*/
template<typename T>
static std::function<decltype(JsonCodec<T>::decode(std::declval<Json::Value>()))
                     (const RestServiceEndpoint::ConnectionId & connection,
                      const RestRequest & request,
                      const RestRequestParsingContext & context)>
createParameterExtractor(Json::Value & argHelp,
                         const JsonParam<T> & p, void * = 0)
{
    Json::Value & v = argHelp["jsonParams"];
    Json::Value & v2 = v[v.size()];
    if (!p.name.empty())
        v2["name"] = p.name;
    v2["description"] = p.description;
    v2["cppType"] = ML::type_name<T>();
    v2["encoding"] = "JSON";
    v2["location"] = "Request Body";

    return [=] (const RestServiceEndpoint::ConnectionId & connection,
                const RestRequest & request,
                const RestRequestParsingContext & context)
        {
            Json::Value parsed = Json::parse(request.payload);
            return JsonCodec<T>::decode(p.name.empty() ? parsed : parsed[p.name]);
        };
}

/** Free function to be called in order to generate a parameter extractor
    for the given parameter.  See the CreateRestParameterGenerator class for more
    details.
*/
template<typename T, typename Codec>
static std::function<decltype(Codec::decode(std::declval<std::string>()))
                     (const RestServiceEndpoint::ConnectionId & connection,
                      const RestRequest & request,
                      const RestRequestParsingContext & context)>
createParameterExtractor(Json::Value & argHelp,
                         const RequestParam<T, Codec> & p, void * = 0)
{
    Json::Value & v = argHelp["resourceParams"];
    Json::Value & v2 = v[v.size()];
    if (!p.name.empty())
        v2["name"] = p.name;
    v2["description"] = p.description;
    v2["cppType"] = ML::type_name<T>();
    v2["encoding"] = "URI encoded";
    v2["location"] = "URI";

    return [=] (const RestServiceEndpoint::ConnectionId & connection,
                const RestRequest & request,
                const RestRequestParsingContext & context)
        {
            int index = p.index;
            //using namespace std;
            //cerr << "index " << index << " with "
            //     << context.resources.size() << " resources" << endl;
            if (index < 0)
                index = context.resources.size() + index;
            std::string paramValue = context.resources.at(index);
            return Codec::decode(paramValue);
        };
}

/** Parameter extractor to generate something of the type

    std::function<void (X)>

    from

    std::function<void (X, ConnectionId, RestRequest) >

    by binding in the last two parameters.

    This is used to create a callback to an asynchronous function that will
    finish off a request by sending back it's results.
*/
template<typename Fn, typename Return, typename... Args>
std::function<std::function<Return (Args...)>
              (const RestServiceEndpoint::ConnectionId & connection,
               const RestRequest & request,
               const RestRequestParsingContext & context)>
createParameterExtractor(Json::Value argHelp,
                         const Fn & fn, std::function<Return (Args...)> * = 0)
{
    return [=] (const RestServiceEndpoint::ConnectionId & connection,
                const RestRequest & request,
                const RestRequestParsingContext & context)
        {
            // TODO: deal with more/less than one parameter...
            return std::bind(fn, std::placeholders::_1, connection, request);
        };
}

/// Type of the function to handle an exception in the REST binding
typedef std::function<RestRequestRouter::MatchResult
                      (const std::string & excStr,
                       std::exception_ptr exc,
                       const RestServiceEndpoint::ConnectionId & connection,
                       const RestRequest & request,
                       const RestRequestParsingContext & context)> ExcFn;

/*************************************************************************/
/* CREATE GENERATOR                                                      */
/*************************************************************************/

/** When we bind a callback to a request, we need to generate all of the
    parameters for the callback.  These parameters are generated from the
    request contents.

    For example,

    if we bind cb : (x, y, z) -> result

    when we get a request r, we need to call

    cb (genx(r), geny(r), genz(r) )

    to create the parameters for the callback.

    This happens in two steps:

    1.  We create a list of generators (genx, geny, genz).  This is done
    at registration time.
    2.  When we get a request (and the parameter r), we then call each
    of the generators in turn with this parameter to generate the
    argument list to be passed to the callback.

    This class deals with creating and calling the generator for a given
    position it will be called for each of (genx, geny, genz).  The RestRequestBinder
    class is used to actually group them and apply them together to
    generate the callback arguments.
*/
        
template<typename X, typename... Params>
struct CreateRestParameterGenerator {
};

template<int Index, typename Arg, typename Param, typename... Params>
struct CreateRestParameterGenerator<ML::PositionedDualType<Index, Arg, Param>, Params...> {

    typedef decltype(createParameterExtractor(*(Json::Value *)0, std::declval<typename ML::ExtractArgAtPosition<0, Index, Params...>::type>(), (typename std::decay<Arg>::type *)0)) Generator;

    //typedef std::decay<Arg> Result;
    typedef decltype(std::declval<Generator>()
                     (std::declval<RestServiceEndpoint::ConnectionId>(),
                      std::declval<RestRequest>(),
                      std::declval<RestRequestParsingContext>())) Result;

    /** Create the generator */
    static Generator create(Json::Value & argHelp, Params&&... params)
    {
        auto param = ML::ExtractArgAtPosition<0, Index, Params...>
            ::extract(std::forward<Params>(params)...);
        return createParameterExtractor(argHelp,
                                        param,
                                        (typename std::decay<Arg>::type *)0);
    }

    /** Apply our generator (which is at index Index within gens,
        a std::tuple<...> of parameter generators) to the given
        rest request in order to generate a parameter for a
        callback.
    */
    template<typename Generators>
    static Result apply(const Generators & gens,
                        const RestServiceEndpoint::ConnectionId & connection,
                        const RestRequest & request,
                        const RestRequestParsingContext & context)
    {
        return std::get<Index>(gens)(connection, request, context);
    }
};

template<typename T>
struct RestRequestBinder {
};

template<typename... PositionedDualTypes>
struct RestRequestBinder<ML::TypeList<PositionedDualTypes...> > {

    /** Create a request handler that will call the given member
        function with parameters extracted from the request.
    */
    template<typename Return, typename Obj, typename... Args, typename Ptr,
             typename... Params>
    static
    std::pair<RestRequestRouter::OnProcessRequest, Json::Value>
    bindSync(Return (Obj::* pmf) (Args...),
             Ptr ptr,
             Params&&... params)
    {
        Json::Value argHelp;

        // Create a tuple of function objects that we can call with
        auto gens = std::make_tuple(CreateRestParameterGenerator<PositionedDualTypes, Params...>
                                    ::create(argHelp, std::forward<Params>(params)...)...);
        // Necessary to deal with a compiler bug
        auto sharedGens = std::make_shared<decltype(gens)>(std::move(gens));

        RestRequestRouter::OnProcessRequest result
            = [=] (const RestServiceEndpoint::ConnectionId & connection,
                   const RestRequest & request,
                   const RestRequestParsingContext & context)
            {
                auto gens = *sharedGens;
                try {
                    Obj & obj = *ptr;
                    ((obj).*(pmf))(CreateRestParameterGenerator<PositionedDualTypes, Params...>
                                   ::apply(gens, connection, request, context)...
                                   );
                        
                    connection.sendResponse(200);
                } catch (const std::exception & exc) {
                    connection.sendErrorResponse(400, exc.what());
                    return RestRequestRouter::MR_ERROR;
                } catch (...) {
                    connection.sendErrorResponse(400, "unknown exception");
                    return RestRequestRouter::MR_ERROR;
                }

                return RestRequestRouter::MR_YES;
            };
            
        return make_pair(result, argHelp);
    }


    template<typename Return, typename Obj, typename... Args, typename Ptr,
             typename... Params>
    static
    std::pair<RestRequestRouter::OnProcessRequest, Json::Value>
    bindSync(Return (Obj::* pmf) (Args...) const,
             Ptr ptr,
             Params&&... params)
    {
        Json::Value argHelp;

        // Create a tuple of function objects that we can call with
        auto gens = std::make_tuple(CreateRestParameterGenerator<PositionedDualTypes, Params...>
                                    ::create(argHelp, std::forward<Params>(params)...)...);
        // Necessary to deal with a compiler bug
        auto sharedGens = std::make_shared<decltype(gens)>(std::move(gens));

        RestRequestRouter::OnProcessRequest result
            = [=] (const RestServiceEndpoint::ConnectionId & connection,
                   const RestRequest & request,
                   const RestRequestParsingContext & context)
            {
                auto gens = *sharedGens;
                try {
                    Obj & obj = *ptr;
                    ((obj).*(pmf))(CreateRestParameterGenerator<PositionedDualTypes, Params...>
                                   ::apply(gens, connection, request, context)...
                                   );
                        
                    connection.sendResponse(200);
                } catch (const std::exception & exc) {
                    connection.sendErrorResponse(400, exc.what());
                    return RestRequestRouter::MR_ERROR;
                } catch (...) {
                    connection.sendErrorResponse(400, "unknown exception");
                    return RestRequestRouter::MR_ERROR;
                }

                return RestRequestRouter::MR_YES;
            };
            
        return make_pair(result, argHelp);
    }

    /** Create a request handler that will call the given member
        function with parameters extracted from the request.
    */
    template<class TransformResultFn,
             typename Return, typename Obj, typename... Args, typename Ptr,
             typename... Params>
    static
    std::pair<RestRequestRouter::OnProcessRequest, Json::Value>
    bindSyncReturn(const TransformResultFn & fn,
                   Return (Obj::* pmf) (Args...),
                   Ptr ptr,
                   Params&&... params)
    {
        Json::Value argHelp;

        // Create a tuple of function objects that we can call with
        auto gens = std::make_tuple(CreateRestParameterGenerator<PositionedDualTypes, Params...>
                                    ::create(argHelp, std::forward<Params>(params)...)...);
        // Necessary to deal with a compiler bug
        auto sharedGens = std::make_shared<decltype(gens)>(std::move(gens));

        RestRequestRouter::OnProcessRequest result
            = [=] (const RestServiceEndpoint::ConnectionId & connection,
                   const RestRequest & request,
                   const RestRequestParsingContext & context)
            {
                auto gens = *sharedGens;
                try {
                    Obj & obj = *ptr;
                    auto res = ((obj).*(pmf))(CreateRestParameterGenerator<PositionedDualTypes, Params...>
                                              ::apply(gens, connection, request, context)...
                                              );
                    connection.sendResponse(200, fn(res));
                } catch (const std::exception & exc) {
                    connection.sendErrorResponse(400, exc.what());
                    return RestRequestRouter::MR_ERROR;
                } catch (...) {
                    connection.sendErrorResponse(400, "unknown exception");
                    return RestRequestRouter::MR_ERROR;
                }

                return RestRequestRouter::MR_YES;
            };
            
        return make_pair(result, argHelp);
    }


    template<class TransformResultFn,
             typename Return, typename Obj, typename... Args, typename Ptr,
             typename... Params>
    static
    std::pair<RestRequestRouter::OnProcessRequest, Json::Value>
    bindSyncReturn(const TransformResultFn & fn,
                   Return (Obj::* pmf) (Args...) const,
                   Ptr ptr,
                   Params&&... params)
    {
        Json::Value argHelp;

        // Create a tuple of function objects that we can call with
        auto gens = std::make_tuple(CreateRestParameterGenerator<PositionedDualTypes, Params...>
                                    ::create(argHelp, std::forward<Params>(params)...)...);
        // Necessary to deal with a compiler bug
        auto sharedGens = std::make_shared<decltype(gens)>(std::move(gens));

        RestRequestRouter::OnProcessRequest result
            = [=] (const RestServiceEndpoint::ConnectionId & connection,
                   const RestRequest & request,
                   const RestRequestParsingContext & context)
            {
                auto gens = *sharedGens;
                try {
                    Obj & obj = *ptr;
                    auto res = ((obj).*(pmf))(CreateRestParameterGenerator<PositionedDualTypes, Params...>
                                              ::apply(gens, connection, request, context)...
                                              );
                    
                    connection.sendResponse(200, fn(res));
                } catch (const std::exception & exc) {
                    connection.sendErrorResponse(400, exc.what());
                    return RestRequestRouter::MR_ERROR;
                } catch (...) {
                    connection.sendErrorResponse(400, "unknown exception");
                    return RestRequestRouter::MR_ERROR;
                }

                return RestRequestRouter::MR_YES;
            };
            
        return make_pair(result, argHelp);
    }


    /** Create a request handler that will call the given member
        function with parameters extracted from the request.
    */
    template<typename Return, typename Obj, typename... Args, typename Ptr,
             typename... Params>
    static
    std::pair<RestRequestRouter::OnProcessRequest, Json::Value>
    bindSyncReturnStatus(std::pair<int, Return> (Obj::* pmf) (Args...),
                         Ptr ptr,
                         Params&&... params)
    {
        Json::Value argHelp;

        // Create a tuple of function objects that we can call with
        auto gens = std::make_tuple(CreateRestParameterGenerator<PositionedDualTypes, Params...>
                                    ::create(argHelp, std::forward<Params>(params)...)...);
        // Necessary to deal with a compiler bug
        auto sharedGens = std::make_shared<decltype(gens)>(std::move(gens));

        RestRequestRouter::OnProcessRequest result
            = [=] (const RestServiceEndpoint::ConnectionId & connection,
                   const RestRequest & request,
                   const RestRequestParsingContext & context)
            {
                auto gens = *sharedGens;
                try {
                    Obj & obj = *ptr;
                    auto res = ((obj).*(pmf))(CreateRestParameterGenerator<PositionedDualTypes, Params...>
                                              ::apply(gens, connection, request, context)...
                                              );
                    connection.sendResponse(res.first, res.second);
                } catch (const std::exception & exc) {
                    connection.sendErrorResponse(400, exc.what());
                    return RestRequestRouter::MR_ERROR;
                } catch (...) {
                    connection.sendErrorResponse(400, "unknown exception");
                    return RestRequestRouter::MR_ERROR;
                }

                return RestRequestRouter::MR_YES;
            };
            
        return make_pair(result, argHelp);
    }


    template<typename Return, typename Obj, typename... Args, typename Ptr,
             typename... Params>
    static
    std::pair<RestRequestRouter::OnProcessRequest, Json::Value>
    bindSyncReturnStatus(Return (Obj::* pmf) (Args...) const,
                         Ptr ptr,
                         Params&&... params)
    {
        Json::Value argHelp;

        // Create a tuple of function objects that we can call with
        auto gens = std::make_tuple(CreateRestParameterGenerator<PositionedDualTypes, Params...>
                                    ::create(argHelp, std::forward<Params>(params)...)...);
        // Necessary to deal with a compiler bug
        auto sharedGens = std::make_shared<decltype(gens)>(std::move(gens));

        RestRequestRouter::OnProcessRequest result
            = [=] (const RestServiceEndpoint::ConnectionId & connection,
                   const RestRequest & request,
                   const RestRequestParsingContext & context)
            {
                auto gens = *sharedGens;
                try {
                    Obj & obj = *ptr;
                    auto res = ((obj).*(pmf))(CreateRestParameterGenerator<PositionedDualTypes, Params...>
                                              ::apply(gens, connection, request, context)...
                                              );
                    
                    connection.sendResponse(res.first, res.second);
                } catch (const std::exception & exc) {
                    connection.sendErrorResponse(400, exc.what());
                    return RestRequestRouter::MR_ERROR;
                } catch (...) {
                    connection.sendErrorResponse(400, "unknown exception");
                    return RestRequestRouter::MR_ERROR;
                }

                return RestRequestRouter::MR_YES;
            };
            
        return make_pair(result, argHelp);
    }


    /** Create a request handler that will call the given member
        function with parameters extracted from the request.
    */
    template<typename Return, typename Obj, typename... Args, typename Ptr,
             typename... Params>
    static
    std::pair<RestRequestRouter::OnProcessRequest, Json::Value>
    bindAsync(Return (Obj::* pmf) (Args...),
              Ptr ptr,
              Params&&... params)
    {
        Json::Value argHelp;

        // Create a tuple of function objects that we can call with
        auto gens = std::make_tuple(CreateRestParameterGenerator<PositionedDualTypes, Params...>
                                    ::create(argHelp, std::forward<Params>(params)...)...);
        // Necessary to deal with a compiler bug
        auto sharedGens = std::make_shared<decltype(gens)>(std::move(gens));

        RestRequestRouter::OnProcessRequest result
            = [=] (const RestServiceEndpoint::ConnectionId & connection,
                   const RestRequest & request,
                   const RestRequestParsingContext & context)
            {
                auto gens = *sharedGens;
                try {
                    Obj & obj = *ptr;
                    ((obj).*(pmf))(CreateRestParameterGenerator<PositionedDualTypes, Params...>
                                   ::apply(gens, connection, request, context)...
                                   );
                } catch (const std::exception & exc) {
                    connection.sendErrorResponse(400, exc.what());
                    return RestRequestRouter::MR_ERROR;
                } catch (...) {
                    connection.sendErrorResponse(400, "unknown exception");
                    return RestRequestRouter::MR_ERROR;
                }

                return RestRequestRouter::MR_YES;
            };
            
        return make_pair(result, argHelp);
    }

    /** Create a request handler that will call the given member
        function with parameters extracted from the request.
    */
    template<typename Return, typename... Args,
             typename... Params, typename ThenFn>
    static
    std::pair<RestRequestRouter::OnProcessRequest, Json::Value>
    bindAsyncThen(const ThenFn & then,
                  const ExcFn & excFn,
                  const std::function<Return (Args...)> & fn,
                  Params&&... params)
    {
        Json::Value argHelp;

        // Create a tuple of function objects that we can call with
        auto gens = std::make_tuple(CreateRestParameterGenerator<PositionedDualTypes, Params...>
                                    ::create(argHelp, std::forward<Params>(params)...)...);
        // Necessary to deal with a compiler bug
        auto sharedGens = std::make_shared<decltype(gens)>(std::move(gens));

        RestRequestRouter::OnProcessRequest result
            = [=] (const RestServiceEndpoint::ConnectionId & connection,
                   const RestRequest & request,
                   const RestRequestParsingContext & context) -> RestRequestRouter::MatchResult
            {
                auto gens = *sharedGens;
                try {
                    return then(fn(CreateRestParameterGenerator<PositionedDualTypes, Params...>
                                   ::apply(gens, connection, request, context)...
                                   ),
                                connection, request, context);
                } catch (const std::exception & exc) {
                    if (excFn)
                        return excFn(exc.what(), std::current_exception(), 
                                     connection, request, context);
                    connection.sendErrorResponse(400, exc.what());
                    return RestRequestRouter::MR_ERROR;
                } catch (...) {
                    if (excFn)
                        return excFn("unknown exception", std::current_exception(),
                                     connection, request, context);
                    connection.sendErrorResponse(400, "unknown exception");
                    return RestRequestRouter::MR_ERROR;
                }
            };
            
        return make_pair(result, argHelp);
    }

};

template<typename Return, typename Obj, typename... Args, typename Ptr,
         typename TransformResult,
         typename... Params>
void
addRouteSyncReturn(RestRequestRouter & router,
                   PathSpec path, RequestFilter filter,
                   const std::string & description,
                   const std::string & resultDescription,
                   const TransformResult & transformResult,
                   Return (Obj::* pmf) (Args...),
                   Ptr ptr,
                   Params&&... params)
{
    static_assert(sizeof...(Args) == sizeof...(Params),
                  "member function and parameter arity must match");

    typedef ML::TypeList<Args...> ArgsList;
    typedef ML::TypeList<Params...> ParamsList;
    typedef ML::PositionedDualTypeList<0, ArgsList, ParamsList> PositionedTypes;

    auto res = RestRequestBinder<typename PositionedTypes::List>
        ::bindSyncReturn(transformResult,
                         pmf, ptr, std::forward<Params>(params)...);
    auto & cb = res.first;
    auto & help = res.second;
    help["result"] = resultDescription;

    router.addRoute(path, filter, description, cb, help);
}

template<typename Return, typename Obj, typename... Args, typename Ptr,
         typename TransformResult,
         typename... Params>
void
addRouteSyncReturn(RestRequestRouter & router,
                   PathSpec path, RequestFilter filter,
                   const std::string & description,
                   const std::string & resultDescription,
                   const TransformResult & transformResult,
                   Return (Obj::* pmf) (Args...) const,
                   Ptr ptr,
                   Params&&... params)
{
    static_assert(sizeof...(Args) == sizeof...(Params),
                  "member function and parameter arity must match");

    typedef ML::TypeList<Args...> ArgsList;
    typedef ML::TypeList<Params...> ParamsList;
    typedef ML::PositionedDualTypeList<0, ArgsList, ParamsList> PositionedTypes;

    auto res = RestRequestBinder<typename PositionedTypes::List>
        ::bindSyncReturn(transformResult,
                         pmf, ptr, std::forward<Params>(params)...);
    auto & cb = res.first;
    auto & help = res.second;
    help["result"] = resultDescription;

    router.addRoute(path, filter, description, cb, help);
}

template<typename Return, typename Obj, typename... Args, typename Ptr,
         typename... Params>
void
addRouteReturnStatus(RestRequestRouter & router,
                         PathSpec path, RequestFilter filter,
                         const std::string & description,
                         const std::string & resultDescription,
                         std::pair<int, Return> (Obj::* pmf) (Args...),
                         Ptr ptr,
                         Params&&... params)
{
    static_assert(sizeof...(Args) == sizeof...(Params),
                  "member function and parameter arity must match");

    typedef ML::TypeList<Args...> ArgsList;
    typedef ML::TypeList<Params...> ParamsList;
    typedef ML::PositionedDualTypeList<0, ArgsList, ParamsList> PositionedTypes;

    auto res = RestRequestBinder<typename PositionedTypes::List>
        ::bindSyncReturnStatus(pmf, ptr, std::forward<Params>(params)...);
    auto & cb = res.first;
    auto & help = res.second;
    help["result"] = resultDescription;

    router.addRoute(path, filter, description, cb, help);
}

template<typename Return, typename Obj, typename... Args, typename Ptr,
         typename... Params>
void
addRouteReturnStatus(RestRequestRouter & router,
                         PathSpec path, RequestFilter filter,
                         const std::string & description,
                         const std::string & resultDescription,
                         std::pair<int, Return> (Obj::* pmf) (Args...) const,
                         Ptr ptr,
                         Params&&... params)
{
    static_assert(sizeof...(Args) == sizeof...(Params),
                  "member function and parameter arity must match");

    typedef ML::TypeList<Args...> ArgsList;
    typedef ML::TypeList<Params...> ParamsList;
    typedef ML::PositionedDualTypeList<0, ArgsList, ParamsList> PositionedTypes;

    auto res = RestRequestBinder<typename PositionedTypes::List>
        ::bindSyncReturnStatus(pmf, ptr, std::forward<Params>(params)...);
    auto & cb = res.first;
    auto & help = res.second;
    help["result"] = resultDescription;

    router.addRoute(path, filter, description, cb, help);
}

// Void return types don't need to convert their result
template<typename Return, typename Obj, typename... Args, typename Ptr,
         typename... Params>
void
addRouteSync(RestRequestRouter & router,
             PathSpec path, RequestFilter filter,
             const std::string & description,
             Return (Obj::* pmf) (Args...),
             Ptr ptr,
             Params&&... params)
{
    static_assert(sizeof...(Args) == sizeof...(Params),
                  "member function and parameter arity must match");

    typedef ML::TypeList<Args...> ArgsList;
    typedef ML::TypeList<Params...> ParamsList;
    typedef ML::PositionedDualTypeList<0, ArgsList, ParamsList> PositionedTypes;

    auto res = RestRequestBinder<typename PositionedTypes::List>
        ::bindSync(pmf, ptr, std::forward<Params>(params)...);
    auto & cb = res.first;
    auto & help = res.second;
    
    router.addRoute(path, filter, description, cb, help);
}

// Void return types don't need to convert their result
template<typename Return, typename Obj, typename... Args, typename Ptr,
         typename... Params>
void
addRouteSync(RestRequestRouter & router,
             PathSpec path, RequestFilter filter,
             const std::string & description,
             Return (Obj::* pmf) (Args...) const,
             Ptr ptr,
             Params&&... params)
{
    static_assert(sizeof...(Args) == sizeof...(Params),
                  "member function and parameter arity must match");

    typedef ML::TypeList<Args...> ArgsList;
    typedef ML::TypeList<Params...> ParamsList;
    typedef ML::PositionedDualTypeList<0, ArgsList, ParamsList> PositionedTypes;

    auto res = RestRequestBinder<typename PositionedTypes::List>
        ::bindSync(pmf, ptr, std::forward<Params>(params)...);
    auto & cb = res.first;
    auto & help = res.second;
        
    router.addRoute(path, filter, description, cb, help);
}

template<typename Return, typename Obj, typename... Args, typename Ptr,
         typename... Params>
void
addRouteAsync(RestRequestRouter & router,
              PathSpec path, RequestFilter filter,
              const std::string & description,
              Return (Obj::* pmf) (Args...),
              Ptr ptr,
              Params&&... params)
{
    static_assert(sizeof...(Args) == sizeof...(Params),
                  "member function and parameter arity must match");

    typedef ML::TypeList<Args...> ArgsList;
    typedef ML::TypeList<Params...> ParamsList;
    typedef ML::PositionedDualTypeList<0, ArgsList, ParamsList> PositionedTypes;

    auto res = RestRequestBinder<typename PositionedTypes::List>
        ::bindAsync(pmf, ptr, std::forward<Params>(params)...);
    auto & cb = res.first;
    auto & help = res.second;

    router.addRoute(path, filter, description, cb, help);
}

template<typename Return, typename Obj, typename... Args, typename Ptr,
         typename... Params>
void
addRouteAsync(RestRequestRouter & router,
              PathSpec path, RequestFilter filter,
              const std::string & description,
              Return (Obj::* pmf) (Args...) const,
              Ptr ptr,
              Params&&... params)
{
    static_assert(sizeof...(Args) == sizeof...(Params),
                  "member function and parameter arity must match");

    typedef ML::TypeList<Args...> ArgsList;
    typedef ML::TypeList<Params...> ParamsList;
    typedef ML::PositionedDualTypeList<0, ArgsList, ParamsList> PositionedTypes;

    auto res = RestRequestBinder<typename PositionedTypes::List>
        ::bindAsync(pmf, ptr, std::forward<Params>(params)...);
    auto & cb = res.first;
    auto & help = res.second;
        
    router.addRoute(path, filter, description, cb, help);
}

template<typename Return, typename... Args,
         typename... Params, typename ThenFn>
void
addRouteAsyncThen(RestRequestRouter & router,
                  PathSpec path, RequestFilter filter,
                  const std::string & description,
                  const std::string & resultDescription,
                  const ThenFn & then,
                  const ExcFn & exc,
                  const std::function<Return (Args...)> & fn,
                  Params&&... params)
{
    static_assert(sizeof...(Args) == sizeof...(Params),
                  "member function and parameter arity must match");

    typedef ML::TypeList<Args...> ArgsList;
    typedef ML::TypeList<Params...> ParamsList;
    typedef ML::PositionedDualTypeList<0, ArgsList, ParamsList> PositionedTypes;

    auto res = RestRequestBinder<typename PositionedTypes::List>
        ::bindAsyncThen(then, exc, fn, std::forward<Params>(params)...);
    auto & cb = res.first;
    auto & help = res.second;
    help["result"] = resultDescription;

    router.addRoute(path, filter, description, cb, help);
}

template<typename Return, typename... Args, typename... Params>
void
addRouteSyncJsonReturn(RestRequestRouter & router,
                       PathSpec path, RequestFilter filter,
                       const std::string & description,
                       const std::string & resultDescription,
                       const std::function<Return (Args...)> & fn,
                       Params&&... params)
{
    auto then = [] (const Return & ret,
                    const RestServiceEndpoint::ConnectionId & connection,
                    const RestRequest &,
                    const RestRequestParsingContext &)
        {
            static std::shared_ptr<ValueDescription> desc(getDefaultDescription((Return *)0));
            std::ostringstream out;
            StreamJsonPrintingContext context(out);
            desc->printJson(&ret, context);
            out << std::endl;
            connection.sendResponse(200, out.str(), "application/json");
            return RestRequestRouter::MR_YES;
        };

    addRouteAsyncThen(router, path, filter, description, resultDescription,
                      then, nullptr /* Exception handler */, fn,
                      std::forward<Params>(params)...);
}

template<typename Return, typename Obj, typename... Args, typename Ptr,
         typename... Params>
void
addRouteSyncJsonReturn(RestRequestRouter & router,
                       PathSpec path, RequestFilter filter,
                       const std::string & description,
                       const std::string & resultDescription,
                       Return (Obj::* pmf) (Args...),
                       Ptr ptr,
                       Params&&... params)
{
    return addRouteSyncJsonReturn(router, path, filter, description, resultDescription,
                                  partial_bind(pmf, ptr), std::forward<Params>(params)...);
}

template<typename Return, typename Obj, typename... Args, typename Ptr,
         typename... Params>
void
addRouteSyncJsonReturn(RestRequestRouter & router,
                       PathSpec path, RequestFilter filter,
                       const std::string & description,
                       const std::string & resultDescription,
                       Return (Obj::* pmf) (Args...) const,
                       Ptr ptr,
                       Params&&... params)
{
    return addRouteSyncJsonReturn(router, path, filter, description, resultDescription,
                                  partial_bind(pmf, ptr), std::forward<Params>(params)...);
}



} // namespace Datacratic

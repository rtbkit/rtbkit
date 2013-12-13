#How to Build a Service Using the Datacratic REST Api

We will try to document the use of the Datacratic REST Api. An example of a Rest service can be found at  soa/service/testing/rest_api_example.cc

## Define the Service

### Base Classes

The service class to implement usually derives from ServiceBase and an endpoint such as RestServiceEndpoint.
ServiceBase offers common features like Zookeeper registration and graphite logging.
The endpoint (RestServiceEndpoint) implements the message loop and the interaction with the routes.

### Members

The service should have a member that is a RestRequestRouter. This will manage the various routes used by the service.

Example:
```c++
#include "soa/service/rest_service_endpoint.h"
#include "soa/service/rest_request_router.h"

struct ExampleService(std::shared_ptr<ServiceProxies> proxies,

                    const std::string & serviceName)

{
RestRequestRouter router;
//...
}

ExampleService::ExampleService(std::shared_ptr<ServiceProxies> proxies, const std::string & serviceName)
   : ServiceBase(serviceName, proxies),
     RestServiceEndpoint(getZmqContext()) // getZmqContext is provided by the ServiceBase
{
//...
}
```

# Define the Routes

The route configuration has many options, but the two main ways of doing this are the route.addroute and the helper functions. The helper functions are defined in soa/service/rest_request.h

## Using the router class

The following example illustrates the implementation of a basic route:
```c++
RestRequestRouter::OnProcessRequest pingRoute
    = [] (const RestServiceEndpoint::ConnectionId & connection,
          const RestRequest & request,
          const RestRequestParsingContext & context) {
    connection.sendResponse(200, "1");
    return RestRequestRouter::MR_YES;
 };

router.addRoute( "/ping", "GET", "Ping the availability of the endpoint",
                pingRoute,
                Json::Value());

```
### Help Route

The Help route collects the information from the other routes automatically to automatically return info on the service.

Example:

```c++
router.addHelpRoute("/", "GET")
```

## Using the helper functions

Most of the helper functions are variations on the addRouteSyncReturn function. The helper functions are either sync or async. The sync helper functions return a result automatically while the async variations don't return anything. A result can be sent back later or manually through the connection. Sending a result through the connection while using a Sync version will throw an exception because it tries to send the result twice.

The addRouteSyncReturn has the following definition:
```c++
addRouteSyncReturn( RestRequestRouter & router,
                    PathSpec path,
                    RequestFilter filter,
                    const std::string & description,
                    const std::string & resultDescription, 
                    const TransformResult &transformResult, 
                    Return (Obj::* pmf) (Args...),
                    Ptr ptr,
                    Params&&... params)
```                    

Parameters of the route helper functions

* *router* is the basic router object

* *path* is the path of the resource

* *filter*  is the http command: (GET, PUT, POST, DELETE)

* *description* is a description to the route, used by the automatic help system.

* resultDescription describes the result that will be returned. It is used by the automatic help system

* transformResult is a function that can be used to modify the results of the pmf function. If no transformation is required the transformResult can simply return the return value of the pfm function.

* *pmf* is a pointer to the function that will be called

* *ptr* is the object that owns the function

* *params* is the list of parameters that are passed to the callback function.

Example:
```c++
  // using the helper functions
 addRouteSync(versionNode,
             "/events/record",
             {"PUT"},
             "store something",
             &ExampleService::recordEvent,
             this,
             JsonParam<string>("", "Payload for events"));
```             

## Tips and Tricks

### Regular expressions

It is possible to define routes based on regular expressions.
RX is a PatchSpec subclass that has this role. 
The most common usage of this is to define a sub route based on a regular expression:

```c++
auto & evtNode = versionNode.addSubRouter(Rx("/([^/]*)","/<path>"), "url path");
```
### Extract Parameters from the Route

The RequestParam template is used to define a parameter that is extracted from directly from the route. This can then be passed as a parameter to a route definition helper function.

### Access the Connection in the Router Function

To have access to the connection from the router function (pfm parameter of the helper functions) specify PassConnectionId() as one of the parameters.

For example:
```c++
   addRouteAsync(asyncNode,
                 "",
                 {"GET"},
                 "echo a part of the path",
                 &RestAPIExampleService::echoAsyncParam,
                 this,
                 RestParamDefault<std::string>("a_value", "a_value", "default_stuff"),
                 PassConnectionId()
                 );
```
Various route examples:
```c++
// add a submode

auto & versionNode = router.addSubRouter("/v1", "version 1 of API");`

// add a routes to the submode`

RestRequestRouter::OnProcessRequest serviceInfoRoute
    = [=] (const RestServiceEndpoint::ConnectionId & connection,
            const RestRequest & request,
            const RestRequestParsingContext & context) {
            Json::Value result;

    result["apiVersions"]["v1"] = "1.0.0";
    connection.sendResponse(200, result);
    return RestRequestRouter::MR_YES;
 };
 
versionNode.addRoute("/info","GET", "Return service information (version, etc)"serviceInfoRoute,
                      Json::Value());
```
# Service Configuration

The ServiceProxyArguments class contains the logic for the usual options of a service, it is initialized with a boost configuration_options object.
The ServiceProxyArguments is then used to initialize a ServiceProxy object.
It has a method makeServiceProxies which configures and returns the services.
The service proxy is then used to specify the service currently being defined.
The ServiceProxyArguments already defines many of the usual options of a service:

For example:

* service-name: the name used to find the service on zookeeper, it must be unique.

* bootstrab: points to a json file that can be used to specify the other ServiceProxyArguments.

* zookeeper-uri: the uri of the zookeeper instance that will be used to register this service.

* carbon-connection: the uri used to connect to a graphite instance for logging metrics

* installation: used by zookeeper to indicate a namespace to keep our stuff seperate from the clients if they use the same third party systems.

* location: indicates the datacenter where we will search for the zookeeper services.

The ServiceProxyArguments is used by the ServiceBase superclass of the service to specify the basic functionalities.

example:
```c+
// construct a BOOST::program_options.configuration_object from the 
// command line and/or config files.
// This is outside of the scope of the current document. Refer to the 
// BOOST documentation.

ServiceProxyArguments serviceArgs;

…

// pass create a ServiceProxies object
auto proxies = serviceArgs.makeServiceProxies();

// you now have the service object
ExampleService service(proxies, "ExampleService)
```

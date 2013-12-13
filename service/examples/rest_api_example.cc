/*
 * This file is used to illustrate various uses of the REST API
 * Start with
 * make run_rest_api_example ARGS=" --listen-port 8088"

*/

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/thread/thread.hpp>
#include <boost/make_shared.hpp>

#include "soa/types/value_description.h"
#include "soa/types/basic_value_descriptions.h"
#include "soa/service/rest_service_endpoint.h"
#include "soa/service/rest_request_router.h"
#include "soa/service/zmq_named_pub_sub.h"
#include "soa/service/s3.h"
#include "soa/service/sqs.h"
#include "soa/service/rest_request_binding.h"

#include "soa/service/service_utils.h"
#include "jml/utils/pair_utils.h"
#include "jml/arch/timers.h"
#include "jml/arch/futex.h"
#include "soa/service/s3.h"
#include "soa/jsoncpp/json.h"


namespace Datacratic {

using Datacratic::jsonDecode;
using Datacratic::jsonEncode;
using namespace Datacratic;
std::vector<std::string>  process_url(std::string url);
std::vector<std::string>  process_url(std::vector<std::string> url_parts);



#include <iostream>


/*****************************************************************************/
/* PIXEL IMPORT SERVICE                                                     */
/*****************************************************************************/

struct RestAPIExampleService: public ServiceBase, public RestServiceEndpoint {

	RestAPIExampleService(std::shared_ptr<ServiceProxies> proxies,
                        const std::string & serviceName,
                        const std::string & outputPath)
	: ServiceBase(serviceName, proxies),
      RestServiceEndpoint(getZmqContext())
	{}


    RestRequestRouter router;
    void echoParamFn(const std::string & path_param,
    		         const std::string & stuff)
       {
       	std::cerr << path_param << "  " << stuff <<  std::endl;
       };
    void echoSyncParam(const std::string & a_value,
        		         const std::string & stuff)
           {
           	std::cerr << a_value << "  " << stuff <<  std::endl;
           };

    void echoAsyncParam(const std::string & a_value,
    		           const RestServiceEndpoint::ConnectionId & connection)
               {
               	std::cerr << a_value << std::endl;
               	connection.sendResponse(200, a_value, "text/plain");
               };

    Json::Value returnValueFn(const std::string & a_value)
    {
      Json::Value result;
      result["the_result"] = a_value;
      return result; 
    }

    void init() {
    	registerServiceProvider(serviceName(), { "RestApiExample" });
    	RestServiceEndpoint::init(getServices()->config, serviceName());
    	router.description = "Datacratic REST API Example";

    	onHandleRequest = router.requestHandler();

    	// defines the help route
    	// The return value of the help route is automatically built using the information from the other routes of the system
    	//
    	router.addHelpRoute("/", "GET");


    	// Illustrates using the Router directly instead of the helper functions to add a route
    	//
        RestRequestRouter::OnProcessRequest pingRoute
            = [] (const RestServiceEndpoint::ConnectionId & connection,
                  const RestRequest & request,
                  const RestRequestParsingContext & context) {
            connection.sendResponse(200, "1");
            return RestRequestRouter::MR_YES;
        };

        router.addRoute("/ping", "GET", "Ping the availability of the endpoint",
                        pingRoute,
                        Json::Value());

        // illustrates using sub routes
        //
        auto & paramsNode = router.addSubRouter("/params", "param example");
        auto & rtnValNode = router.addSubRouter("/rtn", "param example");
        auto & asyncNode = router.addSubRouter("/async", "async call example");
        auto & versionNode = router.addSubRouter("/v1", "version 1 of API");
        auto & urlParamNode = versionNode.addSubRouter(Rx("/([^/]*)","/<path>"), "url path");

        // Illustrates extracting a parameter from the url path
        // example:
        //		http://localhost:8088/v1/params_value
        //			will call echoParaFn with path_param= param_value
        //
        RequestParam<std::string> pathParam(2, "<echo_val>", "echo value");

        addRouteSync(urlParamNode,
                       "",
                       {"GET"},
                       "echo a part of the path",
                       &RestAPIExampleService::echoParamFn,
                       this,
                       pathParam,
                       RestParamDefault<std::string>("stuff", "stuff", "stuff")
                       );

        //Illustrates getting the parameters from the url parameters
        // example:
        //         http://localhost:8088/params?a_value=toto&stuff=bof
        //				will call the echoSyncParam with a_value = toto and stuff = bof
        //         http://localhost:8088/params?a_value=toto
        //              will call the echoSyncParam with a_value = toto and stuff = default_stuff
        addRouteSync(paramsNode,
                           "",
                           {"GET"},
                           "echo a part of the path",
                           &RestAPIExampleService::echoSyncParam,
                           this,
                           RestParam<std::string>("a_value", "a value"),
                           RestParamDefault<std::string>("stuff", "stuff", "default_stuff")
                           );


        // Illustrates using the async helper functions and also the use of the ConnectionId from within the callback.
        // The return value must be managed manually. In this case
        // we use the connection parameter to send the reply.
        // example:
        //         http://localhost:8088/params?a_value=toto
        //              will call the echoAsyncParam with a_value = toto
        addRouteAsync(asyncNode,
                      "",
                      {"GET"},
                      "demonstrate returning result through a connection",
                      &RestAPIExampleService::echoAsyncParam,
                      this,
                      RestParamDefault<std::string>("a_value", "a_value", "default_stuff"),
                      PassConnectionId()
                      );

        // Illustrates using the sync helper functions that returns a value.
        // The return value will be intercepted by the transformResultParameter that can modify it
        // before passing it along.
        // example:
        //         http://localhost:8088/rtns?a_value=toto
        //              will call the returnValueFn with a_value = toto
        // And return the jason string:
        // {
        //     "the_result" : "toto"
        // }
	    addRouteSyncReturn(rtnValNode,
			   "",
               {"GET"},
			   "demonstrate a sync route with return value",
			   "the return value",
			   [] (Json::Value v) { return v;},
			   &RestAPIExampleService::returnValueFn,
			   this,
			   RestParamDefault<std::string>("a_value", "a_value", "default_stuff")
			   );

    }

    std::string bindTcp(const PortRange & portRange = PortRange(),
                        const std::string & host = "localhost")
    {
    	 return httpEndpoint.bindTcp(portRange, host);
    }

    void start()
    {
    	httpEndpoint.spinup(8,false);
    }

    void shutdown()
    {}


};

} // namespace Datacratic





//*******************************************************************************************




using namespace std;
using namespace ML;
using namespace Datacratic;
// can be tested with something like:
// curl  "http://localhost:8088/v1/event-sku-123?count=339&user_id=3"
// ab  -k -v 4 -n 20000 -c 20 "http://localhost:8088/v1/event-sku-123?count=339&user_id=3"
int main(int argc, char ** argv)
{
    using namespace boost::program_options;

    ServiceProxyArguments serviceArgs;

    options_description configuration_options("Configuration options");

    std::string outputPath;
    int listenPort = 4888;
    std::string listenHost = "*";
    std::string rotationInterval = "5m";
    std::string serviceName = "PixelServer";
    string s3KeyId;
    string s3Key;
    string sqsQueueUri;

    configuration_options.add_options()
        ("listen-port,p", value(&listenPort),
         "listen on given port (4888)")
        ("listen-host,h", value(&listenHost),
         "listen to given host (*)")
        ("name,n", value(&serviceName),
         "name of service and namespace for files written (Rest_API_Example))");


    options_description all_opt;
    all_opt
        .add(serviceArgs.makeProgramOptions())
        .add(configuration_options);
    all_opt.add_options()
        ("help,h", "print this message");

    variables_map vm;
    store(command_line_parser(argc, argv)
          .options(all_opt)
          //.positional(p)
          .run(),
          vm);
    notify(vm);

    if (vm.count("help")) {
        cerr << all_opt << endl;
        exit(1);
    }

    auto proxies = serviceArgs.makeServiceProxies();

    RestAPIExampleService service(proxies, "pixelServerService", outputPath);
    service.init();
    auto addr = service.bindTcp(listenPort, listenHost);
    service.start();

    cerr << "import service is started on " << addr << endl;

    proxies->config->dump(cerr);

    for (;;) {
        ML::sleep(100000);
    }
}

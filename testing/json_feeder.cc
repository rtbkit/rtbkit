/* json_feeder.cc
   Wolfgang Sourdeau, February 2013
   Copyright (c) 2013 Datacratic.  All rights reserved.

   A utility that feeds a stream of JSON samples to an HTTP server.
 */


#include <string>

#include <boost/program_options/cmdline.hpp>
#include <boost/program_options/options_description.hpp>
#include <boost/program_options/positional_options.hpp>
#include <boost/program_options/parsers.hpp>
#include <boost/program_options/variables_map.hpp>

#include <curlpp/Easy.hpp>
#include <curlpp/Info.hpp>
#include <curlpp/Infos.hpp>
#include <curlpp/Options.hpp>

#include "jml/utils/filter_streams.h"


using namespace std;
using namespace boost::program_options;
using namespace curlpp;


struct JsonFeeder {
    JsonFeeder(string uri, string filename,
               bool printRequests, bool printResponses,
               int maxSamples = 1000)
        : serverUri(uri), jsonStream(filename), printRequests(printRequests),
          printResponses(printResponses), maxSamples(maxSamples)
        {}

    void perform()
    {
        int sampleNum;

        for (sampleNum = 0; jsonStream && sampleNum < maxSamples;
             sampleNum++) {
            string current;
            getline(jsonStream, current);

            if (current == "") {
                cerr << "current == '' -> leaving after "
                     << sampleNum << " samples" << endl;
                break;
            }

            Easy client;

            /* perform request */
            client.setOpt(options::Url(serverUri));
        
            client.setOpt(options::Post(true));
            client.setOpt(options::PostFields(current));

            list<string> headers;
            headers.push_back("Content-Type: application/json");
            headers.push_back("Expect:"); /* avoid dual-phase post */
            client.setOpt(options::HttpHeader(headers));

            /* separate response headers from body and store response body in "body" */
            stringstream body;
            client.setOpt(options::TcpNoDelay(true));
            client.setOpt(options::Header(false));
            client.setOpt(options::WriteStream(&body));
            client.perform();

            if (sampleNum > 0 && (printRequests || printResponses)) {
                cerr << "----------------------------" << endl
                     << "sample: " << sampleNum << endl;
            }

            if (printRequests) {
                cerr << "rq body: " << current << endl;
            }

            if (printResponses) {
                int code = infos::ResponseCode::get(client);
                cerr << "resp. code: " << code << endl
                     << "resp. body: " << body.str() << endl;
            }
        }

        cerr << "posted " << sampleNum << " samples" << endl;
    }

    string serverUri;
    ML::filter_istream jsonStream;
    bool printRequests;
    bool printResponses;
    int maxSamples;
};


int main(int argc, char *argv[])
{
    string serverUri;
    string filename;
    bool printRequests(false), printResponses(false);

    {
        using namespace boost::program_options;

        options_description configuration_options("Configuration options");

        configuration_options.add_options()
            ("server-uri,s", value(&serverUri),
             "URI of server to feed")
            ("filename,f", value(&filename),
             "filename")
            ("printrequests", value(&printRequests)->zero_tokens(),
             "print requests on console")
            ("printresponses", value(&printResponses)->zero_tokens(),
             "print responses on console");
 
        options_description all_opt;
        all_opt.add(configuration_options);
        all_opt.add_options()
            ("help,h", "print this message");

        variables_map vm;
        store(command_line_parser(argc, argv)
              .options(all_opt)
              .run(),
              vm);
        notify(vm);

        if (vm.count("help")) {
            cerr << all_opt << endl;
            exit(1);
        }

        if (serverUri.empty()) {
            cerr << "'server-uri' parameter is required" << endl;
            exit(1);
        }

        if (filename.empty()) {
            cerr << "'filename' parameter is required" << endl;
            exit(1);
        }
    }

    JsonFeeder feeder(serverUri, filename, printRequests, printResponses);
    feeder.perform();

    return 0;
}

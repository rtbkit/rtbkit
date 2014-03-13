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

#include "jml/arch/exception.h"
#include "jml/arch/timers.h"
#include "jml/utils/filter_streams.h"


using namespace std;
using namespace boost::program_options;
using namespace curlpp;


int deltaDelayMs(const struct timeval & oldTime,
                 const struct timeval & newTime)
{
    int64_t deltaMuSecs;

    if (oldTime.tv_usec > newTime.tv_usec) {
        deltaMuSecs = ((1000000 + newTime.tv_usec - oldTime.tv_usec)
                       + (newTime.tv_sec - oldTime.tv_sec - 1) * 1000000);
    }
    else {
        deltaMuSecs = ((newTime.tv_usec - oldTime.tv_usec)
                       + (newTime.tv_sec - oldTime.tv_sec) * 1000000);
    }

    if (deltaMuSecs < 0)
        throw ML::Exception("the future must not occur before the past");

    return deltaMuSecs / 1000;
}

struct JsonFeeder {
    JsonFeeder(string uri, string filename,
               int nSamples, int delayMs,
               bool printRequests, bool printResponses)
        : serverUri(uri), filename(filename), jsonStream(filename),
          nSamples(nSamples), delayMs(delayMs),
          printRequests(printRequests), printResponses(printResponses)
        {}

    void perform()
    {
        int sampleNum;
        struct timeval lastRequest;
        Easy client;

        for (sampleNum = 0; jsonStream && sampleNum < nSamples;
             sampleNum++) {
            string current;
            struct timeval thisRequest, thisResponse;
            getline(jsonStream, current);

            if (current == "") {
                /* start over from the beginning of the file */
                jsonStream = ML::filter_istream(filename);
                getline(jsonStream, current);
            }

            if (delayMs > 0) {
                ::gettimeofday(&thisRequest, NULL);
                if (sampleNum > 0) {
                    int deltaRqMs = deltaDelayMs(lastRequest, thisRequest);
                    // printf("deltaRqMs: %d\n", deltaRqMs);
                    if (deltaRqMs < delayMs) {
                        float sleepTime = (float) (delayMs - deltaRqMs) / 1000;
                        // cerr << "sleeping for " << sleepTime << " secs\n";
                        ML::sleep(sleepTime);
                        ::gettimeofday(&thisRequest, NULL);
                    }
                }
                lastRequest = thisRequest;
            }

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

            if (delayMs > 0) {
                ::gettimeofday(&thisResponse, NULL);
                int deltaRqMs = deltaDelayMs(thisRequest, thisResponse);
                cerr << "request took " << deltaRqMs << " millisecs\n";
            }

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
    string filename;
    ML::filter_istream jsonStream;
    int nSamples;
    int delayMs;
    bool printRequests;
    bool printResponses;
};


int main(int argc, char *argv[])
{
    string serverUri;
    string filename;
    int delay(0);
    int nSamples(1000);
    bool printRequests(false), printResponses(false);

    {
        using namespace boost::program_options;

        options_description configuration_options("Configuration options");

        configuration_options.add_options()
            ("server-uri,s", value(&serverUri),
             "URI of server to feed")
            ("filename,f", value(&filename),
             "filename")
            ("n-samples,n", value(&nSamples),
             "number of requests to perform")
            ("rq-delay,d", value(&delay),
             "minimal delay (in ms, between requests)")
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

    JsonFeeder feeder(serverUri, filename, nSamples, delay,
                      printRequests, printResponses);
    feeder.perform();

    return 0;
}

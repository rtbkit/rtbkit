/* http_rest_proxy.cc
   Jeremy Barnes, 10 April 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   REST proxy class for http.
*/

#include "http_rest_proxy.h"

#include <curlpp/cURLpp.hpp>
#include <curlpp/Easy.hpp>
#include <curlpp/Options.hpp>
#include <curlpp/Info.hpp>
#include <curlpp/Infos.hpp>



using namespace std;
using namespace ML;


namespace Datacratic {


/*****************************************************************************/
/* HTTP REST PROXY                                                           */
/*****************************************************************************/

HttpRestProxy::
~HttpRestProxy()
{
    for (auto c: inactive)
        delete c;
}

HttpRestProxy::Response
HttpRestProxy::
perform(const std::string & verb,
        const std::string & resource,
        const Content & content,
        const RestParams & queryParams,
        const RestParams & headers,
        int timeout) const
{
    string responseHeaders;
    string body;
    string uri;

    try {
        responseHeaders.clear();
        body.clear();

        Connection connection = getConnection();

        curlpp::Easy & myRequest = *connection;

        using namespace curlpp::options;
        using namespace curlpp::infos;
            
        list<string> curlHeaders;
        for (unsigned i = 0;  i < headers.size();  ++i) {
            curlHeaders.push_back(headers[i].first + ": "
                                  + headers[i].second);
        }

        uri = serviceUri + resource + queryParams.uriEscaped();

        //cerr << "uri = " << uri << endl;
        
        myRequest.setOpt<CustomRequest>(verb);

        myRequest.setOpt<curlpp::options::Url>(uri);

        if (debug)
            myRequest.setOpt<Verbose>(true);

        myRequest.setOpt<ErrorBuffer>((char *)0);
        if (timeout != -1)
            myRequest.setOpt<Timeout>(timeout);
        else myRequest.setOpt<Timeout>(0);
        myRequest.setOpt<NoSignal>(1);

        if (noSSLChecks) {
            myRequest.setOpt<SslVerifyHost>(false);
            myRequest.setOpt<SslVerifyPeer>(false);
        }

        auto onData = [&] (char * data, size_t ofs1, size_t ofs2) -> size_t
            {
                //cerr << "called onData for " << ofs1 << " " << ofs2 << endl;
                return 0;
            };

        auto onWriteData = [&] (char * data, size_t ofs1, size_t ofs2) -> size_t
            {
                body.append(data, ofs1 * ofs2);
                return ofs1 * ofs2;
                //cerr << "called onWrite for " << ofs1 << " " << ofs2 << endl;
                return 0;
            };

        auto onProgress = [&] (double p1, double p2, double p3, double p4) -> int
            {
                cerr << "progress " << p1 << " " << p2 << " " << p3 << " "
                << p4 << endl;
                return 0;
            };

        bool afterContinue = false;

        //cerr << endl << endl << "*******************" << endl;

        auto onHeader = [&] (char * data, size_t ofs1, size_t ofs2) -> size_t
            {
                string headerLine(data, ofs1 * ofs2);

                //cerr << "got header " << headerLine << endl;

                if (headerLine.find("HTTP/1.1 100 Continue") == 0) {
                    afterContinue = true;
                }
                else if (afterContinue) {
                    if (headerLine == "\r\n")
                        afterContinue = false;
                }
                else {
                    responseHeaders.append(headerLine);
                    //cerr << "got header data " << headerLine << endl;
                }
                return ofs1 * ofs2;
            };

        myRequest.setOpt<BoostHeaderFunction>(onHeader);
        myRequest.setOpt<BoostWriteFunction>(onWriteData);
        myRequest.setOpt<BoostProgressFunction>(onProgress);
        for (auto & cookie: cookies)
            myRequest.setOpt<curlpp::options::CookieList>(cookie);

        //myRequest.setOpt<Header>(true);

        if (content.data) {
            string s(content.data, content.size);
            myRequest.setOpt<PostFields>(s);
            myRequest.setOpt<PostFieldSize>(content.size);
            curlHeaders.push_back(ML::format("Content-Length: %lld",
                                             content.size));
            //curlHeaders.push_back("Transfer-Encoding:");
            curlHeaders.push_back("Content-Type: " + content.contentType);
        }
        else {
            myRequest.setOpt<PostFieldSize>(-1);
            myRequest.setOpt<PostFields>("");
        }

        myRequest.setOpt<curlpp::options::HttpHeader>(curlHeaders);

        myRequest.perform();

        Response response;
        response.body_ = body;

        curlpp::InfoGetter::get(myRequest, CURLINFO_RESPONSE_CODE,
                                response.code_);

        double bytesUploaded;
    
        curlpp::InfoGetter::get(myRequest, CURLINFO_SIZE_UPLOAD,
                                bytesUploaded);

        //cerr << "uploaded " << bytesUploaded << " bytes" << endl;

        response.header_.parse(responseHeaders);

        return response;
    } catch (const curlpp::LibcurlRuntimeError & exc) {
        cerr << "libCurl returned an error with code " << exc.whatCode()
             << endl;
        cerr << "error is " << curl_easy_strerror(exc.whatCode())
             << endl;
        cerr << "verb is " << verb << endl;
        cerr << "uri is " << uri << endl;
        //cerr << "query params are " << queryParams << endl;
        cerr << "headers are " << responseHeaders << endl;
        cerr << "body contains " << body.size() << " bytes" << endl;
        throw;
    }
}

HttpRestProxy::Connection::
~Connection()
{
    if (!conn)
        return;
    proxy->doneConnection(conn);
}

HttpRestProxy::Connection
HttpRestProxy::
getConnection() const
{
    std::unique_lock<std::mutex> guard(lock);

    if (inactive.empty()) {
        return Connection(new curlpp::Easy, const_cast<HttpRestProxy *>(this));
    }
    else {
        auto res = inactive.back();
        inactive.pop_back();
        return Connection(res, const_cast<HttpRestProxy *>(this));
    }
}

void
HttpRestProxy::
doneConnection(curlpp::Easy * conn)
{
    std::unique_lock<std::mutex> guard(lock);
    inactive.push_back(conn);
}

} // namespace Datacratic

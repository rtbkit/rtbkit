/* http_rest_proxy.cc
   Jeremy Barnes, 10 April 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   REST proxy class for http.
*/

#include <curl/curl.h>
#include "jml/arch/threads.h"
#include "jml/arch/timers.h"
#include "soa/types/basic_value_descriptions.h"
#include "openssl_threading.h"

#include "http_rest_proxy.h"


using namespace std;
using namespace ML;
using namespace Datacratic;


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
        double timeout,
        bool exceptions,
        OnData onData,
        OnHeader onHeader) const
{
    string responseHeaders;
    string body;
    string uri;
    RestParams curlHeaders = headers;
    
    try {
        responseHeaders.clear();
        body.clear();

        Connection connection = getConnection();

        CurlWrapper::Easy & myRequest = *connection;

        uri = serviceUri + resource + queryParams.uriEscaped();

        myRequest.add_option(CURLOPT_CUSTOMREQUEST, verb);
        
        myRequest.add_option(CURLOPT_URL, uri);
        
        if (debug)
            myRequest.add_option(CURLOPT_VERBOSE, 1L);

        if (timeout != -1)
            myRequest.add_option(CURLOPT_TIMEOUT, timeout);
        else myRequest.add_option(CURLOPT_TIMEOUT, 0L);
        myRequest.add_option(CURLOPT_NOSIGNAL, 1L);

        if (noSSLChecks) {
            myRequest.add_option(CURLOPT_SSL_VERIFYHOST, 0L);
            myRequest.add_option(CURLOPT_SSL_VERIFYPEER, 0L);
        }

        CurlWrapper::Easy::CurlCallback onWriteData
            = [&] (char * data, size_t ofs1, size_t ofs2) -> size_t
            {
                if (debug)
                    cerr << "got data " << string(data, data + ofs1 * ofs2) << endl;

                if (onData) {
                    if (!onData(string(data, data + ofs1 * ofs2)))
                        return ofs1 * ofs2 + 1; //indicate an error
                    return ofs1 * ofs2;
                }

                body.append(data, ofs1 * ofs2);
                return ofs1 * ofs2;
            };

        bool afterContinue = false;

        Response response;
        bool headerParsed = false;

        CurlWrapper::Easy::CurlCallback onHeaderLine
            = [&] (char * data, size_t ofs1, size_t ofs2) -> size_t
            {
                ExcAssert(!headerParsed);

                string headerLine(data, ofs1 * ofs2);

                if (debug)
                    cerr << "got header " << headerLine << endl;

                if (headerLine.find("HTTP/1.1 100 Continue") == 0) {
                    afterContinue = true;
                }
                else if (afterContinue) {
                    if (headerLine == "\r\n")
                        afterContinue = false;
                }
                else {
                    responseHeaders.append(headerLine);
                    if (headerLine == "\r\n") {
                        response.header_.parse(responseHeaders);
                        headerParsed = true;

                        if (onHeader)
                            if (!onHeader(response.header_))
                                return ofs1 * ofs2 + 1;  // indicate an error
                    }
                }
                return ofs1 * ofs2;
            };

        myRequest.add_callback_option(CURLOPT_HEADERFUNCTION, CURLOPT_HEADERDATA, onHeaderLine);
        myRequest.add_callback_option(CURLOPT_WRITEFUNCTION, CURLOPT_WRITEDATA, onWriteData);

        for (auto & cookie: cookies)
            myRequest.add_option(CURLOPT_COOKIELIST, cookie);

        if (content.data) {
            myRequest.add_option(CURLOPT_POSTFIELDSIZE, content.size);
            myRequest.add_data_option(CURLOPT_POSTFIELDS, content.data);
            curlHeaders.emplace_back(make_pair("Content-Length", ML::format("%lld", content.size)));
            curlHeaders.emplace_back(make_pair("Content-Type", content.contentType));
        }
        else {
            myRequest.add_option(CURLOPT_POSTFIELDSIZE, 0L);
            myRequest.add_option(CURLOPT_POSTFIELDS, "");
        }

        myRequest.add_header_option(curlHeaders);

        CURLcode code = myRequest.perform();
        response.body_ = body;
        if (code != CURLE_OK) {
            if (!exceptions) {
                response.errorCode_ = code;
                response.errorMessage_ = curl_easy_strerror(code);
                return response;
            }
            else {
                throw CurlWrapper::RuntimeError("performing HTTP request",
                                                code);
            }
        }

        myRequest.get_info(CURLINFO_RESPONSE_CODE, response.code_);

        double bytesUploaded;
    
        myRequest.get_info(CURLINFO_SIZE_UPLOAD, bytesUploaded);

        //cerr << "uploaded " << bytesUploaded << " bytes" << endl;

        return response;
    } catch (const CurlWrapper::RuntimeError & exc) {
        if (exc.whatCode() == CURLE_OPERATION_TIMEDOUT)
            throw;
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
        return Connection(new CurlWrapper::Easy, const_cast<HttpRestProxy *>(this));
    }
    else {
        auto res = inactive.back();
        inactive.pop_back();
        return Connection(res, const_cast<HttpRestProxy *>(this));
    }
}

void
HttpRestProxy::
doneConnection(CurlWrapper::Easy * conn)
{
    std::unique_lock<std::mutex> guard(lock);
    conn->reset();
    inactive.push_back(conn);
}


/****************************************************************************/
/* JSON REST PROXY                                                          */
/****************************************************************************/

JsonRestProxy::
JsonRestProxy(const string & url)
    : HttpRestProxy(url), maxRetries(10), maxBackoffTime(900)
{
    if (url.find("https://") == 0) {
        cerr << "warning: no validation will be performed on the SSL cert.\n";
        noSSLChecks = true;
    }
}

HttpRestProxy::Response
JsonRestProxy::
performWithBackoff(const string & method, const string & resource,
                   const string & body)
    const
{
    HttpRestProxy::Response response;

    JML_TRACE_EXCEPTIONS(false);

    Content content(body, "application/json");
    RestParams headers;

    // cerr << "posting data to " + resource + "\n";
    if (authToken.size() > 0) {
        headers.emplace_back(make_pair("Cookie", "token=\"" + authToken + "\""));
    }

    pid_t tid = gettid();
    for (int retries = 0;
         (maxRetries == -1) || (retries < maxRetries);
         retries++) {
        response = this->perform(method, resource, content, RestParams(),
                                 headers);
        int code = response.code();
        if (code < 400) {
            break;
        }

        /* error handling */
        if (retries == 0) {
            string respBody = response.body();
            ::fprintf(stderr,
                      "[%d] %s %s returned response code %d"
                      " (attempt %d):\n"
                      "request body (%lu) = '%s'\n"
                      "response body (%lu): '%s'\n",
                      tid, method.c_str(), resource.c_str(), code, retries,
                      body.size(), body.c_str(),
                      respBody.size(), respBody.c_str());
        }
        if (code < 500) {
            break;
        }

        /* recoverable errors */
        if (maxRetries == -1 || retries < maxRetries) {
            sleepAfterRetry(retries, maxBackoffTime);
            ::fprintf(stderr, "[%d] retrying %s %s after error (%d/%d)\n",
                      tid, method.c_str(), resource.c_str(), retries + 1, maxRetries);
        }
        else {
            throw ML::Exception("[%d] too many retries\n", tid);
        }
    }

    return response;
}

bool
JsonRestProxy::
authenticate(const JsonAuthenticationRequest & creds)
{
    bool result;

    try {
        auto authResponse = postTyped<JsonAuthenticationResponse>("/authenticate", creds, 200);
        authToken = authResponse.token;
        result = true;
    }
    catch (const ML::Exception & exc) {
        result = false;
    }

    return result;
}

void
JsonRestProxy::
sleepAfterRetry(int retryNbr, int maxBaseTime)
{
    static const double sleepUnit(0.2);

    int maxSlot = (1 << retryNbr) - 1;
    double baseTime = maxSlot * sleepUnit;
    if (baseTime > maxBaseTime) {
        baseTime = maxBaseTime;
    }
    int rnd = random();
    double timeToSleep = ((double) rnd / RAND_MAX) * baseTime;
    // cerr << "sleeping " << timeToSleep << endl;

    ML::sleep(timeToSleep);
}


/****************************************************************************/
/* JSON AUTHENTICATION REQUEST                                              */
/****************************************************************************/

JsonAuthenticationRequestDescription::
JsonAuthenticationRequestDescription()
{
    addField("email", &JsonAuthenticationRequest::email, "");
    addField("password", &JsonAuthenticationRequest::password, "");
}


/****************************************************************************/
/* JSON AUTHENTICATION RESPONSE                                             */
/****************************************************************************/

JsonAuthenticationResponseDescription::
JsonAuthenticationResponseDescription()
{
    addField("token", &JsonAuthenticationResponse::token, "");
}

} // namespace Datacratic

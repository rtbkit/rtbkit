/* http_client.cc
   Wolfgang Sourdeau, January 2014
   Copyright (c) 2014 Datacratic.  All rights reserved.
*/

#include "http_client.h"
#include "http_client_v1.h"
#include "http_client_v2.h"

using namespace std;
using namespace Datacratic;


namespace {

int httpClientImplVersion;

struct AtInit {
    AtInit()
    {
        httpClientImplVersion = 1;

        char * value = ::getenv("HTTP_CLIENT_IMPL");
        if (!value) {
            return;
        }

        if (::strcmp(value, "1") == 0) {
            httpClientImplVersion = 1;
        }
        else if (::strcmp(value, "2") == 0) {
            httpClientImplVersion = 2;
        }
        else {
            ::fprintf(stderr, "HttpClient: no handling for HttpClientImpl"
                      " version '%s', using default\n", value);
        }
    }
} atInit;

} // file scope


/****************************************************************************/
/* HTTP CLIENT ERROR                                                        */
/****************************************************************************/

std::ostream &
Datacratic::
operator << (std::ostream & stream, HttpClientError error)
{
    return stream << HttpClientCallbacks::errorMessage(error);
}


/****************************************************************************/
/* HTTP CLIENT                                                              */
/****************************************************************************/

void
HttpClient::
setHttpClientImplVersion(int version)
{
    if (version < 1 || version > 2) {
        throw ML::Exception("invalid value for 'version': "
                            + to_string(version));
    }
    httpClientImplVersion = version;
}

HttpClient::
HttpClient(const string & baseUrl, int numParallel, int queueSize,
           int implVersion)
{
    bool isHttps(baseUrl.compare(0, 8, "https://") == 0);

    if (baseUrl.compare(0, 7, "http://") != 0 && !isHttps) {
        throw ML::Exception("'url' has an invalid value: " + baseUrl);
    }
    if (numParallel < 1) {
        throw ML::Exception("'numParallel' must at least be equal to 1");
    }

    if (implVersion == 0) {
        implVersion = httpClientImplVersion;
    }

    if (implVersion == 1) {
        impl.reset(new HttpClientV1(baseUrl, numParallel, queueSize));
    }
    else if (implVersion == 2) {
        if (isHttps) {
            impl.reset(new HttpClientV1(baseUrl, numParallel, queueSize));
        }
        else {
            impl.reset(new HttpClientV2(baseUrl, numParallel, queueSize));
        }
    }
    else {
        throw ML::Exception("invalid httpclient impl version");
    }

    /* centralize the default values */
    enableDebug(false);
    enableSSLChecks(true);
    enableTcpNoDelay(false);
    enablePipelining(false);
}

void
HttpClient::
sendExpect100Continue(bool value)
{
    if (value) {
        throw ML::Exception("HttpClient has no support for"
                            " 'Expect: 100 Continue' requests");
    }
}


/****************************************************************************/
/* HTTP CLIENT CALLBACKS                                                    */
/****************************************************************************/

const string &
HttpClientCallbacks::
errorMessage(HttpClientError errorCode)
{
    static const string none = "No error";
    static const string unknown = "Unknown error";
    static const string hostNotFound = "Host not found";
    static const string couldNotConnect = "Could not connect";
    static const string timeout = "Request timed out";
    static const string sendError = "Failure sending network data";
    static const string recvError = "Failure receiving network data";

    switch (errorCode) {
    case HttpClientError::None:
        return none;
    case HttpClientError::Unknown:
        return unknown;
    case HttpClientError::Timeout:
        return timeout;
    case HttpClientError::HostNotFound:
        return hostNotFound;
    case HttpClientError::CouldNotConnect:
        return couldNotConnect;
    case HttpClientError::SendError:
        return sendError;
    case HttpClientError::RecvError:
        return recvError;
    default:
        throw ML::Exception("invalid error code");
    };
}

void
HttpClientCallbacks::
onResponseStart(const HttpRequest & rq,
                const string & httpVersion, int code)
{
    if (onResponseStart_)
        onResponseStart_(rq, httpVersion, code);
}

void
HttpClientCallbacks::
onHeader(const HttpRequest & rq, const char * data, size_t size)
{
    if (onHeader_)
        onHeader_(rq, data, size);
}

void
HttpClientCallbacks::
onData(const HttpRequest & rq, const char * data, size_t size)
{
    if (onData_)
        onData_(rq, data, size);
}

void
HttpClientCallbacks::
onDone(const HttpRequest & rq, HttpClientError errorCode)
{
    if (onDone_)
        onDone_(rq, errorCode);
}


/****************************************************************************/
/* HTTP CLIENT SIMPLE CALLBACKS                                             */
/****************************************************************************/

HttpClientSimpleCallbacks::
HttpClientSimpleCallbacks(const OnResponse & onResponse)
    : onResponse_(onResponse)
{
}

void
HttpClientSimpleCallbacks::
onResponseStart(const HttpRequest & rq,
                const string & httpVersion, int code)
{
    statusCode_ = code;
}

void
HttpClientSimpleCallbacks::
onHeader(const HttpRequest & rq, const char * data, size_t size)
{
    headers_.append(data, size);
}

void
HttpClientSimpleCallbacks::
onData(const HttpRequest & rq, const char * data, size_t size)
{
    body_.append(data, size);
}

void
HttpClientSimpleCallbacks::
onDone(const HttpRequest & rq, HttpClientError error)
{
    onResponse(rq, error, statusCode_, move(headers_), move(body_));
    statusCode_ = 0;
    headers_ = "";
    body_ = "";
}

void
HttpClientSimpleCallbacks::
onResponse(const HttpRequest & rq,
           HttpClientError error, int status,
           string && headers, string && body)
{
    if (onResponse_) {
        onResponse_(rq, error, status, move(headers), move(body));
    }
}

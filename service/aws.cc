/* aws.cc
   Jeremy Barnes, 8 August 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

*/

#include "aws.h"
#include "jml/arch/format.h"
#include "jml/utils/string_functions.h"
#include <iostream>

#include "xml_helpers.h"
#include <boost/algorithm/string.hpp>

#define CRYPTOPP_ENABLE_NAMESPACE_WEAK 1
#include "crypto++/sha.h"
#include "crypto++/md5.h"
#include "crypto++/hmac.h"
#include "crypto++/base64.h"
#include "crypto++/hex.h"


using namespace std;
using namespace ML;


namespace {

std::mutex awsCredentialsLock;
std::map<string, std::string> awsCredentials;

} // file scope


namespace Datacratic {

template<class Hash>
std::string
AwsApi::
hmacDigest(const std::string & stringToSign,
           const std::string & accessKey)
{
    size_t digestLen = Hash::DIGESTSIZE;
    byte digest[digestLen];
    CryptoPP::HMAC<Hash> hmac((byte *)accessKey.c_str(), accessKey.length());
    hmac.CalculateDigest(digest,
                         (byte *)stringToSign.c_str(),
                         stringToSign.length());

    return std::string((const char *)digest,
                       digestLen);
}

std::string
AwsApi::
hmacSha1Digest(const std::string & stringToSign,
               const std::string & accessKey)
{
    return hmacDigest<CryptoPP::SHA1>(stringToSign, accessKey);
}

std::string
AwsApi::
hmacSha256Digest(const std::string & stringToSign,
                 const std::string & accessKey)
{
    return hmacDigest<CryptoPP::SHA256>(stringToSign, accessKey);
}

std::string
AwsApi::
sha256Digest(const std::string & stringToSign)
{
    typedef CryptoPP::SHA256 Hash;
    size_t digestLen = Hash::DIGESTSIZE;
    byte digest[digestLen];
    Hash h;
    h.CalculateDigest(digest,
                      (byte *)stringToSign.c_str(),
                      stringToSign.length());
    
    return std::string((const char *)digest,
                       digestLen);
}

template<typename Encoder>
std::string
AwsApi::
encodeDigest(const std::string & digest)
{
    char outBuf[256];

    Encoder encoder;
    encoder.Put((byte *)digest.c_str(), digest.size());
    encoder.MessageEnd();
    size_t got = encoder.Get((byte *)outBuf, 256);
    outBuf[got] = 0;

    //cerr << "signing " << digest.size() << " characters" << endl;
    //cerr << "last character is " << (int)outBuf[got - 1] << endl;
    //cerr << "got " << got << " characters" << endl;

    string result(outBuf, outBuf + got);
    boost::trim(result);
    return result;
}

std::string
AwsApi::
base64EncodeDigest(const std::string & digest)
{
    return encodeDigest<CryptoPP::Base64Encoder>(digest);
}

std::string
AwsApi::
hexEncodeDigest(const std::string & digest)
{
    return ML::lowercase(encodeDigest<CryptoPP::HexEncoder>(digest));
}

std::string
AwsApi::
getStringToSignV2Multi(const std::string & verb,
                       const std::string & bucket,
                       const std::string & resource,
                       const std::string & subResource,
                       const std::string & contentType,
                       const std::string & contentMd5,
                       const std::string & date,
                       const std::vector<std::pair<std::string, std::string> > & headers)
{
    map<string, string> canonHeaders;
    for (auto it = headers.begin(), end = headers.end();
         it != end;  ++it) {
        string key = lowercase(it->first);
        if (key.find("x-amz") != 0) continue;

        string value = it->second;
        if (canonHeaders.count(key))
            canonHeaders[key] += ",";
        canonHeaders[key] += value;
    }
    
    return getStringToSignV2(verb, bucket, resource, subResource,
                             contentType, contentMd5, date, canonHeaders);
}

/*
Authorization = "AWS" + " " + AWSAccessKeyId + ":" + Signature;

Signature = Base64( HMAC-SHA1( YourSecretAccessKeyID, UTF-8-Encoding-Of( StringToSign ) ) );

StringToSign = HTTP-Verb + "\n" +
    Content-MD5 + "\n" +
    Content-Type + "\n" +
    Date + "\n" +
    CanonicalizedAmzHeaders +
    CanonicalizedResource;

CanonicalizedResource = [ "/" + Bucket ] +
    <HTTP-Request-URI, from the protocol name up to the query string> +
    [ sub-resource, if present. For example "?acl", "?location", "?logging", or "?torrent"];

CanonicalizedAmzHeaders = <described below>
To construct the CanonicalizedAmzHeaders part of StringToSign, select all HTTP request headers that start with 'x-amz-' (using a case-insensitive comparison) and use the following process.

CanonicalizedAmzHeaders Process
1	Convert each HTTP header name to lower-case. For example, 'X-Amz-Date' becomes 'x-amz-date'.
2	Sort the collection of headers lexicographically by header name.
3	Combine header fields with the same name into one "header-name:comma-separated-value-list" pair as prescribed by RFC 2616, section 4.2, without any white-space between values. For example, the two metadata headers 'x-amz-meta-username: fred' and 'x-amz-meta-username: barney' would be combined into the single header 'x-amz-meta-username: fred,barney'.
4	"Unfold" long headers that span multiple lines (as allowed by RFC 2616, section 4.2) by replacing the folding white-space (including new-line) by a single space.
5	Trim any white-space around the colon in the header. For example, the header 'x-amz-meta-username: fred,barney' would become 'x-amz-meta-username:fred,barney'
6	Finally, append a new-line (U+000A) to each canonicalized header in the resulting list. Construct the CanonicalizedResource element by concatenating all headers in this list into a single string.


*/
std::string
AwsApi::
getStringToSignV2(const std::string & verb,
                  const std::string & bucket,
                  const std::string & resource,
                  const std::string & subResource,
                  const std::string & contentType,
                  const std::string & contentMd5,
                  const std::string & date,
                  const std::map<std::string, std::string> & headers)
{
    string canonHeaderString;

    for (auto it = headers.begin(), end = headers.end();
         it != end;  ++it) {
        string key = lowercase(it->first);
        if (key.find("x-amz") != 0) continue;

        string value = it->second;

        canonHeaderString += key + ":" + value + "\n";
    }

    //cerr << "bucket = " << bucket << " resource = " << resource << endl;

    string canonResource
        = (bucket == "" ? "" : "/" + bucket)
        + resource
        + (subResource.empty() ? "" : "?")
        + subResource;

    string stringToSign
        = verb + "\n"
        + contentMd5 + "\n"
        + contentType + "\n"
        + date + "\n"
        + canonHeaderString
        + canonResource;

    return stringToSign;
}

std::string
AwsApi::
signV2(const std::string & stringToSign,
       const std::string & accessKey)
{
    return base64EncodeDigest(hmacSha1Digest(stringToSign, accessKey));
}

std::string
AwsApi::
uriEncode(const std::string & str)
{
    std::string result;
    for (auto c: str) {

        if (isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~')
            result += c;
        else result += ML::format("%%%02X", c);
    }

    return result;
}

std::string
AwsApi::
escapeResource(const std::string & resource)
{
    if (resource.size() == 0) {
        throw ML::Exception("empty resource name");
    }

    if (resource[0] != '/') {
        throw ML::Exception("resource name must start with a '/'");
    }

    return "/" + uriEncode(resource.substr(1));
}

std::string
AwsApi::
signingKeyV4(const std::string & accessKey,
             const std::string & date,
             const std::string & region,
             const std::string & service,
             const std::string & signing)
{
    auto hmac = [&] (const std::string & key, const std::string & data)
        {
            return hmacSha256Digest(data, key);
        };
    
    string signingKey
        = hmac(hmac(hmac(hmac("AWS4" + accessKey,
                              date),
                         region),
                    service),
               signing);
    return signingKey;
}

std::string
AwsApi::
signV4(const std::string & stringToSign,
       const std::string & accessKey,
       const std::string & date,
       const std::string & region,
       const std::string & service,
       const std::string & signing)
{
    
    string signingKey = signingKeyV4(accessKey, date, region, service, signing);
    //cerr << "signingKey " << hexEncodeDigest(signingKey) << endl;
    return hexEncodeDigest(hmacSha256Digest(stringToSign, signingKey));
}

void
AwsApi::
addSignatureV4(BasicRequest & request,
               std::string service,
               std::string region,
               std::string accessKeyId,
               std::string accessKey,
               Date now)
{
    string dateStr = now.print("%Y%m%dT%H%M%SZ");

    //cerr << "dateStr = " << dateStr << endl;

    request.headers.push_back({"X-Amz-Date", dateStr});

    string canonicalHeaders;
    string signedHeaders;

    if (!request.headers.empty()) {
        RestParams headers = request.headers;
        for (auto & h: headers) {
            h.first = lowercase(h.first);
            boost::trim(h.second);
        }
        std::sort(headers.begin(), headers.end());
        
        for (auto h: headers) {
            canonicalHeaders += h.first + ":" + h.second + "\n";
            signedHeaders += h.first + ";";
        }

        signedHeaders.erase(signedHeaders.size() - 1);
    }

    string canonicalQueryParams;

    if (!request.queryParams.empty()) {
        RestParams queryParams = request.queryParams;
        std::sort(queryParams.begin(), queryParams.end());
        
        for (auto h: queryParams)
            canonicalQueryParams += uriEncode(h.first) + "=" + uriEncode(h.second) + "&";

        canonicalQueryParams.erase(canonicalQueryParams.size() - 1);
    }

    //cerr << "payload = " << request.payload << endl;

    string payloadHash = hexEncodeDigest(sha256Digest(request.payload));
    
    string canonicalRequest
        = request.method + "\n"
        + "/" + request.relativeUri + "\n"
        + canonicalQueryParams + "\n"
        + canonicalHeaders + "\n"
        + signedHeaders + "\n"
        + payloadHash;

    //cerr << "canonicalRequest = " << canonicalRequest << endl;

    RestParams authParams;

    string authHeader = "AWS4-HMAC-SHA256 ";

    auto addParam = [&] (string key, string value)
        {
            authHeader += key + "=" + value + ", ";

#if 0
            if (request.method == "POST") {
                authHeader

                if (!request.payload.empty())
                    request.payload += "&";
                request.payload += uriEncode(key) + "=" + uriEncode(value);
            }
            else if (request.method == "GET") {
                request.queryParams.push_back({key, value});
            }
#endif
        };



    string credentialScope = string(dateStr, 0, 8) + "/" + region + "/" + service + "/" + "aws4_request";
    
    addParam("Credential", accessKeyId + "/" + credentialScope);
    addParam("SignedHeaders", signedHeaders);
    
    //addParam("SignatureVersion", "4");
    //addParam("SignatureMethod", "AWS4-HMAC-SHA256");

    string hashedCanonicalRequest = hexEncodeDigest(sha256Digest(canonicalRequest));
    
    string stringToSign
        = "AWS4-HMAC-SHA256\n"
        + dateStr + "\n"
        + credentialScope + "\n"
        + hashedCanonicalRequest;

    //cerr << "stringToSign = " << stringToSign << endl;

    string signature = AwsApi::signV4(stringToSign, accessKey, string(dateStr, 0, 8), region, service);
    addParam("Signature", signature);

    authHeader.erase(authHeader.size() - 2);

    request.headers.push_back({"Authorization", authHeader});
}



/*****************************************************************************/
/* AWS BASIC API                                                             */
/*****************************************************************************/

#if 0

AwsBasicApi(const std::string & accessKeyId,
                    const std::string & accessKey,
                    const std::string & service,
                    const std::string & serviceUri = "",
                    const std::string & region = "us-east-1");

    void init(const std::string & accessKeyId,
              const std::string & accessKey,
              const std::string & service,
              const std::string & serviceUri = "",
              const std::string & region = "us-east-1");
              
    std::string accessKeyId;
    std::string accessKey;
    std::string serviceUri;
    std::string service;
    std::string region;

    HttpRestProxy proxy;


void
AwsBasicApi::
init(const std::string & accessKeyId,
     const std::string & accessKey,
     const std::string & serviceUri)
{
    this->serviceUri = serviceUri;

}

#endif

AwsBasicApi::
AwsBasicApi()
{
}

void
AwsBasicApi::
setService(const std::string & serviceName,
           const std::string & protocol,
           const std::string & region)
{
    this->serviceName = serviceName;
    this->protocol = protocol;
    this->region = region;

    this->serviceHost = serviceName + "." + region + ".amazonaws.com";
    this->serviceUri = protocol + "://" + serviceHost + "/";

    proxy.init(serviceUri);
    //proxy.debug = true;
}

void
AwsBasicApi::
setCredentials(const std::string & accessKeyId,
               const std::string & accessKey)
{
    this->accessKeyId = accessKeyId;
    this->accessKey = accessKey;
}

AwsBasicApi::BasicRequest
AwsBasicApi::
signPost(RestParams && params, const std::string & resource)
{
    BasicRequest result;
    result.method = "POST";
    result.relativeUri = resource;
    result.headers.push_back({"Host", serviceHost});
    result.headers.push_back({"Content-Type", "application/x-www-form-urlencoded; charset=utf-8"});

    std::string encodedPayload;

    for (auto p: params) {
        encodedPayload += p.first + "=";
        encodedPayload += uriEncode(p.second) + "&";
    }

    if (!params.empty())
        encodedPayload.erase(encodedPayload.size() - 1);

    //cerr << "encodedPayload = " << encodedPayload << endl;

    result.payload = encodedPayload;
    
    addSignatureV4(result, serviceName, region, accessKeyId, accessKey);

    return result;

}

AwsBasicApi::BasicRequest
AwsBasicApi::
signGet(RestParams && params, const std::string & resource)
{
    BasicRequest result;
    result.method = "GET";
    result.relativeUri = resource;
    result.headers.push_back({"Host", serviceHost});
    result.queryParams = params;

    addSignatureV4(result, serviceName, region, accessKeyId, accessKey);

    return result;
}

std::unique_ptr<tinyxml2::XMLDocument>
AwsBasicApi::
performPost(RestParams && params,
            const std::string & resource,
            double timeoutSeconds)
{
    return perform(signPost(std::move(params), resource), timeoutSeconds, 3);
}

std::string
AwsBasicApi::
performPost(RestParams && params,
            const std::string & resource,
            const std::string & resultSelector,
            double timeoutSeconds)
{
    return extract<string>(*performPost(std::move(params), resource, timeoutSeconds),
                           resultSelector);
}

std::unique_ptr<tinyxml2::XMLDocument>
AwsBasicApi::
performGet(RestParams && params,
           const std::string & resource,
           double timeoutSeconds)
{
    return perform(signGet(std::move(params), resource), timeoutSeconds, 3);
}

std::unique_ptr<tinyxml2::XMLDocument>
AwsBasicApi::
perform(const BasicRequest & request,
        double timeoutSeconds,
        int retries)
{
    int retry = 0;
    for (; retry < retries;  ++retry) {
        HttpRestProxy::Response response;
        try {
            response = proxy.perform(request.method,
                                     request.relativeUri,
                                     HttpRestProxy::Content(request.payload),
                                     request.queryParams,
                                     request.headers,
                                     timeoutSeconds);

            if (response.code() == 200) {
                std::unique_ptr<tinyxml2::XMLDocument> body(new tinyxml2::XMLDocument());
                body->Parse(response.body().c_str());
                return body;
            }
            else if (response.code() == 503)
                continue;
            else {
                cerr << "request failed: " << response << endl;
                break;
            }
        } catch (const std::exception & exc) {
            cerr << "error on request: " << exc.what() << endl;
        }
    }

    throw ML::Exception("failed request after %d retries", retries);
}

std::string
AwsBasicApi::
performGet(RestParams && params,
           const std::string & resource,
           const std::string & resultSelector,
           double timeoutSeconds)
{
    return extract<string>(*performGet(std::move(params), resource, timeoutSeconds),
                           resultSelector);
}

void registerAwsCredentials(const string & accessKeyId,
                            const string & accessKey)
{
    unique_lock<mutex> guard(awsCredentialsLock);

    string & entry = awsCredentials[accessKeyId];
    if (entry.empty()) {
        entry = accessKey;
    }
    else {
        if (entry != accessKey) {
            throw ML::Exception("access key id '%s' already registered with a"
                                " different key", accessKeyId.c_str());
        }
    }
}

string getAwsAccessKey(const string & accessKeyId)
{
    auto it = awsCredentials.find(accessKeyId);
    if (it == awsCredentials.end()) {
        throw ML::Exception("no access key registered for id '%s'",
                            accessKeyId.c_str());
    }

    return it->second;
}

} // namespace Datacratic

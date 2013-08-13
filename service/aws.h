/* aws.h                                                           -*- C++ -*-
   Jeremy Barnes, 8 August 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Amazon Web Services support code, especially signing of requests.
*/

#pragma once
#include <string>
#include "http_rest_proxy.h"


namespace Datacratic {


/*****************************************************************************/
/* AWS API                                                                   */
/*****************************************************************************/

/** Base functionality for dealing with Amazon's APIs. */

struct AwsApi {
    template<typename Hash>
    static std::string hmacDigest(const std::string & stringToSign,
                                  const std::string & accessKey);

    template<typename Encoder>
    static std::string encodeDigest(const std::string & digest);

    static std::string hmacSha1Digest(const std::string & stringToSign,
                                      const std::string & accessKey);

    static std::string hmacSha256Digest(const std::string & stringToSign,
                                        const std::string & accessKey);
    
    static std::string sha256Digest(const std::string & stringToSign);

    static std::string base64EncodeDigest(const std::string & digest);

    static std::string hexEncodeDigest(const std::string & digest);

    /** Sign the given digest string with the given access key and return
        a base64 encoded signature.
    */
    static std::string signV2(const std::string & stringToSign,
                              const std::string & accessKey);

    /** URI encode the given string according to RFC 3986 */
    static std::string uriEncode(const std::string & str);

    /** See http://docs.aws.amazon.com/general/latest/gr/sigv4-calculate-signature.html */
    static std::string signingKeyV4(const std::string & accessKey,
                                    const std::string & date,
                                    const std::string & region,
                                    const std::string & service,
                                    const std::string & signing = "aws4_request");
    
    /** See http://docs.aws.amazon.com/general/latest/gr/sigv4-calculate-signature.html */
    static std::string signV4(const std::string & stringToSign,
                              const std::string & accessKey,
                              const std::string & date,
                              const std::string & region,
                              const std::string & service,
                              const std::string & signing = "aws4_request");

    struct BasicRequest {
        std::string method;
        std::string relativeUri;
        RestParams queryParams;
        RestParams headers;
        std::string payload;
    };
    
    static void
    addSignatureV4(BasicRequest & request,
                   std::string service,
                   std::string region,
                   std::string accessKeyId,
                   std::string accessKey,
                   Date now = Date::now());
};


/*****************************************************************************/
/* AWS BASIC API                                                             */
/*****************************************************************************/

/** Base class for basic services that use the Action API and signature
    V4.
*/

struct AwsBasicApi : public AwsApi {

    AwsBasicApi();

    void setService(const std::string & serviceName,
                    const std::string & protocol = "http",
                    const std::string & region = "us-east-1");

    void setCredentials(const std::string & accessKeyId,
                        const std::string & accessKey);
              
    std::string accessKeyId;
    std::string accessKey;

    std::string protocol;
    std::string serviceName;
    std::string serviceHost;
    std::string region;
    std::string serviceUri;

    std::string performPost(RestParams && params, const std::string & resultSelector);
    std::string performGet(RestParams && params, const std::string & resultSelector);
    
    BasicRequest signPost(RestParams && params);
    BasicRequest signGet(RestParams && params);

    HttpRestProxy proxy;
};

} // namespace Datacratic

/* sns.h                                                           -*- C++ -*-
   Jeremy Barnes, 12 May 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Client for Amazon's Simple Notification Service.
*/

#pragma once

#include "s3.h"

namespace Datacratic {


/*****************************************************************************/
/* SNS API                                                                   */
/*****************************************************************************/
 
struct SnsApi {

    std::string accessKeyId;
    std::string accessKey;
    std::string serviceUri;

    HttpRestProxy proxy;

    SnsApi()
    {
    }

    /** Set up the API to called with the given credentials. */
    SnsApi(const std::string & accessKeyId,
          const std::string & accessKey,
          const std::string & serviceUri = "http://sns.us-east-1.amazonaws.com/")
    {
        init(accessKeyId, accessKey, serviceUri);
    }

    /** Set up the API to called with the given credentials. */
    void init(const std::string & accessKeyId,
              const std::string & accessKey,
              const std::string & serviceUri = "http://sns.us-east-1.amazonaws.com/")
    {
        this->accessKeyId = accessKeyId;
        this->accessKey = accessKey;
        this->serviceUri = serviceUri;

        proxy.init(serviceUri);
        proxy.debug = true;
    }

    void publish(const std::string & topicArn,
                 const std::string & message,
                 const std::string & subject = "",
                 const std::map<std::string, std::string> & protocolMessages
                 = std::map<std::string, std::string>())
    {
        using namespace std;
        RestParams queryParams;
        if (subject != "")
            queryParams.push_back({"Subject", subject});
        queryParams.push_back({"TopicArn", topicArn});
        queryParams.push_back({"Action", "Publish"});
        queryParams.push_back({"Message", message});

        string timestamp = Date::now().printIso8601();
        queryParams.push_back({"Timestamp", timestamp});

        queryParams.push_back({"SignatureVersion", "2"});
        queryParams.push_back({"SignatureMethod", "HmacSHA1"});
        queryParams.push_back({"AWSAccessKeyId", accessKeyId});

        std::sort(queryParams.begin(), queryParams.end());

        string host = "sns.us-east-1.amazonaws.com";  // TODO: really do
        string path = "/";
        
        string toSign = "POST\n";
        toSign += host + "\n";
        toSign += path + "\n";
        
        for (unsigned i = 0;  i < queryParams.size();  ++i) {
            if (i != 0)
                toSign += "&";
            toSign += S3Api::uriEncode(queryParams[i].first);
            toSign += "=";
            toSign += S3Api::uriEncode(queryParams[i].second);
        }

        //cerr << "toSign = " << toSign << endl;

        string signature = S3Api::sign(toSign, accessKey);

        queryParams.push_back({"Signature", signature});

        auto result = proxy.post("", HttpRestProxy::Content(), queryParams);

        //cerr << result << endl;
    }

};


} // namespace Datacratic

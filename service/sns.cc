/* sns.cc
   Jeremy Barnes, 12 May 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Basic interface to Amazon's SNS service.
*/

#include "sns.h"
#include "xml_helpers.h"


using namespace std;


namespace Datacratic {


/*****************************************************************************/
/* SNS API                                                                   */
/*****************************************************************************/

SnsApi::
SnsApi()
{
}
 
SnsApi::
SnsApi(const std::string & accessKeyId,
       const std::string & accessKey,
       const std::string & serviceUri)
{
    init(accessKeyId, accessKey, serviceUri);
}

void
SnsApi::
init(const std::string & accessKeyId,
     const std::string & accessKey,
     const std::string & serviceUri)
{
    this->accessKeyId = accessKeyId;
    this->accessKey = accessKey;
    this->serviceUri = serviceUri;

    proxy.init(serviceUri);
    //proxy.debug = true;
}

std::string
SnsApi::
publish(const std::string & topicArn,
        const std::string & message,
        int timeout,
        const std::string & subject)
{
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
        toSign += uriEncode(queryParams[i].first);
        toSign += "=";
        toSign += uriEncode(queryParams[i].second);
    }

    //cerr << "toSign = " << toSign << endl;

    string signature = signV2(toSign, accessKey);

    queryParams.push_back({"Signature", signature});

    int retry = 0;
    for (; retry < 3;  ++retry) {
        HttpRestProxy::Response response;
        try {
            response = proxy.post("", HttpRestProxy::Content(), queryParams, {}, timeout);

            if (response.code() == 200) {
                tinyxml2::XMLDocument body;
                body.Parse(response.body().c_str());

                string messageId
                    = extract<string>(body, "PublishResponse/PublishResult/MessageId");
                return messageId;
            }
            
            cerr << "request failed: " << response << endl;
        } catch (const std::exception & exc) {
            cerr << "error on SNS notification: " << exc.what() << endl;
        }
    }

    throw ML::Exception("failed SNS request after 3 retries");
}

} // namespace Datacratic

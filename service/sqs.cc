/* sqs.cc
   Jeremy Barnes, 12 May 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Basic interface to Amazon's SQS service.
*/

#include "sqs.h"
#include "xml_helpers.h"
#include <boost/algorithm/string.hpp>


using namespace std;
using namespace ML;


namespace Datacratic {


/*****************************************************************************/
/* SQS API                                                                   */
/*****************************************************************************/

SqsApi::
SqsApi(const std::string & protocol,
       const std::string & region)
{
    setService("sqs", protocol, region);
}
 
std::string
SqsApi::
createQueue(const std::string & queueName,
            const QueueParams & params)
{
    RestParams queryParams;
    queryParams.push_back({"Action", "CreateQueue"});
    queryParams.push_back({"QueueName", queueName});
    queryParams.push_back({"Version", "2012-11-05"});

    return performPost(std::move(queryParams), "CreateQueueResponse/CreateQueueResult/QueueUrl");
}

std::string
SqsApi::
getQueueUrl(const std::string & queueName)
{
    RestParams queryParams;
    queryParams.push_back({"Action", "GetQueueUrl"});
    queryParams.push_back({"QueueName", queueName});
    queryParams.push_back({"Version", "2012-11-05"});

    return performGet(std::move(queryParams), "GetQueueUrlResponse/GetQueueUrlResult/QueueUrl");
}


#if 0
    // Attributes...

    string method = "POST";
    string canonicalUri = "/";
    string canonicalQueryString = ...;
    string canonicalHeaders = ...;
    string signedHeaders = ...;
    string payload = ...;

    

    string canonicalRequest = "POST\n";


    CanonicalRequest =
  HTTPRequestMethod + '\n' +
  CanonicalURI + '\n' +
  CanonicalQueryString + '\n' +
  CanonicalHeaders + '\n' +
  SignedHeaders + '\n' +
  HexEncode(Hash(Payload))
    

http://sqs.us-east-1.amazonaws.com/
?Action=CreateQueue
&QueueName=testQueue
&Attribute.1.Name=VisibilityTimeout
&Attribute.1.Value=40
&Version=2011-10-01
&SignatureMethod=HmacSHA256
&Expires=2011-10-18T22%3A52%3A43PST
&AWSAccessKeyId=AKIAIOSFODNN7EXAMPLE
&SignatureVersion=2
&Signature=Dqlp3Sd6ljTUA9Uf6SGtEExwUQEXAMPLE

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

    string host = "sqs.us-east-1.amazonaws.com";  // TODO: really do
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
            cerr << "error on SQS notification: " << exc.what() << endl;
        }
    }

    throw ML::Exception("failed SQS request after 3 retries");
}
#endif

#if 0
std::string
SqsApi::
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

    string host = "sqs.us-east-1.amazonaws.com";  // TODO: really do
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
            cerr << "error on SQS notification: " << exc.what() << endl;
        }
    }

    throw ML::Exception("failed SQS request after 3 retries");
}
#endif

} // namespace Datacratic

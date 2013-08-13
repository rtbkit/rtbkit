/* sqs.cc
   Jeremy Barnes, 12 May 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Basic interface to Amazon's SQS service.
*/

#include "sqs.h"
#include "xml_helpers.h"
#include <boost/algorithm/string.hpp>
#include "jml/utils/exc_assert.h"


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

    return performPost(std::move(queryParams), "",
                       "CreateQueueResponse/CreateQueueResult/QueueUrl");
}

std::string
SqsApi::
getQueueUrl(const std::string & queueName,
            const std::string & ownerAccountId)
{
    RestParams queryParams;
    queryParams.push_back({"Action", "GetQueueUrl"});
    queryParams.push_back({"QueueName", queueName});
    queryParams.push_back({"Version", "2012-11-05"});
    if (ownerAccountId != "")
        queryParams.push_back({"QueueOwnerAWSAccountId", ownerAccountId});

    return performGet(std::move(queryParams), "",
                      "GetQueueUrlResponse/GetQueueUrlResult/QueueUrl");
}

std::string
SqsApi::
sendMessage(const std::string & queueUri,
            const std::string & message,
            int timeoutSeconds,
            int delaySeconds)
{
    RestParams queryParams;
    queryParams.push_back({"Action", "SendMessage"});
    queryParams.push_back({"Version", "2012-11-05"});
    queryParams.push_back({"MessageBody", message});

    return performPost(std::move(queryParams), getQueueResource(queueUri),
                       "SendMessageResponse/SendMessageResult/MD5OfMessageBody");
}

SqsApi::Message
SqsApi::
receiveMessage(const std::string & queueUri,
               int visibilityTimeout,
               int waitTimeSeconds)
{
    RestParams queryParams;
    queryParams.push_back({"Action", "ReceiveMessage"});
    queryParams.push_back({"Version", "2012-11-05"});
    if (visibilityTimeout != -1)
        queryParams.push_back({"VisibilityTimeout", to_string(visibilityTimeout)});
    if (waitTimeSeconds != -1)
        queryParams.push_back({"WaitTimeSeconds", to_string(waitTimeSeconds)});
    
    auto xml = performGet(std::move(queryParams), getQueueResource(queueUri));

    xml->Print();

    throw Exception("not finished");
}

std::string
SqsApi::
getQueueResource(const std::string & queueUri) const
{
    ExcAssert(!serviceUri.empty());

    if (queueUri.find(serviceUri) != 0)
        throw ML::Exception("unknown queue URI");
    string resource(queueUri, serviceUri.size());

    return resource;
}


} // namespace Datacratic

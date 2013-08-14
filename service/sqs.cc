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

void
SqsApi::
deleteMessage(const std::string & queueUri,
              const std::string & receiptHandle)
{
    RestParams queryParams;
    queryParams.push_back({"Action", "ReceiveMessage"});
    queryParams.push_back({"ReceiptHandle", receiptHandle});
    queryParams.push_back({"Version", "2012-11-05"});

    auto xml = performGet(std::move(queryParams), getQueueResource(queueUri));

    xml->Print();
}

void
SqsApi::
deleteMessageBatch(const std::string & queueUri,
                   const std::vector<std::string> & receiptHandles)
{
    RestParams queryParams;
    queryParams.push_back({"Action", "ReceiveMessage"});
    queryParams.push_back({"Version", "2012-11-05"});

    for (int i = 0; i < receiptHandles.size(); i++) {
        string prefix = "DeleteMessageBatchRequestEntry." + to_string(i);
        queryParams.push_back({prefix + ".Id", "msg" + to_string(i)});
        queryParams.push_back({prefix + ".ReceiptHandle", receiptHandles[i]});
    }
    
    auto xml = performGet(std::move(queryParams), getQueueResource(queueUri));

    xml->Print();
}

void
SqsApi::
changeMessageVisibility(const std::string & queueUri,
                        const std::string & receiptHandle,
                        int visibilityTimeout)
{
    RestParams queryParams;
    queryParams.push_back({"Action", "ChangeMessageVisibility"});
    queryParams.push_back({"Version", "2012-11-05"});
    queryParams.push_back({"ReceiptHandle", receiptHandle});
    queryParams.push_back({"VisibilityTimeout", to_string(visibilityTimeout)});

    auto xml = performGet(std::move(queryParams), getQueueResource(queueUri));

    xml->Print();
}

void
SqsApi::
changeMessageVisibilityBatch(const std::string & queueUri,
                             const std::vector<VisibilityPair> & visibilities)
{
    RestParams queryParams;
    queryParams.push_back({"Action", "ChangeMessageVisibilityBatch"});
    queryParams.push_back({"Version", "2012-11-05"});

    int counter(0);
    for (const auto & pair: visibilities) {
        string prefix = ("ChangeMessageVisibilityBatchRequestEntry."
                         + to_string(counter));
        queryParams.push_back({prefix + ".Id", pair.receiptHandle});
        queryParams.push_back({prefix + ".VisibilityTimeout",
                               to_string(pair.visibilityTimeout)});
    }

    auto xml = performGet(std::move(queryParams), getQueueResource(queueUri));

    xml->Print();
}

void
SqsApi::
addPermission(const std::string & queueUri, const std::string & label,
              const vector<RightsPair> & rights)
{
    RestParams queryParams;
    queryParams.push_back({"Action", "ReceiveMessage"});
    queryParams.push_back({"Version", "2012-11-05"});
    queryParams.push_back({"Label", label});

    int counter(0);
    for (const RightsPair & pair: rights) {
        if (pair.rights == Rights::All) {
            queryParams.push_back({"AWSAccountId." + to_string(counter),
                                   pair.principal});
            queryParams.push_back({"ActionName." + to_string(counter),
                                   "*"});
            counter++;
        }
        else {
            Rights currentRights = pair.rights;
            for (int i = 0; currentRights != None && i < 5; i++) {
                Rights currentRight = static_cast<Rights>(1 << i);
                if (currentRights & currentRight) {
                    queryParams.push_back({"AWSAccountId." + to_string(counter),
                                           pair.principal});
                    queryParams.push_back({"ActionName." + to_string(counter),
                                           SqsApi::rightToString(currentRight)});
                    counter++;
                    currentRights = static_cast<Rights>(currentRights & ~currentRight);
                }
            }
        }
    }

    auto xml = performGet(std::move(queryParams), getQueueResource(queueUri));

    xml->Print();
}

void
SqsApi::
removePermission(const std::string & queueUri, const std::string & label)
{
    RestParams queryParams;
    queryParams.push_back({"Action", "ReceiveMessage"});
    queryParams.push_back({"Version", "2012-11-05"});
    queryParams.push_back({"Label", label});

    auto xml = performGet(std::move(queryParams), getQueueResource(queueUri));

    xml->Print();
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

std::string
SqsApi::
rightToString(enum SqsApi::Rights rights)
{
    switch (rights) {
    case SendMessage: return "SendMessage";
    case DeleteMessage: return "DeleteMessage";
    case ChangeMessageVisibility: return "ChangeMessageVisibility";
    case GetQueueAttributes: return "GetQueueAttributes";
    case GetQueueUrl: return "GetQueueUrl";
    case All: return "*";
    default:
        throw ML::Exception("unknown right");
    };
}


} // namespace Datacratic

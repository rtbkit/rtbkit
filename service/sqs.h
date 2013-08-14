/* sqs.h                                                           -*- C++ -*-
   Jeremy Barnes, 12 May 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Client for Amazon's Simple Notification Service.
*/

#pragma once

#include "aws.h"
#include "http_rest_proxy.h"


namespace Datacratic {


/*****************************************************************************/
/* SQS API                                                                   */
/*****************************************************************************/
 
struct SqsApi : public AwsBasicApi {

    SqsApi(const std::string & protocol = "http",
           const std::string & region = "us-east-1");

    /** Parameters to create a queue */
    struct QueueParams {
        QueueParams()
            : delaySeconds(-1),
              maximumMessageSize(-1),
              messageRetentionPeriod(-1),
              receiveMessageWaitTimeSeconds(-1),
              visibilityTimeout(-1)
        {
        }

        int delaySeconds;
        int maximumMessageSize;
        int messageRetentionPeriod;
        std::string policy;
        int receiveMessageWaitTimeSeconds;
        int visibilityTimeout;
    };

    /** Create a queue.
     */
    std::string createQueue(const std::string & queueName,
                            const QueueParams & params = QueueParams());
                            
    /** Return the URL for the given queue. */
    std::string getQueueUrl(const std::string & queueName,
                            const std::string & ownerAccountId = "");

    /** Publish a message to a given SQS queue.  Returns the Message ID assigned
        by Amazon.

        By default, this will retry a failure 3 times before throwing an
        exception.

        \param queueName     The queue name to send the message to
        \param accountOwner  The account ID of the owner of the queue
        \param message       The message to be sent
        \param timeout       The timeout after which to retry
        \param delay         How many seconds (0-900) to delay the message within the
                             queue (see the AWS documentation).

        See also http://docs.aws.amazon.com/AWSSimpleQueueService/latest/APIReference/Query_QuerySendMessage.html
    */
    std::string
    sendMessage(const std::string & queueUrl,
                const std::string & message,
                int timeoutSeconds = 10,
                int delaySeconds = -1);

    struct Message {
        std::string body;
        std::string bodyMd5;
        std::string messageId;
        std::string receiptHandle;
        std::string senderId;
        Date sentTimestamp;
        int approximateReceiveCount;
        Date approximateFirstReceiveTimestamp;
    };

    Message receiveMessage(const std::string & queueUri,
                           int visibilityTimeout = -1,
                           int waitTimeSeconds = -1);

    /* Delete a message from a queue.

       \param receiptHandle   The receipt handle identifying the message.
                              Note: this is not the message id

    */
    void deleteMessage(const std::string & queueUri,
                       const std::string & receiptHandle);

    /** Turns a queue URI into a relative resource path for the HttpRestProxy */
    std::string getQueueResource(const std::string & queueUri) const;

};


} // namespace Datacratic

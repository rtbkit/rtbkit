/* sns.h                                                           -*- C++ -*-
   Jeremy Barnes, 12 May 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Client for Amazon's Simple Notification Service.
*/

#pragma once

#include <queue>
#include "aws.h"
#include "http_rest_proxy.h"


namespace Datacratic {


/*****************************************************************************/
/* SNS API                                                                   */
/*****************************************************************************/
 
struct SnsApi : public AwsBasicApi {
    SnsApi();

    /** Set up the API to called with the given credentials. */
    SnsApi(const std::string & accessKeyId,
           const std::string & accessKey);

    /** Set up the API to called with the given credentials. */
    void init(const std::string & accessKeyId,
              const std::string & accessKey);

    /** Publish a message to a given SNS topic.  Returns the Message ID assigned
        by Amazon.

        By default, this will retry a failure 3 times before throwing an
        exceptoin.

        \param topicArn      The Amazon topic to send the message to.
        \param message       The message to be sent
        \param timeout       The timeout after which to retry
        \param subject       The optional subject to give to the message
    */
    std::string
    publish(const std::string & topicArn,
            const std::string & message,
            int timeout = 10,
            const std::string & subject = "");
};

/**
 * Wraps SnsApi in order to use the same topic arn on each publish
 */
struct SnsApiWrapper {
    protected:
        SnsApi api;
        std::string defaultTopicArn;
        SnsApiWrapper(){}

    public:
        SnsApiWrapper(const std::string & accessKeyId,
                      const std::string & accessKey,
                      const std::string & defaultTopicArn) {
            api.init(accessKeyId, accessKey);
            this->defaultTopicArn = defaultTopicArn;
        }

        virtual void init(const std::string & accessKeyId,
                          const std::string & accessKey,
                          const std::string & defaultTopicArn) {
            api.init(accessKeyId, accessKey);
            this->defaultTopicArn = defaultTopicArn;
        }

        virtual std::string
        publish(const std::string & message,
                int timeout = 10,
                const std::string & subject = "") {
            return api.publish(defaultTopicArn, message, timeout, subject);
        }
};

struct MockSnsApiWrapper : SnsApiWrapper {

    private:
        int cacheSize;

    public:
        std::queue<std::string> queue;

        MockSnsApiWrapper(int cacheSize = 0) : cacheSize(cacheSize){
            if (cacheSize < 0) {
                throw ML::Exception("Cache size cannot be below 0");
            }
        }

        void init(const std::string & accessKeyId,
                  const std::string & accessKey,
                  const std::string & fdefaultTopicArn) {}

        std::string
        publish(const std::string & message,
                int timeout = 10,
                const std::string & subject = "") {
            if (cacheSize == 0) {
                return "";
            }
            while (queue.size() >= cacheSize) {
                queue.pop();
            }
            queue.push(message);
            return "";
        }

        MockSnsApiWrapper(const std::string & accessKeyId,
                          const std::string & accessKey,
                          const std::string & defaultTopicArn) = delete;

};
} // namespace Datacratic

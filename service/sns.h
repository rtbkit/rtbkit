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

    SnsApi();

    /** Set up the API to called with the given credentials. */
    SnsApi(const std::string & accessKeyId,
           const std::string & accessKey,
           const std::string & serviceUri = "http://sns.us-east-1.amazonaws.com/");

    /** Set up the API to called with the given credentials. */
    void init(const std::string & accessKeyId,
              const std::string & accessKey,
              const std::string & serviceUri = "http://sns.us-east-1.amazonaws.com/");

    std::string publish(const std::string & topicArn,
                        const std::string & message,
                        int timeout = 10,
                        const std::string & subject = "",
                        const std::map<std::string, std::string> & protocolMessages
                            = std::map<std::string, std::string>());
};


} // namespace Datacratic

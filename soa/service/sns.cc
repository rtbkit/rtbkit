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
       const std::string & accessKey)
{
    init(accessKeyId, accessKey);
}

void
SnsApi::
init(const std::string & accessKeyId,
     const std::string & accessKey)
{
    setService("sns");
    setCredentials(accessKeyId, accessKey);
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

    return performPost(std::move(queryParams), "",
                       "PublishResponse/PublishResult/MessageId",
                       timeout);
}

} // namespace Datacratic

/* aws.h                                                           -*- C++ -*-
   Jeremy Barnes, 8 August 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Amazon Web Services support code, especially signing of requests.
*/

#pragma once
#include <string>

namespace Datacratic {


/*****************************************************************************/
/* AWS                                                                       */
/*****************************************************************************/

struct AwsApi {
    /** Sign the given digest string with the given access key and return
        a base64 encoded signature.
    */
    static std::string signV2(const std::string & stringToSign,
                              const std::string & accessKey);

    /** URI encode the given string according to RFC 3986 */
    static std::string uriEncode(const std::string & str);


};


} // namespace Datacratic

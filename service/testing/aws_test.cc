/* aws_test.cc
   Jeremy Barnes, 12 May 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

   Test of the basic functionality of authenticating AWS requests.
*/


#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "soa/service/sqs.h"
#include "jml/utils/file_functions.h"
#include <iostream>
#include <stdlib.h>
#include "jml/utils/vector_utils.h"
#include "jml/utils/pair_utils.h"


using namespace std;
using namespace Datacratic;
using namespace ML;

BOOST_AUTO_TEST_CASE( test_signing_v4 )
{
    // Test cases are from
    // http://docs.aws.amazon.com/general/latest/gr/sigv4-calculate-signature.html

    string sampleSigningKey = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";
    string digest = AwsApi::signingKeyV4(sampleSigningKey, "20110909","us-east-1","iam","aws4_request");
    string hexDigest = AwsApi::hexEncodeDigest(digest);

    BOOST_CHECK_EQUAL(hexDigest, "98f1d889fec4f4421adc522bab0ce1f82e6929c262ed15e5a94c90efd1e3b0e7");

    
    string stringToSign =
        "AWS4-HMAC-SHA256\n"
        "20110909T233600Z\n"
        "20110909/us-east-1/iam/aws4_request\n"
        "3511de7e95d28ecd39e9513b642aee07e54f4941150d8df8bf94b328ef7e55e2";

    string signature = AwsApi::signV4(stringToSign, sampleSigningKey, "20110909", "us-east-1", "iam", "aws4_request");

    BOOST_CHECK_EQUAL(signature, "ced6826de92d2bdeed8f846f0bf508e8559e98e4b0199114b84c54174deb456c");
}


BOOST_AUTO_TEST_CASE( check_canonical_request )
{
    // See here:

    // http://docs.aws.amazon.com/general/latest/gr/sigv4-signed-request-examples.html

    /*
      POST http://iam.amazonaws.com/ HTTP/1.1
      host: iam.amazonaws.com
      Content-type: application/x-www-form-urlencoded; charset=utf-8
      x-amz-date: 20110909T233600Z

      Action=ListUsers&Version=2010-05-08
    */

    // Authorization: AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20110909/us-east-1/iam/aws4_request, SignedHeaders=content-type;host;x-amz-date, Signature=ced6826de92d2bdeed8f846f0bf508e8559e98e4b0199114b84c54174deb456c

    
    //QueryParams params;
    //params.push_back({"Action","ListUsers"});
    //params.push_back({"Version","2010-05-08"});

    string accessKeyId = "AKIDEXAMPLE";
    string accessKey   = "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY";

    AwsApi::BasicRequest request;
    request.method = "POST";
    request.relativeUri = "";
    request.headers.push_back({"host", "iam.amazonaws.com"});
    request.headers.push_back({"Content-Type", "application/x-www-form-urlencoded; charset=utf-8"});
    //request.headers.push_back({"x-amz-date", "20110909T233600Z"});
    request.payload = "Action=ListUsers&Version=2010-05-08";


    AwsApi::addSignatureV4(request, "iam", "us-east-1", accessKeyId, accessKey, Date(2011,9,9,23,36,00));

    string auth;

    for (auto h: request.headers)
        if (h.first == "Authorization")
            auth = h.second;

    BOOST_CHECK_EQUAL(auth, "AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/20110909/us-east-1/iam/aws4_request, SignedHeaders=content-type;host;x-amz-date, Signature=ced6826de92d2bdeed8f846f0bf508e8559e98e4b0199114b84c54174deb456c");
}

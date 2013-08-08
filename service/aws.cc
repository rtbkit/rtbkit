/* aws.cc
   Jeremy Barnes, 8 August 2013
   Copyright (c) 2013 Datacratic Inc.  All rights reserved.

*/

#include "aws.h"
#include "jml/arch/format.h"

#define CRYPTOPP_ENABLE_NAMESPACE_WEAK 1
#include "crypto++/sha.h"
#include "crypto++/md5.h"
#include "crypto++/hmac.h"
#include "crypto++/base64.h"


using namespace std;


namespace Datacratic {

std::string
AwsApi::
signV2(const std::string & stringToSign,
       const std::string & accessKey)
{
    typedef CryptoPP::SHA1 Hash;

    size_t digestLen = Hash::DIGESTSIZE;
    byte digest[digestLen];
    CryptoPP::HMAC<Hash> hmac((byte *)accessKey.c_str(), accessKey.length());
    hmac.CalculateDigest(digest,
                         (byte *)stringToSign.c_str(),
                         stringToSign.length());

    // base64
    char outBuf[256];

    CryptoPP::Base64Encoder baseEncoder;
    baseEncoder.Put(digest, digestLen);
    baseEncoder.MessageEnd();
    size_t got = baseEncoder.Get((byte *)outBuf, 256);
    outBuf[got] = 0;

    //cerr << "got " << got << " characters" << endl;

    string base64digest(outBuf, outBuf + got - 1);

    //cerr << "base64digest.size() = " << base64digest.size() << endl;

    return base64digest;
}

std::string
AwsApi::
uriEncode(const std::string & str)
{
    std::string result;
    for (auto c: str) {
        if (c <= ' ' || c >= 127) {
            result += ML::format("%%%02X", c);
            continue;
        }

        switch (c) {
        case '!':
        case '#':
        case '$':
        case '&':
        case '\'':
        case '(':
        case ')':
        case '*':
        case '+':
        case ',':
        case '/':
        case ':':
        case ';':
        case '=':
        case '?':
        case '@':
        case '[':
        case ']':
        case '%':
            result += ML::format("%%%02X", c);
            break;

        default:
            result += c;
        }
    }

    return result;
}

} // namespace Datacratic

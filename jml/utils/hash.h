/* md5.h                                                           -*- C++ -*-
   Jeremy Barnes, 25 October 2012
   Copyright (c) 2012 Datacratic.  All rights reserved.

   Md5 hash functions.
*/

#ifndef __jml__utils__md5_h__
#define __jml__utils__md5_h__

#include <string>

namespace ML {

std::string base64Encode(const std::string & str);
std::string base64Encode(const char * buf, size_t nBytes);

std::string md5HashToHex(const std::string & str);
std::string md5HashToHex(const char * buf, size_t nBytes);

std::string md5HashToBase64(const std::string & str);
std::string md5HashToBase64(const char * buf, size_t nBytes);

std::string hmacSha1Base64(const std::string & stringToSign,
                           const std::string & privateKey);
std::string hmacSha256Base64(const std::string & stringToSign,
                             const std::string & privateKey);



} // namespace ML

#endif /* __jml__utils__md5_h__ */

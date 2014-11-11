/* string_encryption.h                                    -*- C++ -*-
   Michael Burkat, 7 Octobre 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

*/

#pragma once

#include <string>
#include "cryptopp/hex.h"

namespace Datacratic {

/*****************************************************************************/
/* PASSBACK ENCRYPTION                                                       */
/*****************************************************************************/

/**
 * Used to encrypt and decrypt passbacks in exchange connector
*/

struct StringEncryption {

    typedef unsigned char byte;
    
    StringEncryption();

    std::string generateKey();

    std::string generateIV();
    
    std::string encrypt(const std::string & passback,
                        const std::string & key,
                        const std::string & iv);

    std::string decrypt(const std::string & passback,
                        const std::string & key,
                        const std::string & iv);
private: 

    CryptoPP::HexEncoder hexEncoder;
    CryptoPP::HexDecoder hexDecoder;

    std::string digest(const std::string & encrypted);

    std::string addDigest(const std::string & encrypted);

    std::string removeDigest(const std::string & digested);

    std::string byteToStr(byte * encode, size_t size);

    std::string hexEncode(const std::string & decoded);

    std::string hexDecode(const std::string & encoded);
};

} // namespace Datacratic

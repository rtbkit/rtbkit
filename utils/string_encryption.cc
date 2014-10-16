/* string_encryption.cc                                   -*- C++ -*-
   Michael Burkat, 7 Octobre 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

*/

#include "string_encryption.h"
#include "cryptopp/modes.h"
#include "cryptopp/crc.h"
#include "cryptopp/osrng.h"
#include "cryptopp/aes.h"
#include <iostream>

using namespace std; 
using namespace CryptoPP;

namespace Datacratic {

/*****************************************************************************/
/* PASSBACK ENCRYPTION                                                       */
/*****************************************************************************/

/**
 * Used to encrypt and decrypt passbacks in exchange connector
*/

StringEncryption::StringEncryption() : hexEncoder(), hexDecoder() {
}

string
StringEncryption::encrypt(const string & passback, const string & key, const string & iv) {
    const byte * bk = reinterpret_cast<const byte *>(key.c_str());
    const byte * biv =  reinterpret_cast<const byte *>(iv.c_str());

    CFB_Mode<AES>::Encryption e(bk, AES::DEFAULT_KEYLENGTH, biv);
    StreamTransformationFilter stfE(e);
    stfE.Put((const byte *) passback.c_str(), passback.size());
    byte pEnc[passback.size()];
    stfE.Get(pEnc, passback.size());
    string cipher(pEnc, pEnc + passback.size());
    return addDigest(cipher);
}

string
StringEncryption::decrypt(const string & passback, const string & key, const string & iv) {
    const byte * bk = reinterpret_cast<const byte *>(key.c_str());
    const byte * biv =  reinterpret_cast<const byte *>(iv.c_str());

    string noDigest = removeDigest(hexDecode(passback));
    if (noDigest == "")
        return "";

    CFB_Mode<AES>::Decryption d(bk, AES::DEFAULT_KEYLENGTH, biv);
    StreamTransformationFilter stfD(d);
    stfD.Put((const byte *) noDigest.c_str(), noDigest.size());
    byte pDec[noDigest.size()];
    stfD.Get(pDec, noDigest.size());
    string recovered(pDec, pDec + noDigest.size());
    return recovered;
}

string StringEncryption::addDigest(const string & encrypted) {
    return hexEncode(encrypted) + hexEncode(digest(encrypted));
}

string StringEncryption::removeDigest(const string & digested) {
    if (digested.size() <= CRC32::DIGESTSIZE) {
        return "";
    }

    const string digestStr = digested.substr(digested.size() - CRC32::DIGESTSIZE); 
    const string encrypted = digested.substr(0, digested.size() - CRC32::DIGESTSIZE);

    if (digestStr == digest(encrypted))
        return encrypted;
    else
        return "";
}

string StringEncryption::digest(const string & encrypted) {
    int size = CRC32::DIGESTSIZE;
    byte digest[size];
    const byte * bencrypt = reinterpret_cast<const byte *>(encrypted.c_str());
    CRC32 crc;
    crc.CalculateDigest(digest, bencrypt, encrypted.size());
    string dig(digest, digest + size);
    return dig;
}

string
StringEncryption::hexEncode(const string & decoded) {
    hexEncoder.Put((const byte *)decoded.c_str(), decoded.size());
    size_t dLen = decoded.size() * 2;
    byte outBuf[dLen];
    hexEncoder.Get(outBuf, dLen);
    string encoded(outBuf, outBuf + dLen);
    return encoded;
}
    
string
StringEncryption::hexDecode(const string & encoded) {
    hexDecoder.Put((const byte *)encoded.c_str(), encoded.size());
    size_t eLen = encoded.size() / 2;
    byte outBuf[eLen];
    hexDecoder.Get(outBuf, eLen);
    string decoded(outBuf, outBuf + eLen);
    return decoded;
}

string
StringEncryption::generateKey() {
    AutoSeededRandomPool rnd;
    int size = AES::DEFAULT_KEYLENGTH;
    byte key[size];
    rnd.GenerateBlock(key, size);
    string sKey(key, key + size - 1);
    return hexEncode(sKey);
}

string
StringEncryption::generateIV() {
    AutoSeededRandomPool rnd;
    int size = AES::BLOCKSIZE;
    byte iv[size];
    rnd.GenerateBlock(iv, size);
    string sIv(iv, iv + size - 1);
    return hexEncode(sIv);
}

} // namespace Datacratic

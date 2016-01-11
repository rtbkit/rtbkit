/** fnv_hash.h                                 -*- C++ -*-
    jsbejeau, 17 Nov 2015
    Copyright (c) 2015 Datacratic.  All rights reserved.

    FNV Hash : Same implementation as Go
*/

namespace Datacratic {

/******************************************************************************/
/* FNV HASH                                                                   */
/******************************************************************************/

const uint32_t offset32 = 2166136261;
const uint64_t offset64 = 14695981039346656037ULL;
const uint64_t prime32  = 16777619 ;
const uint64_t prime64  = 1099511628211L;

uint32_t fnv_hash32(const std::string &str) {

    uint32_t hash{offset32};
    auto i = 0;

    while(str[i]) {
        hash *= prime32;
        hash ^= str[i];
        i++;
    }

    return hash;
}

uint32_t fnv_hash32a(const std::string &str) {

    uint32_t hash{offset32};
    auto i = 0;

    while(str[i]) {
        hash ^= str[i];
        hash *= prime32;
        i++;
    }

    return hash;
}

uint64_t fnv_hash64(const std::string &str) {

    uint64_t hash{offset64};
    auto i = 0;

    while(str[i]) {
        hash *= prime64;
        hash ^= str[i];
        i++;
    }

    return hash;
}

uint64_t fnv_hash64a(const std::string &str) {

    uint64_t hash{offset64};
    auto i = 0;

    while(str[i]) {
        hash ^= str[i];
        hash *= prime64;
        i++;
    }

    return hash;
}

} // namespace Datacratic


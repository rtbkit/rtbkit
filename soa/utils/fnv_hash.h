/** fnv_hash.h                                 -*- C++ -*-
    jsbejeau, 17 Nov 2015
    Copyright (c) 2015 Datacratic.  All rights reserved.

    FNV Hash : Same implementation as Go
*/

namespace Datacratic {

/******************************************************************************/
/* FNV HASH                                                                   */
/******************************************************************************/

static constexpr uint32_t offset32 = 2166136261;
static constexpr uint64_t offset64 = 14695981039346656037ULL;
static constexpr uint64_t prime32  = 16777619 ;
static constexpr uint64_t prime64  = 1099511628211L;

inline uint32_t fnv_hash32(const std::string &str) {

    uint32_t hash{offset32};
    auto i = 0;

    while(str[i]) {
        hash *= prime32;
        hash ^= str[i];
        i++;
    }

    return hash;
}

inline uint32_t fnv_hash32a(const std::string &str) {

    uint32_t hash{offset32};
    auto i = 0;

    while(str[i]) {
        hash ^= str[i];
        hash *= prime32;
        i++;
    }

    return hash;
}

inline uint64_t fnv_hash64(const std::string &str) {

    uint64_t hash{offset64};
    auto i = 0;

    while(str[i]) {
        hash *= prime64;
        hash ^= str[i];
        i++;
    }

    return hash;
}

inline uint64_t fnv_hash64a(const std::string &str) {

    uint64_t hash{offset64};
    auto i = 0;

    while(str[i]) {
        hash ^= str[i];
        hash *= prime64;
        i++;
    }

    return hash;
}

// constexpr versions
//
namespace impl {
    constexpr uint32_t fnv_hash32(const char head, const char *tail, uint32_t value)
    {
        return head == 0 ? value : fnv_hash32(tail[0], tail + 1, (value * prime32) ^ head);
    }

    constexpr uint32_t fnv_hash32a(const char head, const char *tail, uint32_t value)
    {
        return head == 0 ? value : fnv_hash32a(tail[0], tail + 1, (value ^ head) * prime32);
    }

    constexpr uint64_t fnv_hash64(const char head, const char *tail, uint64_t value)
    {
        return head == 0 ? value : fnv_hash64(tail[0], tail + 1, (value * prime64) ^ head);
    }

    constexpr uint64_t fnv_hash64a(const char head, const char *tail, uint64_t value)
    {
        return head == 0 ? value : fnv_hash64a(tail[0], tail + 1, (value ^ head) * prime64);
    }
}

constexpr uint32_t fnv_hash32(const char *str) {
    return impl::fnv_hash32(str[0], str + 1, offset32);
}

constexpr uint32_t fnv_hash32a(const char *str) {
    return impl::fnv_hash32a(str[0], str + 1, offset32);
}

constexpr uint64_t fnv_hash64(const char *str) {
    return impl::fnv_hash64(str[0], str + 1, offset64);
}

constexpr uint64_t fnv_hash64a(const char *str) {
    return impl::fnv_hash64a(str[0], str + 1, offset64);
}


} // namespace Datacratic


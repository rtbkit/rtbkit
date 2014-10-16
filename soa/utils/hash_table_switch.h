/** hash_table_switch.h                                 -*- C++ -*-
    Jean-Michel Bouchard, 23 May 2014
    Copyright (c) 2014 Datacratic.  All rights reserved.

    Hash Table Switch
*/

#pragma once

namespace Datacratic {

/******************************************************************************/
/* HASH TABLE SWITCH                                                          */
/******************************************************************************/

typedef uint64_t hash_t;

const hash_t prime = 0x100000001B3ull;
const hash_t basis = 0xBF29CE484222325ull;

hash_t hash(const std::string & str) {

    hash_t ret{basis};
    auto i = 0;

    while(str[i]) {
        ret ^= str[i];
        ret *= prime;
        i++;
    }

    return ret;
}

constexpr hash_t hash_compile_time(char const* str, hash_t last_value = basis) {

    return *str ? hash_compile_time(str+1, (*str ^ last_value) * prime) : last_value; 
}

} // namespace Datacratic


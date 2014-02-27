#pragma once

#include <city.h>
#include <sstream>

namespace variadic_hash_all{
void hashAll(std::stringstream& ss){}

template<typename first, typename... args>
void hashAll(std::stringstream& ss, first&& f, args&&... a){
    ss << std::forward<first>(f);
    hashAll(ss, std::forward<args>(a)...);
}
}//end of variadic_hash_all

template<typename... args>
uint64 hashAll(args&&... a){
    std::stringstream ss;
    variadic_hash_all::hashAll(ss, std::forward<args>(a)...);
    std::string str = ss.str();
    return CityHash64(str.c_str(), str.length());
}

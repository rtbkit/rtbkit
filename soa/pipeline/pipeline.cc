/* pipeline.cc
   Eric Robert, 26 February 2014
   Copyright (c) 2014 Datacratic Inc.  All rights reserved.

*/

std::string Environment::expandVariables(std::string const & text) const {
    size_t i = 0;
    std::string result;
    for(;;) {
        auto j = text.find("$(", i);
        result.append(text, i, j);

        if(j == std::string::npos) {
            break;
        }

        i = j + 2;
        j = text.find(")", i);

        if(j == std::string::npos) {
            break;
        }

        std::string key(text, i, j - i);
        auto k = keys.find(key);
        if(keys.end() != k) {
            result += k->second;
        }

        i = j + 1;
    }

    return result;
}

void Environment::set(std::string key, std::string value) {
    keys[key] = value;
}

EnvironmentDescription::
EnvironmentDescription() {
}

Pipeline::Pipeline() :
    Blocks(this),
    environment(this, "environment") {
}

